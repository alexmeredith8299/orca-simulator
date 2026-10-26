"""Exogenous context features built from measured signals only.

The learned residual models predict a signal from context that the fault under test does not
touch. The context is the sunlit state, inferred from the string powers; the time since the last
eclipse entry and exit; the operation gates (pass, take, slew, burn, SAA, shed); and the delivered
heater states. Each is also passed through first-order filters of several time constants, so a
linear model sees the slow thermal memory without lagged states of the signal itself. No node
temperature is a feature of another node's model: F7 cools its neighbours and they would absorb it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter

from expand3.detect.config import DetectConfig
from expand3.detect.data import SeedData
from expand3.params import load_params

F64 = NDArray[np.float64]
GATES = ("gate_pass", "gate_take", "gate_slew", "gate_burn", "gate_saa", "gate_shed")
HEATERS = ("heater_battery_delivered", "heater_payload_delivered")
N_STRINGS = 4
INIT_TICKS = 400


def bus_of(params_dir: str) -> dict[str, str]:
    return {str(k): str(v) for k, v in load_params(params_dir).scenario.bus_of.items()}


def sunlit(m: dict[str, F64], cfg: DetectConfig) -> F64:
    """1.0 where the strings produce: their measured mean against its own high quantile."""
    p = np.stack([m[f"string_p_{i + 1}_W_meas"] for i in range(N_STRINGS)])
    p = p[~np.all(np.isnan(p), axis=1)]
    mean = p.mean(axis=0)
    ref = np.nanquantile(mean, cfg.f2_sunlit_quantile)
    return (mean > cfg.f2_sunlit_fraction * ref).astype(np.float64)


def ewma(x: F64, tau_s: float, dt_s: float) -> F64:
    """Causal first-order filter, started at the mean of the first ticks."""
    a = 1.0 - np.exp(-dt_s / tau_s)
    x0 = float(np.mean(x[:INIT_TICKS]))
    y, _ = lfilter([a], [1.0, -(1.0 - a)], x, zi=[(1.0 - a) * x0])
    return np.asarray(y, dtype=np.float64)


def since_change(flag: F64, dt_s: float, cap_s: float, target: float) -> F64:
    """Seconds since `flag` last became `target` (capped), zero while it is not `target`."""
    out = np.zeros(flag.size)
    last = -np.inf
    prev = flag[0]
    for i, v in enumerate(flag):
        if v == target and prev != target:
            last = i * dt_s
        out[i] = min(i * dt_s - last, cap_s) if v == target else 0.0
        prev = v
    return out


def context(d: SeedData, sat: str, cfg: DetectConfig) -> tuple[F64, F64, list[str]]:
    """(t_s, X, names) of the exogenous features of one vehicle at every tick."""
    m = d.measurements(sat)
    t = m["t_s"]
    dt = float(np.median(np.diff(t)))
    sun = sunlit(m, cfg)
    cols: dict[str, F64] = {"sunlit": sun}
    for tau in cfg.ctx_taus_s:
        cols[f"sunlit_ewma_{tau:g}"] = ewma(sun, tau, dt)
    cap = cfg.ctx_event_cap_s
    cols["since_eclipse_entry"] = since_change(sun, dt, cap, 0.0) / cap
    cols["since_eclipse_exit"] = since_change(sun, dt, cap, 1.0) / cap
    for g in GATES:
        cols[g] = m[g]
    maneuver = np.maximum.reduce([m["gate_take"], m["gate_slew"]])
    for tau in cfg.ctx_gate_taus_s:
        cols[f"pass_ewma_{tau:g}"] = ewma(m["gate_pass"], tau, dt)
        cols[f"burn_ewma_{tau:g}"] = ewma(m["gate_burn"], tau, dt)
        cols[f"maneuver_ewma_{tau:g}"] = ewma(maneuver, tau, dt)
    for h in HEATERS:
        cols[h] = m[h]
        cols[f"{h}_ewma"] = ewma(m[h], cfg.ctx_gate_taus_s[1], dt)
    names = list(cols)
    return t, np.stack([cols[n] for n in names], axis=1), names


def node_columns(m: dict[str, Any]) -> list[str]:
    """Thermistor columns that carry data on this vehicle (Bus B lacks one wing)."""
    return [c for c in m if c.startswith("T_") and c.endswith("_K_meas")
            and not np.all(np.isnan(m[c]))]
