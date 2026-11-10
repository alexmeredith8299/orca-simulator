"""Multi-satellite Monte Carlo dropout forecasters (extra `detect-dl`).

The single-vehicle detectors in `mcdrop.py` predict a vehicle's signal from its own context. Here
the dropout LSTM sees the context features of all six vehicles and one network is shared by all of
them (a bus flag per slot tells Bus A from Bus B). The targets, the stochastic passes, the
aleatoric term and the GLR are those of `mcdrop.py`, so the two differ only in what goes in.

Each vehicle gets its own view of the constellation, in a fixed slot order: itself, its cluster
mates, then the other cluster. A slot holds that vehicle's context features, how old they are and
its bus flag. Two variants:

- `mcdm_*`: every vehicle's features arrive at once, the upper bound on what sharing can give.
- `mcdm_*_comms`: what the vehicle could really hold. Its own features are current. A cluster
  mate's features arrive over the crosslink on the ticks the message was delivered (F8 drops make
  gaps) and are held between them. The other cluster's features go through a ground station
  (spec: Earth blocks line of sight between the clusters), store and forward, see `comms.py`. A
  slot not yet received is zero with its age at the cap.

The measured inputs of the F1 model (current, SoC estimate and battery temperature) are the
vehicle's own and always current. No other vehicle's measured signal is an input.

Detectors read batch outputs only, and take the clusters and buses from the parameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from expand3.detect.base import Detector
from expand3.detect.comms import hold, relay_source
from expand3.detect.config import DetectConfig
from expand3.detect.data import SeedData
from expand3.detect.features import context, node_columns
from expand3.detect.learned import _voltage_inputs
from expand3.detect.mcdrop import McdEps, McdThermal
from expand3.params import load_params

F64 = NDArray[np.float64]
F32 = NDArray[np.float32]
IntArr = NDArray[np.int64]
BoolArr = NDArray[np.bool_]
ALL = "all"  # the one shared model


def slot_order(sat: str, clusters: list[list[str]]) -> list[str]:
    """The vehicle itself, its cluster mates, then the other clusters, each in cluster order."""
    own = next(c for c in clusters if sat in c)
    return [sat, *[s for s in own if s != sat],
            *[s for c in clusters if c is not own for s in c]]


def assemble(feats: dict[str, F64], idx: dict[str, IntArr], age_s: dict[str, F64],
             order: list[str], bus_flag: dict[str, float], cap_s: float) -> F64:
    """(n, slots x (F + 2)) input: per slot the held features, age / cap (at most 1), bus flag.

    `idx[s]` is the tick of `s` held at each tick (-1 before anything arrived, which reads as zero
    features at the cap). A vehicle in `order` without data is a slot of zeros at the cap.
    """
    n, width = next(iter(feats.values())).shape
    cols: list[F64] = []
    for s in order:
        if s in feats:
            held = np.where((idx[s] >= 0)[:, None], feats[s][np.maximum(idx[s], 0)], 0.0)
            age = np.minimum(age_s[s], cap_s) / cap_s
            bus = bus_flag[s]
        else:
            held, age, bus = np.zeros((n, width)), np.ones(n), 0.0
        cols += [held, age[:, None], np.full((n, 1), bus)]
    return np.concatenate(cols, axis=1)


class _MultiView:
    """Builds a vehicle's view of the constellation; `comms` picks the variant."""

    comms: bool
    cfg: DetectConfig
    params_dir: str | None

    def _scenario(self) -> Any:
        assert self.params_dir is not None
        return load_params(self.params_dir).scenario

    def _buses(self) -> dict[str, str]:
        return {s: ALL for c in self._scenario().clusters for s in c}

    def _seed(self, d: SeedData) -> dict[str, Any]:
        """Per seed: the tick grid, every vehicle's context features, cluster pass flags and
        delivered crosslink messages. Read once and kept on the seed."""
        cache: dict[str, Any] = d.__dict__.setdefault("_mcdm_cache", {})
        if "t" not in cache:
            feats: dict[str, F64] = {}
            t: F64 | None = None
            for s in d.sat_ids:
                ts, x, _ = context(d, s, self.cfg)
                if t is None:
                    t = ts
                elif not np.array_equal(t, ts):
                    raise ValueError(f"{s} is on a different tick grid than the other vehicles")
                feats[s] = x
            assert t is not None
            cache["t"], cache["feats"] = t, feats
            clusters = self._scenario().clusters
            cache["pass"] = [np.any([d.measurements(s)["gate_pass"] > 0.5
                                     for s in c if s in feats], axis=0) for c in clusters]
            link = d.table("crosslink", ["t_s", "sender", "receiver", "delivered"])
            sender, receiver = (np.asarray(link.column(c)) for c in ("sender", "receiver"))
            t_link = np.asarray(link.column("t_s"), dtype=np.float64)
            ok = np.asarray(link.column("delivered"), dtype=bool)
            delivered: dict[tuple[str, str], BoolArr] = {}
            for a in feats:
                for b in feats:
                    sel = (sender == a) & (receiver == b)
                    mask = np.zeros(len(t), dtype=bool)
                    mask[np.searchsorted(t, t_link[sel])] = ok[sel]
                    delivered[(a, b)] = mask
            cache["delivered"] = delivered
        return cache

    def inputs(self, d: SeedData, sat: str) -> tuple[F64, F64]:
        """(t_s, X) of one vehicle's view at every tick."""
        sc = self._scenario()
        seed = self._seed(d)
        t, feats = seed["t"], seed["feats"]
        dt = float(np.median(np.diff(t)))
        ticks = np.arange(len(t), dtype=np.int64)
        cluster_of = {s: k for k, c in enumerate(sc.clusters) for s in c}
        idx: dict[str, IntArr] = {}
        age: dict[str, F64] = {}
        for s in slot_order(sat, sc.clusters):
            if s not in feats:
                continue
            if s == sat or not self.comms:
                idx[s], age[s] = ticks, np.zeros(len(t))
            elif cluster_of[s] == cluster_of[sat]:
                idx[s], age[s] = hold(seed["delivered"][(s, sat)], dt)
            else:
                idx[s] = relay_source(seed["pass"][cluster_of[s]], seed["pass"][cluster_of[sat]])
                age[s] = np.where(idx[s] >= 0, (ticks - idx[s]) * dt, np.inf)
        bus = {s: float(b == "B") for s, b in sc.bus_of.items()}
        x = assemble(feats, idx, age, slot_order(sat, sc.clusters), bus, self.cfg.mcdm_age_cap_s)
        return t, x


class McdMultiThermal(_MultiView, McdThermal):
    """F7 from every vehicle's context: per-node temperature misses, scored by the worst node."""

    comms = False
    name = "mcdm_thermal"

    def series(self, d: SeedData, sat: str) -> tuple[F64, F32, dict[str, F64], F64]:
        step = self.cfg.mcd_stride
        t, x = self.inputs(d, sat)
        m = d.measurements(sat)
        y = {c: m[c][::step] for c in node_columns(m)}
        return t[::step], x[::step].astype(np.float32), y, np.zeros(len(t[::step]))


class McdMultiEps(_MultiView, McdEps):
    """F1 from every vehicle's context and the vehicle's own current, SoC estimate and battery
    temperature: the slope of the pack-voltage miss on the measured current."""

    comms = False
    name = "mcdm_eps"

    def series(self, d: SeedData, sat: str) -> tuple[F64, F32, dict[str, F64], F64]:
        step = self.cfg.mcd_stride
        t, x = self.inputs(d, sat)
        m = d.measurements(sat)
        inputs = np.concatenate([_voltage_inputs(m), x], axis=1)[::step].astype(np.float32)
        return (t[::step], np.nan_to_num(inputs), {"pack_v_V_meas": m["pack_v_V_meas"][::step]},
                m["pack_i_A_meas"][::step])


class McdMultiThermalComms(McdMultiThermal):
    comms = True
    name = "mcdm_thermal_comms"


class McdMultiEpsComms(McdMultiEps):
    comms = True
    name = "mcdm_eps_comms"


MCDM: dict[str, type[Detector]] = {c.name: c for c in (
    McdMultiThermal, McdMultiEps, McdMultiThermalComms, McdMultiEpsComms)}
