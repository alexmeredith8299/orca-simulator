"""Intra-cluster crosslink: range, line of sight, drops, receive noise, partner-state age (M6).

Spec: Model: sensors, crosslink, SAA, imager (Crosslink, F8); Parameters > Links. Inside each
cluster, a member within crosslink range and in line of sight sends its state every control
tick; the receiver gets r and v with Gaussian noise. Under F8 each message drops independently
with probability p.

Conventions (CLAUDE.md Known gaps): line of sight means the segment between the two vehicles
clears a spherical Earth of radius R_E. Range uses the sender's crosslink range. p is read
through ParamView every tick for both ends of a link; a message drops with probability
1 - (1 - p_sender)(1 - p_receiver), so a vehicle's margin loss affects what it sends and what it
receives. Draws come from `drop.crosslink.<sender>.<receiver>` and
`noise.crosslink.<sender>.<receiver>` for every tick, sent or not, so the streams never shift.

Received signal strength (not in the spec): sender power plus both antenna gains minus free-space
loss at the true range, minus the sender's `sensor_faults.radio.tx_loss_dB` (tx_fade), plus a
reading error from `noise.rssi.<sender>.<receiver>`. Only the receiver senses it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from expand3.params import Params, ParamView
from expand3.timebase import rng
from expand3.units import HZ_PER_GHZ, M_PER_KM, SPEED_OF_LIGHT_M_S

F64 = NDArray[np.float64]
B = NDArray[np.bool_]


@dataclass
class CrosslinkLog:
    """Per tick and ordered pair (sender, receiver): sent, delivered and the received state (NaN
    unless delivered); receipts per receiver per tick; age [s] of each partner's latest state at
    each receiver (inf before the first receipt). rssi_dBm is the received signal strength of each
    delivered message (NaN otherwise)."""

    pairs: list[tuple[str, str]]
    t_s: F64
    sent: B
    delivered: B
    r_meas_km: F64
    v_meas_km_s: F64
    receipts: dict[str, NDArray[np.int64]] = field(default_factory=dict)
    age_s: dict[tuple[str, str], F64] = field(default_factory=dict)
    rssi_dBm: F64 = field(default_factory=lambda: np.zeros((0, 0)))


def line_of_sight(r1_km: F64, r2_km: F64, R_E_km: float) -> B:
    """True where the segment r1-r2 stays outside a sphere of radius R_E."""
    r1 = np.asarray(r1_km, dtype=np.float64)
    d = np.asarray(r2_km, dtype=np.float64) - r1
    s = np.clip(-np.sum(r1 * d, axis=-1) / np.sum(d * d, axis=-1), 0.0, 1.0)
    closest = r1 + s[..., None] * d
    out: B = np.linalg.norm(closest, axis=-1) > R_E_km
    return out


def received_power_dbm(sender: ParamView, range_km: F64, t_s: F64) -> F64:
    """Received power at the true range: Friis free-space loss between omni antennas, less the
    sender's transmit loss (tx_fade)."""
    wavelength_m = SPEED_OF_LIGHT_M_S / (sender.get("link_budget.frequency_GHz") * HZ_PER_GHZ)
    fspl_dB = 20.0 * np.log10(4.0 * np.pi * range_km * M_PER_KM / wavelength_m)
    out: F64 = (sender.get("link_budget.tx_power_dBm") + sender.get("link_budget.tx_gain_dBi")
                + sender.get("link_budget.rx_gain_dBi") - fspl_dB
                - sender.get_series("sensor_faults.radio.tx_loss_dB", t_s))
    return out


def crosslink(
    seed: int, params: Params, views: dict[str, ParamView], sats: list[str], t_s: F64,
    r_km: F64, v_km_s: F64,
) -> CrosslinkLog:
    """Messages for every ordered pair in each cluster. r_km, v_km_s are (tick, vehicle, 3)."""
    pairs = [(a, b) for cluster in params.scenario.clusters for a in cluster for b in cluster
             if a != b and a in sats and b in sats]
    n_t, n_p = len(t_s), len(pairs)
    sent = np.zeros((n_t, n_p), dtype=bool)
    delivered = np.zeros((n_t, n_p), dtype=bool)
    r_meas = np.full((n_t, n_p, 3), np.nan)
    v_meas = np.full((n_t, n_p, 3), np.nan)
    rssi = np.full((n_t, n_p), np.nan)
    r_e = views[sats[0]].get("scenario.constants.R_E_km")
    p_of = {s: views[s].get_series("crosslink.drop_probability", t_s) for s in sats}
    for j, (a, b) in enumerate(pairs):
        ka, kb = sats.index(a), sats.index(b)
        va = views[a]
        dist = np.linalg.norm(r_km[:, ka] - r_km[:, kb], axis=-1)
        sent[:, j] = (dist <= va.get("crosslink.range_km")) & line_of_sight(
            r_km[:, ka], r_km[:, kb], r_e)
        p = 1.0 - (1.0 - p_of[a]) * (1.0 - p_of[b])
        drop = rng(seed, f"drop.crosslink.{a}.{b}").random(n_t) < p
        delivered[:, j] = sent[:, j] & ~drop
        noise = rng(seed, f"noise.crosslink.{a}.{b}")
        dr = noise.normal(0.0, va.get("crosslink.rx_noise_pos_m") / M_PER_KM, (n_t, 3))
        dv = noise.normal(0.0, va.get("crosslink.rx_noise_vel_m_s") / M_PER_KM, (n_t, 3))
        ok = delivered[:, j]
        reading = rng(seed, f"noise.rssi.{a}.{b}").normal(
            0.0, va.get("link_budget.rssi_sigma_dB"), n_t)
        rssi[ok, j] = received_power_dbm(va, dist, t_s)[ok] + reading[ok]
        r_meas[ok, j] = r_km[ok, ka] + dr[ok]
        v_meas[ok, j] = v_km_s[ok, ka] + dv[ok]
    receipts = {s: delivered[:, [j for j, p in enumerate(pairs) if p[1] == s]].sum(axis=1)
                .astype(np.int64) for s in sats}
    age: dict[tuple[str, str], F64] = {}
    for j, (a, b) in enumerate(pairs):
        out = np.full(n_t, np.inf)
        last = -np.inf
        for n in range(n_t):
            if delivered[n, j]:
                last = float(t_s[n])
            out[n] = float(t_s[n]) - last
        age[(b, a)] = out
    return CrosslinkLog(pairs, np.asarray(t_s), sent, delivered, r_meas, v_meas, receipts, age,
                        rssi)
