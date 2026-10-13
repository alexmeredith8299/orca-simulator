"""SAA and C&DH: IGRF-14 field magnitude, SAA flag, edge computer, Poisson upsets (M6).

Spec: Model: sensors, crosslink, SAA, imager (SAA and C&DH, F9); Parameters > Links, SAA, C&DH.
|B| comes from IGRF-14 (ppigrf) at the vehicle; the vehicle is inside the SAA while |B| is below
the boundary. The edge computer is off inside and returns its reboot time after the vehicle
leaves. C&DH upsets are Poisson with rate lambda_out everywhere plus lambda_in inside the SAA; a
fixed fraction of upsets reset the computer, and both are only counted.

Conventions (CLAUDE.md Known gaps): the field uses ppigrf's geocentric routine on the ECEF
position, with the coefficients evaluated at the run epoch for the whole run. Edge-computer state
and upsets are on the control-tick grid. The exit time is the first tick outside the SAA, so the
computer is off for the ticks before exit + reboot. Upsets per tick are Poisson(lambda dt), with
lambda_in read through ParamView every tick (F9), and resets are Binomial(upsets, fraction).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import ppigrf
from numpy.typing import NDArray

from expand3.params import ParamView
from expand3.timebase import rng

F64 = NDArray[np.float64]
B = NDArray[np.bool_]
I64 = NDArray[np.int64]


@dataclass
class SaaHistory:
    """Per tick: field magnitude [nT], SAA flag, edge computer off, upset and reset counts."""

    B_nT: F64
    in_saa: B
    edge_off: B
    upsets: I64
    resets: I64


def field_magnitude_nt(r_ecef_km: F64, epoch_utc: datetime) -> F64:
    """|B| [nT] from IGRF-14 at ECEF positions (..., 3) [km]."""
    r_ecef = np.atleast_2d(np.asarray(r_ecef_km, dtype=np.float64))
    rad = np.linalg.norm(r_ecef, axis=-1)
    colat = np.rad2deg(np.arccos(np.clip(r_ecef[:, 2] / rad, -1.0, 1.0)))
    lon = np.rad2deg(np.arctan2(r_ecef[:, 1], r_ecef[:, 0]))
    date = epoch_utc.astimezone(UTC).replace(tzinfo=None)
    br, bt, bp = ppigrf.igrf_gc(rad, colat, lon, date)
    return np.asarray(np.sqrt(br**2 + bt**2 + bp**2), dtype=np.float64).reshape(-1)


def edge_computer_off(t_s: F64, in_saa: B, reboot_s: float) -> B:
    """Off while inside the SAA and until reboot_s after the first tick outside."""
    off = np.zeros(len(t_s), dtype=bool)
    exit_t = -np.inf
    prev = False
    for n, inside in enumerate(in_saa):
        if inside:
            off[n] = True
        else:
            if prev:
                exit_t = float(t_s[n])
            off[n] = float(t_s[n]) < exit_t + reboot_s
        prev = bool(inside)
    return off


def upsets(
    rng: np.random.Generator, view: ParamView, in_saa: B, dt: float, t_s: F64 | None = None
) -> tuple[I64, I64]:
    """Spec: C&DH upsets (F9). Poisson counts per tick and the resets among them; lambda_in is
    read at each tick time (t_s, default every dt from 0)."""
    lam_out = view.get("cdh_upsets.lambda_out_per_s")
    times = np.asarray(np.arange(len(in_saa)) * dt if t_s is None else t_s, dtype=np.float64)
    lam_in = view.get_series("cdh_upsets.lambda_in_per_s", times)
    lam = lam_out + lam_in * np.asarray(in_saa, dtype=np.float64)
    counts = rng.poisson(lam * dt).astype(np.int64)
    resets = rng.binomial(counts, view.get("cdh_upsets.reset_fraction")).astype(np.int64)
    return counts, resets


def saa_history(
    seed: int, view: ParamView, t_s: F64, r_ecef_km: F64, epoch_utc: datetime
) -> SaaHistory:
    """SAA, edge computer and upsets for one vehicle over the run (stream `upsets.<sat>`)."""
    b = field_magnitude_nt(r_ecef_km, epoch_utc)
    in_saa = b < view.get("saa.boundary_nT")
    dt = float(t_s[1] - t_s[0]) if len(t_s) > 1 else view.get("scenario.control_tick_s")
    counts, resets = upsets(rng(seed, f"upsets.{view.sat_id}"), view, in_saa, dt, t_s)
    return SaaHistory(b, in_saa, edge_computer_off(t_s, in_saa, view.get("edge_computer_reboot_s")),
                      counts, resets)
