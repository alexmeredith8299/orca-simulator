"""Propagate all six vehicles and compute the M1 environment products at each control tick.

Spec: Scenario; Model: orbit and attitude; Parameters > Scenario and environment. States are
recorded at control ticks. Frames, Sun and shadow are evaluated at the same ticks, so no
interpolation is needed in M1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from expand3.env.eclipse import shadow_fraction
from expand3.env.frames import eci_to_ecef
from expand3.env.ground import elevation_deg, pass_intervals, stations
from expand3.env.sun import sun_eci
from expand3.params import Params, ParamView
from expand3.timebase import Clock, SeedDraws
from expand3.vehicle.orbit import Forces, OrbitModel, initial_states, propagate

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class ConstellationRun:
    """Arrays are indexed [tick, vehicle, ...]; elevation and visible are [tick, vehicle,
    station]."""

    sats: list[str]
    station_names: list[str]
    t_s: F64
    pole_eci: F64
    r_km: F64
    v_km_s: F64
    eci_to_ecef: F64
    sun_unit: F64
    sun_dist_km: F64
    shadow: F64
    elevation_deg: F64
    visible: NDArray[np.bool_]
    passes: dict[tuple[str, str], list[tuple[float, float]]]
    wall_s: float


def propagate_constellation(
    params: Params,
    draws: SeedDraws,
    forces: Forces | None = None,
    duration_s: float | None = None,
    sats: list[str] | None = None,
) -> ConstellationRun:
    """Propagate `sats` (default all six) from the seed's epoch for `duration_s` (default the
    seed's run length) and evaluate Sun, shadow and ground-station visibility at every tick."""
    start = time.perf_counter()
    sats = list(params.sat_ids) if sats is None else sats
    view = ParamView(params, sats[0])
    clock = Clock.from_view(view, draws.run_length_s if duration_s is None else duration_s)
    t = clock.tick_times_s()

    rot = eci_to_ecef(draws.epoch_utc, t)
    pole = np.asarray(rot[0, 2], dtype=np.float64)
    model = OrbitModel.from_params(params, sats, draws, pole, forces or Forces())
    y0 = initial_states(params, sats, draws.epoch_utc)
    states = propagate(model, y0, clock.step_s, (clock.n_ticks - 1) * clock.steps_per_tick,
                       clock.steps_per_tick)
    r, v = states[..., :3], states[..., 3:]

    sun_u, sun_d = sun_eci(draws.epoch_utc, t)
    shadow = shadow_fraction(
        r, sun_u[:, None, :], sun_d[:, None],
        view.get("scenario.constants.R_E_km"), view.get("scenario.constants.R_sun_km"),
    )

    r_ecef = np.einsum("nij,nkj->nki", rot, r)
    sites = stations(view)
    elev = np.stack([elevation_deg(r_ecef, s) for s in sites], axis=-1)
    visible = elev >= view.get("scenario.elevation_mask_deg")
    passes = {
        (sat, site.name): pass_intervals(t, visible[:, k, j])
        for k, sat in enumerate(sats)
        for j, site in enumerate(sites)
    }
    return ConstellationRun(
        sats=sats,
        station_names=[s.name for s in sites],
        t_s=t,
        pole_eci=pole,
        r_km=r,
        v_km_s=v,
        eci_to_ecef=rot,
        sun_unit=sun_u,
        sun_dist_km=sun_d,
        shadow=shadow,
        elevation_deg=elev,
        visible=visible,
        passes=passes,
        wall_s=time.perf_counter() - start,
    )
