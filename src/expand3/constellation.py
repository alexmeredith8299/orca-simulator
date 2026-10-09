"""Propagate all six vehicles and compute the M1 environment products at each control tick."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from expand3.params import Params
from expand3.timebase import SeedDraws
from expand3.vehicle.orbit import Forces

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class ConstellationRun:
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
    raise NotImplementedError
