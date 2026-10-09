"""Time-critical imaging takes: take time, target, slews in and out (M3)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from expand3.params import ParamView

F64 = NDArray[np.float64]


@dataclass
class TakeRecord:
    sat: str
    t_take_s: float
    t_end_s: float
    target_ecef_km: F64
    target_lat_deg: float
    target_lon_deg: float
    slew_in_start_s: float = 0.0
    slew_out_end_s: float = 0.0
    tick_times_s: list[float] = field(default_factory=list)
    sun_elevation_deg: list[float] = field(default_factory=list)
    off_nadir_deg: list[float] = field(default_factory=list)
    boresight_error_rad: list[float] = field(default_factory=list)


def subsatellite_point(r_ecef_km: F64) -> tuple[F64, F64, F64, F64]:
    raise NotImplementedError


def sun_elevation_deg(point_ecef_km: F64, up: F64, sun_ecef_km: F64) -> F64:
    raise NotImplementedError


def off_nadir_deg(r_eci_km: F64, target_eci_km: F64) -> F64:
    raise NotImplementedError


def find_take_tick(
    view: ParamView, t_s: F64, r_ecef_km: F64, sun_ecef_km: F64
) -> int | None:
    raise NotImplementedError
