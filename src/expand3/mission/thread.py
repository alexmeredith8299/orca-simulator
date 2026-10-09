"""The mission thread end to end: six vehicles, burns, takes, slews, pointing (M3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from expand3.mission.burns import BurnRecord
from expand3.mission.slew import SlewResult
from expand3.mission.takes import TakeRecord
from expand3.params import Params
from expand3.timebase import SeedDraws

F64 = NDArray[np.float64]
B = NDArray[np.bool_]


@dataclass(frozen=True)
class MissionRun:
    """Per-tick truth, indexed [tick, vehicle, ...]."""

    draws: SeedDraws
    sats: list[str]
    station_names: list[str]
    t_s: F64
    r_km: F64
    v_km_s: F64
    q: F64
    omega: F64
    h_wheel: F64
    mass_kg: F64
    propellant_kg: F64
    a_cmd_m_s2: F64
    dv_m_s: F64
    gate_pass: B
    gate_take: B
    gate_slew: B
    gate_burn: B
    eci_to_ecef: F64
    sun_unit: F64
    sun_dist_km: F64
    shadow: F64
    elevation_deg: F64
    burns: dict[str, BurnRecord]
    takes: dict[str, TakeRecord]
    slews: dict[str, list[SlewResult]]
    wall_s: float


def run_mission(
    params: Params,
    seed: int,
    duration_s: float | None = None,
    draws: SeedDraws | None = None,
    overrides: dict[str, dict[str, float]] | None = None,
) -> MissionRun:
    raise NotImplementedError
