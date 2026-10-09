"""Monopropellant thruster: commanded and delivered thrust, acceleration, tank drain (M3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from expand3.params import ParamView

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class ThrustStep:
    F_cmd_N: float
    F_N: float
    a_thr_km_s2: F64
    mdot_kg_s: float
    dv_m_s: float
    dm_kg: float


def dry_mass_kg(view: ParamView) -> float:
    raise NotImplementedError


def thrust_step(
    view: ParamView, a_cmd_m_s2: F64, mass_kg: float, propellant_kg: float, q: F64, dt: float
) -> ThrustStep:
    raise NotImplementedError
