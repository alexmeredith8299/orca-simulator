"""Translational dynamics: two-body, J2, exponential drag, thruster; fixed-step RK4 (M1)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from expand3.params import Params
from expand3.timebase import SeedDraws

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class Forces:
    j2: bool = True
    drag: bool = True


@dataclass(frozen=True)
class Elements:
    a_km: F64
    e: F64
    i_rad: F64
    raan_rad: F64
    argp_rad: F64
    nu_rad: F64
    u_rad: F64


@dataclass(frozen=True)
class OrbitModel:
    mu_km3_s2: float
    J2: float
    R_E_km: float
    omega_E_rad_s: float
    pole: F64
    ballistic_m2_kg: F64
    rho0_kg_m3: float
    h0_km: float
    H_km: float
    forces: Forces = field(default_factory=Forces)

    @classmethod
    def from_params(
        cls, params: Params, sats: list[str], draws: SeedDraws, pole: F64, forces: Forces
    ) -> OrbitModel:
        raise NotImplementedError

    def accel(self, r: F64, v: F64, a_thr: F64 | None = None) -> F64:
        raise NotImplementedError

    def deriv(self, y: F64, a_thr: F64 | None = None) -> F64:
        raise NotImplementedError


def rk4_step(f: Callable[[F64], F64], y: F64, dt: float) -> F64:
    raise NotImplementedError


def propagate(
    model: OrbitModel, y0: F64, step_s: float, n_steps: int, record_every: int
) -> F64:
    raise NotImplementedError


def elements_to_rv(
    a_km: float, e: float, i_rad: float, raan_rad: float, argp_rad: float, nu_rad: float,
    mu_km3_s2: float, basis: F64 | None = None,
) -> tuple[F64, F64]:
    raise NotImplementedError


def rv_to_elements(r: F64, v: F64, mu_km3_s2: float, basis: F64 | None = None) -> Elements:
    raise NotImplementedError


def kepler_propagate(r0: F64, v0: F64, dt_s: F64, mu_km3_s2: float) -> tuple[F64, F64]:
    raise NotImplementedError


def equatorial_basis(pole: F64, x_hint: F64) -> F64:
    raise NotImplementedError


def initial_states(params: Params, sats: list[str], epoch_utc: datetime) -> F64:
    raise NotImplementedError
