"""Attitude: scalar-first body-to-inertial quaternions, kinematics, rigid-body dynamics (M2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from expand3.params import ParamView
from expand3.vehicle.wheels import WheelArray

F64 = NDArray[np.float64]


def quat_normalize(q: F64) -> F64:
    raise NotImplementedError


def quat_mul(p: F64, q: F64) -> F64:
    raise NotImplementedError


def quat_conj(q: F64) -> F64:
    raise NotImplementedError


def quat_to_dcm(q: F64) -> F64:
    raise NotImplementedError


def dcm_to_quat(m: F64) -> F64:
    raise NotImplementedError


def quat_rotate(q: F64, v: F64) -> F64:
    raise NotImplementedError


def quat_from_axis_angle(axis: F64, angle_rad: float) -> F64:
    raise NotImplementedError


def quat_angle_rad(q1: F64, q2: F64) -> float:
    raise NotImplementedError


def quat_derivative(q: F64, omega: F64) -> F64:
    raise NotImplementedError


def inertia(view: ParamView) -> F64:
    raise NotImplementedError


def gravity_gradient_torque(inertia_b: F64, r_eci: F64, q: F64, mu_km3_s2: float) -> F64:
    raise NotImplementedError


@dataclass(frozen=True)
class AttitudeState:
    q: F64
    omega: F64
    h: F64


def total_momentum_inertial(inertia_b: F64, wheels: WheelArray, state: AttitudeState) -> F64:
    raise NotImplementedError


def step(
    inertia_b: F64,
    wheels: WheelArray,
    state: AttitudeState,
    tau_d: F64,
    dt: float,
    r_eci: F64 | None = None,
    mu_km3_s2: float | None = None,
) -> tuple[AttitudeState, F64, F64]:
    raise NotImplementedError
