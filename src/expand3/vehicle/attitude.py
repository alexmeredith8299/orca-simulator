"""Attitude: scalar-first body-to-inertial quaternions, kinematics, rigid-body dynamics (M2).

Spec: Model: orbit and attitude. q maps body vectors to inertial, v_I = q (x) [0, v_B] (x) q*.

    q_dot = 1/2 q (x) [0, omega]
    I omega_dot = tau_gg + tau_w - omega x (I omega + A h)
    tau_gg = 3 mu / |r|^3  n_hat x I n_hat   (n_hat = nadir in body)

The rotational equations are integrated only during wheel slews, where the spec allows dropping
tau_gg. It is implemented but off unless a position is passed to `step`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from expand3.params import ParamView
from expand3.vec3 import cross3
from expand3.vehicle.wheels import WheelArray

F64 = NDArray[np.float64]


def quat_normalize(q: F64) -> F64:
    out: F64 = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return out


def quat_mul(p: F64, q: F64) -> F64:
    """Hamilton product p (x) q."""
    pw, pv = p[0], p[1:]
    qw, qv = q[0], q[1:]
    return np.concatenate([[pw * qw - pv @ qv], pw * qv + qw * pv + cross3(pv, qv)])


def quat_conj(q: F64) -> F64:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_dcm(q: F64) -> F64:
    """Rotation matrix R(q) with v_I = R v_B."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def dcm_to_quat(m: F64) -> F64:
    """Quaternion (scalar first, w >= 0) of a rotation matrix, Shepperd's method."""
    tr = np.trace(m)
    cand = np.array([tr, m[0, 0], m[1, 1], m[2, 2]])
    i = int(np.argmax(cand))
    if i == 0:
        w = np.sqrt(1 + tr) / 2
        q = np.array([w, (m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w),
                      (m[1, 0] - m[0, 1]) / (4 * w)])
    elif i == 1:
        x = np.sqrt(1 + 2 * m[0, 0] - tr) / 2
        q = np.array([(m[2, 1] - m[1, 2]) / (4 * x), x, (m[0, 1] + m[1, 0]) / (4 * x),
                      (m[0, 2] + m[2, 0]) / (4 * x)])
    elif i == 2:
        y = np.sqrt(1 + 2 * m[1, 1] - tr) / 2
        q = np.array([(m[0, 2] - m[2, 0]) / (4 * y), (m[0, 1] + m[1, 0]) / (4 * y), y,
                      (m[1, 2] + m[2, 1]) / (4 * y)])
    else:
        z = np.sqrt(1 + 2 * m[2, 2] - tr) / 2
        q = np.array([(m[1, 0] - m[0, 1]) / (4 * z), (m[0, 2] + m[2, 0]) / (4 * z),
                      (m[1, 2] + m[2, 1]) / (4 * z), z])
    q = quat_normalize(q)
    return q if q[0] >= 0 else -q


def quat_rotate(q: F64, v: F64) -> F64:
    """Body vector v to inertial."""
    out: F64 = quat_to_dcm(q) @ v
    return out


def quat_from_axis_angle(axis: F64, angle_rad: float) -> F64:
    a = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle_rad / 2)], np.sin(angle_rad / 2) * a])


def quat_angle_rad(q1: F64, q2: F64) -> float:
    """Rotation angle between two attitudes, in [0, pi]."""
    d = quat_mul(quat_conj(q1), q2)
    return float(2 * np.arctan2(np.linalg.norm(d[1:]), abs(d[0])))


def quat_derivative(q: F64, omega: F64) -> F64:
    """Spec: q_dot = 1/2 q (x) [0, omega], omega in body axes."""
    return 0.5 * quat_mul(q, np.concatenate([[0.0], omega]))


def inertia(view: ParamView) -> F64:
    """Body inertia matrix [kg m^2]. Spec: Parameters > Vehicle and actuators."""
    return np.diag(view.array("mass.inertia_diag_kg_m2"))


def gravity_gradient_torque(inertia_b: F64, r_eci: F64, q: F64, mu_km3_s2: float) -> F64:
    """Spec: tau_gg = 3 mu / |r|^3 n_hat x I n_hat [N m]; mu/|r|^3 is in 1/s^2 for any length
    unit."""
    rn = np.linalg.norm(r_eci)
    n_b = quat_to_dcm(q).T @ (-r_eci / rn)
    out: F64 = 3 * mu_km3_s2 / rn**3 * cross3(n_b, inertia_b @ n_b)
    return out


@dataclass(frozen=True)
class AttitudeState:
    """q body to inertial, omega body rate [rad/s], h wheel momenta [N m s]."""

    q: F64
    omega: F64
    h: F64


def total_momentum_inertial(inertia_b: F64, wheels: WheelArray, state: AttitudeState) -> F64:
    """R(q) (I omega + A h), the vehicle's total angular momentum in ECI."""
    out: F64 = quat_to_dcm(state.q) @ (inertia_b @ state.omega + wheels.body_momentum(state.h))
    return out


def step(
    inertia_b: F64,
    wheels: WheelArray,
    state: AttitudeState,
    tau_d: F64,
    dt: float,
    r_eci: F64 | None = None,
    mu_km3_s2: float | None = None,
) -> tuple[AttitudeState, F64, F64]:
    """One attitude + wheel step of dt with the commanded torque tau_d held.

    The wheels advance exactly (wheels.advance). The body equation is integrated in its
    momentum form: the inertial total momentum H = R(q) (I omega + A h) changes only by external
    torque (dH/dt = R(q) tau_gg), and omega = I^-1 (R(q)^T H - A h). This is the spec's
    I omega_dot = tau_gg + tau_w - omega x (I omega + A h) rewritten, with tau_w the reaction to
    dh/dt, so torque-free motion conserves H to roundoff. h is interpolated linearly across the
    step; (q, H) advance by RK4 and q is renormalized. Returns the new state, the allocated wheel
    torques u, and the mean wheel torque on the body tau_w = -A (h1 - h0) / dt.
    """
    u = wheels.allocate(tau_d)
    h0, h1 = state.h, wheels.advance(state.h, u, dt)
    tau_w = wheels.body_torque(h0, h1, dt)
    i_inv = np.linalg.inv(inertia_b)
    a = wheels.A

    def omega_of(q: F64, big_h: F64, frac: float) -> F64:
        out: F64 = i_inv @ (quat_to_dcm(q).T @ big_h - a @ (h0 + (h1 - h0) * frac))
        return out

    def f(frac: float, q: F64, big_h: F64) -> tuple[F64, F64]:
        w = omega_of(q, big_h, frac)
        dh = np.zeros(3)
        if r_eci is not None and mu_km3_s2 is not None:
            dh = quat_to_dcm(q) @ gravity_gradient_torque(inertia_b, r_eci, q, mu_km3_s2)
        return quat_derivative(q, w), dh

    q = state.q
    big_h = total_momentum_inertial(inertia_b, wheels, state)
    k1q, k1h = f(0.0, q, big_h)
    k2q, k2h = f(0.5, q + 0.5 * dt * k1q, big_h + 0.5 * dt * k1h)
    k3q, k3h = f(0.5, q + 0.5 * dt * k2q, big_h + 0.5 * dt * k2h)
    k4q, k4h = f(1.0, q + dt * k3q, big_h + dt * k3h)
    q_new = quat_normalize(q + dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q))
    h_new = big_h + dt / 6 * (k1h + 2 * k2h + 2 * k3h + k4h)
    return AttitudeState(q_new, omega_of(q_new, h_new, 1.0), h1), u, tau_w
