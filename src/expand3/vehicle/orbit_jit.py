"""Compiled RK4 step of the orbit force model (numba). Spec: Model: orbit and attitude.

The same equations as `OrbitModel.accel` (point mass, J2 about the pole, exponential-atmosphere
drag, ordered thrust) with the same operation order, but in one compiled loop over vehicles, so the
10 s integration does not pay numpy's per-call overhead on six-row arrays. No model values live
here: every number is an argument."""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray

F64 = NDArray[np.float64]


@njit(cache=True)
def _deriv(
    y: F64, a_thr: F64, ball: F64, out: F64, mu: float, j2: float, r_e: float, w_e: float,
    pole: F64, rho0: float, h0: float, scale_h: float, use_j2: bool, use_drag: bool,
    m_per_km: float,
) -> None:
    for i in range(y.shape[0]):
        x, yy, z_ = y[i, 0], y[i, 1], y[i, 2]
        vx, vy, vz = y[i, 3], y[i, 4], y[i, 5]
        rn = np.sqrt(x * x + yy * yy + z_ * z_)
        c = -mu / rn**3
        ax, ay, az = c * x, c * yy, c * z_
        if use_j2:
            z = x * pole[0] + yy * pole[1] + z_ * pole[2]
            k = -1.5 * j2 * mu * r_e**2 / rn**5
            f = 1.0 - 5.0 * z**2 / rn**2
            ax += k * (f * x + 2.0 * z * pole[0])
            ay += k * (f * yy + 2.0 * z * pole[1])
            az += k * (f * z_ + 2.0 * z * pole[2])
        if use_drag:
            rx = vx - w_e * (pole[1] * z_ - pole[2] * yy)
            ry = vy - w_e * (pole[2] * x - pole[0] * z_)
            rz = vz - w_e * (pole[0] * yy - pole[1] * x)
            speed = np.sqrt(rx * rx + ry * ry + rz * rz)
            rho = rho0 * np.exp(-((rn - r_e) - h0) / scale_h)
            d = 0.5 * rho * ball[i] * m_per_km * speed
            ax -= d * rx
            ay -= d * ry
            az -= d * rz
        out[i, 0], out[i, 1], out[i, 2] = vx, vy, vz
        out[i, 3] = ax + a_thr[i, 0]
        out[i, 4] = ay + a_thr[i, 1]
        out[i, 5] = az + a_thr[i, 2]


@njit(cache=True)
def rk4_orbit(
    y: F64, dt: float, a_thr: F64, ball: F64, mu: float, j2: float, r_e: float, w_e: float,
    pole: F64, rho0: float, h0: float, scale_h: float, use_j2: bool, use_drag: bool,
    m_per_km: float,
) -> F64:
    """One classical RK4 step of dt for the (K, 6) states, thrust held over the step."""
    k1 = np.empty_like(y)
    k2 = np.empty_like(y)
    k3 = np.empty_like(y)
    k4 = np.empty_like(y)
    _deriv(y, a_thr, ball, k1, mu, j2, r_e, w_e, pole, rho0, h0, scale_h, use_j2, use_drag,
           m_per_km)
    _deriv(y + 0.5 * dt * k1, a_thr, ball, k2, mu, j2, r_e, w_e, pole, rho0, h0, scale_h, use_j2,
           use_drag, m_per_km)
    _deriv(y + 0.5 * dt * k2, a_thr, ball, k3, mu, j2, r_e, w_e, pole, rho0, h0, scale_h, use_j2,
           use_drag, m_per_km)
    _deriv(y + dt * k3, a_thr, ball, k4, mu, j2, r_e, w_e, pole, rho0, h0, scale_h, use_j2,
           use_drag, m_per_km)
    out: F64 = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return out
