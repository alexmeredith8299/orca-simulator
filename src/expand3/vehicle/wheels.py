"""Reaction-wheel pyramid: allocation, momentum update, saturation, torquing power (M2).

Spec: Model: actuators (F3, F4). Four wheels with spin axes a_k at a fixed angle off +z (columns
of A, 3 x 4). Each step dt:

    u = clip(A+ tau_d, +-tau_max)
    h_k(t + dt) = h_k(t) - (k_k u_k + c_k h_k) dt
    tau_w = A (k o u + c o h)

The momentum update is the spec's discrete form exactly as written, evaluated with h(t): under
friction alone h_n = h_0 (1 - c dt)^n. A wheel at |h_k| = h_max saturates: h_k is held at h_max
and the delivered torque is cut to match. The torque on the body over a step is
-A (h(t + dt) - h(t)) / dt, which equals the spec's A (k o u + c o h(t)) when no wheel saturates.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from expand3.params import ParamView

F64 = NDArray[np.float64]


def pyramid_axes(angle_off_z_deg: float, azimuth_deg: F64) -> F64:
    """A (3, K): unit spin axes at angle_off_z_deg from +z and the given azimuths from +x."""
    beta = np.deg2rad(angle_off_z_deg)
    az = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64))
    return np.vstack([np.sin(beta) * np.cos(az), np.sin(beta) * np.sin(az),
                      np.cos(beta) * np.ones_like(az)])


class WheelArray:
    """One vehicle's wheel array. k and c are read through ParamView (`wheels.k`, `wheels.c`) so
    fault overrides (F3, F4) apply."""

    A: F64
    A_pinv: F64
    tau_max: float
    h_max: float
    k: F64
    c: F64
    power_per_wheel_W: float

    def __init__(self, view: ParamView) -> None:
        self.A = pyramid_axes(
            view.get("wheels.spin_axis_angle_off_z_deg"), view.array("wheels.azimuth_deg")
        )
        self.A_pinv = np.linalg.pinv(self.A)
        self.tau_max = view.get("wheels.tau_max_Nm")
        self.h_max = view.get("wheels.h_max_Nms")
        self.k = view.array("wheels.k")
        self.c = view.array("wheels.c")
        self.power_per_wheel_W = view.get("wheels.power_while_torquing_W")

    def allocate(self, tau_d: F64) -> F64:
        """Spec: u = clip(A+ tau_d, +-tau_max), minimum-norm allocation."""
        u: F64 = np.clip(self.A_pinv @ tau_d, -self.tau_max, self.tau_max)
        return u

    def advance(self, h: F64, u: F64, dt: float) -> F64:
        """Spec: h_k(t + dt) = h_k(t) - (k_k u_k + c_k h_k(t)) dt, then saturation at h_max."""
        h_new = h - (self.k * u + self.c * h) * dt
        out: F64 = np.clip(h_new, -self.h_max, self.h_max)
        return out

    def body_torque(self, h: F64, h_new: F64, dt: float) -> F64:
        """Mean torque on the body over the step: -A dh/dt (momentum leaves the wheels)."""
        out: F64 = -self.A @ (h_new - h) / dt
        return out

    def body_momentum(self, h: F64) -> F64:
        """Spec: stored body momentum is A h."""
        out: F64 = self.A @ h
        return out

    def torquing_power_w(self, u: F64) -> float:
        """Spec: each wheel draws its torquing power while it torques (|u_k| > 0)."""
        return float(self.power_per_wheel_W * np.count_nonzero(np.abs(u) > 0))
