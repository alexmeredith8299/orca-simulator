"""Rest-to-rest reaction-wheel slews: minimum-torque eigen-axis plan plus PD trim (M2).

Spec: Model: actuators ("Slews follow a minimum-torque plan with a PD trim") and Model: orbit and
attitude (rotational equations integrated only during a wheel slew). The spec gives no plan
details, so these defaults apply (CLAUDE.md Known gaps; values in scenario.yaml `slew`):

- Eigen-axis rotation by angle theta about the body axis e of q0* (x) q1.
- Bang-coast-bang angle profile. Acceleration alpha uses `torque_fraction` of the torque the
  array can deliver, unclipped, in the required direction I e. The coast rate uses
  `momentum_fraction` of the wheel momentum capacity in that direction.
- Phase lengths are rounded up to whole slew steps (`step_s`), with alpha lowered to match, so
  the zero-order-hold feedforward I e alpha reproduces the plan exactly. `predicted_s` is the
  unrounded continuous time.
- PD trim on the attitude and rate error: K_p = I w_n^2, K_d = 2 zeta w_n I. The feedforward
  also cancels the gyroscopic term omega x (I omega + A h).
- The slew is complete at the first step after the planned end where the attitude error is below
  `tol_angle_deg` and the rate below `tol_rate_rad_s`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from expand3.params import ParamView
from expand3.vehicle.attitude import (
    AttitudeState,
    inertia,
    quat_angle_rad,
    quat_conj,
    quat_from_axis_angle,
    quat_mul,
    quat_normalize,
    step,
)
from expand3.vehicle.wheels import WheelArray

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class SlewPlan:
    axis_body: F64
    angle_rad: float
    alpha_rad_s2: float
    omega_peak_rad_s: float
    t_acc_s: float
    t_coast_s: float
    duration_s: float
    predicted_s: float

    def profile(self, t: float) -> tuple[float, float, float]:
        """(theta, theta_dot, theta_ddot) of the reference at time t."""
        ta, tc, a, wp = self.t_acc_s, self.t_coast_s, self.alpha_rad_s2, self.omega_peak_rad_s
        if t <= 0 or self.duration_s == 0:
            return 0.0, 0.0, 0.0
        if t < ta:
            return 0.5 * a * t * t, a * t, a
        if t < ta + tc:
            return 0.5 * a * ta * ta + wp * (t - ta), wp, 0.0
        if t < self.duration_s:
            rem = self.duration_s - t
            return self.angle_rad - 0.5 * a * rem * rem, a * rem, -a
        return self.angle_rad, 0.0, 0.0


@dataclass(frozen=True)
class SlewResult:
    """Histories are per slew step: q, omega, h after each step at t_s (end of step); u, tau_w and
    power_W applied during that step (the "wheel momentum, commanded wheel torque each slew
    step" signal)."""

    plan: SlewPlan
    t_s: F64
    q: F64
    omega: F64
    h: F64
    u: F64
    tau_w: F64
    power_W: F64
    duration_s: float
    converged: bool
    final_error_deg: float
    final_rate_rad_s: float
    peak_tracking_error_deg: float


def _eigen_axis(q0: F64, q1: F64) -> tuple[F64, float]:
    d = quat_mul(quat_conj(q0), q1)
    if d[0] < 0:
        d = -d
    s = float(np.linalg.norm(d[1:]))
    if s < 1e-15:
        return np.array([1.0, 0.0, 0.0]), 0.0
    return d[1:] / s, 2.0 * math.atan2(s, float(d[0]))


def plan_slew(view: ParamView, q0: F64, q1: F64) -> SlewPlan:
    """Bang-coast-bang eigen-axis plan from q0 to q1 (both at rest)."""
    sl = view.params.scenario.slew
    wheels = WheelArray(view)
    e, theta = _eigen_axis(q0, q1)
    ie = inertia(view) @ e
    per_unit = float(np.abs(wheels.A_pinv @ (ie / np.linalg.norm(ie))).max())
    alpha_max = sl.torque_fraction * wheels.tau_max / per_unit / np.linalg.norm(ie)
    omega_max = sl.momentum_fraction * wheels.h_max / per_unit / np.linalg.norm(ie)

    if theta == 0.0:
        return SlewPlan(e, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    dt = sl.step_s
    if theta <= omega_max**2 / alpha_max:  # triangular: never reaches the coast rate
        predicted = 2.0 * math.sqrt(theta / alpha_max)
        t_acc = math.ceil(math.sqrt(theta / alpha_max) / dt) * dt
        t_coast = 0.0
        alpha = theta / t_acc**2
    else:
        predicted = theta / omega_max + omega_max / alpha_max
        t_acc = math.ceil(omega_max / alpha_max / dt) * dt
        t_coast = math.ceil((theta / omega_max - omega_max / alpha_max) / dt) * dt
        alpha = theta / (t_acc * (t_acc + t_coast))
    return SlewPlan(
        axis_body=e,
        angle_rad=theta,
        alpha_rad_s2=alpha,
        omega_peak_rad_s=alpha * t_acc,
        t_acc_s=t_acc,
        t_coast_s=t_coast,
        duration_s=2 * t_acc + t_coast,
        predicted_s=predicted,
    )


def slew(
    view: ParamView, q0: F64, q1: F64, h0: F64 | None = None, omega0: F64 | None = None
) -> SlewResult:
    """Fly a rest-to-rest slew from q0 to q1 with the wheels at h0 (default zero)."""
    sl = view.params.scenario.slew
    plan = plan_slew(view, q0, q1)
    i_b = inertia(view)
    wheels = WheelArray(view)
    dt = sl.step_s
    kp = i_b * sl.pd_natural_freq_rad_s**2
    kd = i_b * 2.0 * sl.pd_damping * sl.pd_natural_freq_rad_s
    tol_angle = np.deg2rad(sl.tol_angle_deg)
    e = plan.axis_body

    state = AttitudeState(
        quat_normalize(q0),
        np.zeros(3) if omega0 is None else np.asarray(omega0, dtype=np.float64),
        np.zeros(wheels.A.shape[1]) if h0 is None else np.asarray(h0, dtype=np.float64),
    )
    rec: dict[str, list[F64]] = {k: [] for k in ("t", "q", "w", "h", "u", "tau", "p")}
    max_steps = math.ceil((plan.duration_s + sl.max_settle_s) / dt)
    converged, peak, t_end = False, 0.0, 0.0
    for n in range(max_steps):
        t = n * dt
        th, thd, _ = plan.profile(t)
        _, _, thdd = plan.profile(t + 0.5 * dt)
        q_ref = quat_mul(q0, quat_from_axis_angle(e, th))
        q_err = quat_mul(quat_conj(q_ref), state.q)
        if q_err[0] < 0:
            q_err = -q_err
        w = state.omega
        tau_d = (
            i_b @ e * thdd
            + np.cross(w, i_b @ w + wheels.body_momentum(state.h))
            - kp @ (2.0 * q_err[1:])
            - kd @ (w - e * thd)
        )
        state, u, tau_w = step(i_b, wheels, state, tau_d, dt)
        t_end = t + dt
        rec["t"].append(np.array(t_end))
        rec["q"].append(state.q)
        rec["w"].append(state.omega)
        rec["h"].append(state.h)
        rec["u"].append(u)
        rec["tau"].append(tau_w)
        rec["p"].append(np.array(wheels.torquing_power_w(u)))
        if t_end <= plan.duration_s:
            th1, _, _ = plan.profile(t_end)
            peak = max(peak, quat_angle_rad(quat_mul(q0, quat_from_axis_angle(e, th1)), state.q))
        if t_end >= plan.duration_s - 1e-9 and (
            quat_angle_rad(state.q, q1) < tol_angle
            and float(np.linalg.norm(state.omega)) < sl.tol_rate_rad_s
        ):
            converged = True
            break

    return SlewResult(
        plan=plan,
        t_s=np.array(rec["t"]),
        q=np.array(rec["q"]),
        omega=np.array(rec["w"]),
        h=np.array(rec["h"]),
        u=np.array(rec["u"]),
        tau_w=np.array(rec["tau"]),
        power_W=np.array(rec["p"]),
        duration_s=t_end,
        converged=converged,
        final_error_deg=float(np.rad2deg(quat_angle_rad(state.q, q1))),
        final_rate_rad_s=float(np.linalg.norm(state.omega)),
        peak_tracking_error_deg=float(np.rad2deg(peak)),
    )
