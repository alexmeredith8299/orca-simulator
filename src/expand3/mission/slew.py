"""Rest-to-rest reaction-wheel slews: minimum-torque eigen-axis plan plus PD trim (M2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from expand3.params import ParamView

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


@dataclass(frozen=True)
class SlewResult:
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


def plan_slew(view: ParamView, q0: F64, q1: F64) -> SlewPlan:
    raise NotImplementedError


def slew(
    view: ParamView, q0: F64, q1: F64, h0: F64 | None = None, omega0: F64 | None = None
) -> SlewResult:
    raise NotImplementedError
