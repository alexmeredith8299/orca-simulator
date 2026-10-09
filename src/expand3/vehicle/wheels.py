"""Reaction-wheel pyramid: allocation, momentum update, saturation, torquing power (M2)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from expand3.params import ParamView

F64 = NDArray[np.float64]


def pyramid_axes(angle_off_z_deg: float, azimuth_deg: F64) -> F64:
    raise NotImplementedError


class WheelArray:
    A: F64
    A_pinv: F64
    tau_max: float
    h_max: float
    k: F64
    c: F64
    power_per_wheel_W: float

    def __init__(self, view: ParamView) -> None:
        raise NotImplementedError

    def allocate(self, tau_d: F64) -> F64:
        raise NotImplementedError

    def advance(self, h: F64, u: F64, dt: float) -> F64:
        raise NotImplementedError

    def body_torque(self, h: F64, h_new: F64, dt: float) -> F64:
        raise NotImplementedError

    def body_momentum(self, h: F64) -> F64:
        raise NotImplementedError

    def torquing_power_w(self, u: F64) -> float:
        raise NotImplementedError
