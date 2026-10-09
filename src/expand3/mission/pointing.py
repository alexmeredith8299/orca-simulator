"""Pointing law in place of the flight stack: idle Sun hold, burn, imaging take (M2)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from expand3.params import ParamView

F64 = NDArray[np.float64]


def axis_from_label(label: str) -> F64:
    raise NotImplementedError


def triad(b1: F64, i1: F64, b2: F64, i2: F64) -> F64:
    raise NotImplementedError


def sun_hold(view: ParamView, sun_eci: F64, orbit_normal_eci: F64) -> F64:
    raise NotImplementedError


def burn(view: ParamView, a_cmd_eci: F64, sun_eci: F64) -> F64:
    raise NotImplementedError


def take(
    view: ParamView, r_eci: F64, target_ecef: F64, eci_to_ecef: F64, orbit_normal_eci: F64
) -> F64:
    raise NotImplementedError
