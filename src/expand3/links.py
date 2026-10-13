"""Intra-cluster crosslink: range, line of sight, drops, receive noise, partner-state age (M6)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from expand3.params import Params, ParamView

F64 = NDArray[np.float64]
B = NDArray[np.bool_]


@dataclass
class CrosslinkLog:
    pairs: list[tuple[str, str]]
    t_s: F64
    sent: B
    delivered: B
    r_meas_km: F64
    v_meas_km_s: F64
    receipts: dict[str, NDArray[np.int64]] = field(default_factory=dict)
    age_s: dict[tuple[str, str], F64] = field(default_factory=dict)


def line_of_sight(r1_km: F64, r2_km: F64, R_E_km: float) -> B:
    raise NotImplementedError


def crosslink(
    seed: int, params: Params, views: dict[str, ParamView], sats: list[str], t_s: F64,
    r_km: F64, v_km_s: F64,
) -> CrosslinkLog:
    raise NotImplementedError
