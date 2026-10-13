"""SAA and C&DH: IGRF-14 field magnitude, SAA flag, edge computer, Poisson upsets (M6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from expand3.params import ParamView

F64 = NDArray[np.float64]
B = NDArray[np.bool_]
I64 = NDArray[np.int64]


@dataclass
class SaaHistory:
    B_nT: F64
    in_saa: B
    edge_off: B
    upsets: I64
    resets: I64


def field_magnitude_nt(r_ecef_km: F64, epoch_utc: datetime) -> F64:
    raise NotImplementedError


def edge_computer_off(t_s: F64, in_saa: B, reboot_s: float) -> B:
    raise NotImplementedError


def upsets(
    rng: np.random.Generator, view: ParamView, in_saa: B, dt: float
) -> tuple[I64, I64]:
    raise NotImplementedError


def saa_history(
    seed: int, view: ParamView, t_s: F64, r_ecef_km: F64, epoch_utc: datetime
) -> SaaHistory:
    raise NotImplementedError
