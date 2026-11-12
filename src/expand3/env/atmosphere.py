"""Exponential atmosphere (M1)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]


def density_kg_m3(h_km: F64, rho0_kg_m3: float, h0_km: float, H_km: float) -> F64:
    """Spec: Model: orbit and attitude, rho(h) = rho_0 exp(-(h - h_0)/H), h = |r| - R_E."""
    return np.asarray(rho0_kg_m3 * np.exp(-(np.asarray(h_km) - h0_km) / H_km), dtype=np.float64)
