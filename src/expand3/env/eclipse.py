"""Dual-cone (umbra/penumbra) shadow fraction (M1).

Spec: Model: electrical power, "s is the shadow fraction (dual-cone penumbra, 1 = full Sun)".
Earth is a sphere of radius R_E. The visible fraction of the solar disk is computed from the
apparent radii of the Sun and Earth and their angular separation as seen from the vehicle,
using the area of overlap of two flat disks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]


def shadow_fraction(
    r_km: F64, sun_unit: F64, sun_dist_km: F64 | float, R_E_km: float, R_sun_km: float
) -> F64:
    """Shadow fraction s in [0, 1] (1 = full Sun) for positions r_km (..., 3) in ECI.

    sun_unit (..., 3) and sun_dist_km (...) are the geocentric Sun direction and distance and
    broadcast against r_km.
    """
    r = np.asarray(r_km, dtype=np.float64)
    to_sun = np.asarray(sun_unit) * np.asarray(sun_dist_km)[..., None] - r
    d_sun = np.linalg.norm(to_sun, axis=-1)
    d_earth = np.linalg.norm(r, axis=-1)
    a = np.arcsin(np.clip(R_sun_km / d_sun, -1.0, 1.0))  # Sun apparent radius
    b = np.arcsin(np.clip(R_E_km / d_earth, -1.0, 1.0))  # Earth apparent radius
    cos_c = np.sum(to_sun * -r, axis=-1) / (d_sun * d_earth)
    c = np.arccos(np.clip(cos_c, -1.0, 1.0))  # separation of the two disk centers

    a, b, c = np.broadcast_arrays(a, b, c)
    s = np.ones(c.shape)
    umbra = c <= b - a
    annular = c <= a - b  # Earth's disk entirely inside the Sun's (not reachable in LEO)
    partial = (c < a + b) & ~umbra & ~annular
    s[umbra] = 0.0
    s[annular] = 1.0 - (b[annular] / a[annular]) ** 2

    ap, bp, cp = a[partial], b[partial], c[partial]
    x1 = np.clip((cp**2 + ap**2 - bp**2) / (2 * ap * cp), -1.0, 1.0)
    x2 = np.clip((cp**2 + bp**2 - ap**2) / (2 * bp * cp), -1.0, 1.0)
    k = (-cp + ap + bp) * (cp + ap - bp) * (cp - ap + bp) * (cp + ap + bp)
    overlap = ap**2 * np.arccos(x1) + bp**2 * np.arccos(x2) - 0.5 * np.sqrt(np.maximum(k, 0.0))
    s[partial] = np.clip(1.0 - overlap / (np.pi * ap**2), 0.0, 1.0)
    return s
