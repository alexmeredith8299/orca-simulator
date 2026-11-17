"""Onboard field reference (Phase C): the vehicle's model of what its magnetometer should read.

Not in the spec; vehicle software like the navigation filter. The model is IGRF-14 alone (no
disturbance) evaluated at the navigation solution's position, rotated into the body frame by the
star-tracker attitude. It needs a tracker fix, so it is NaN on ticks without one; its magnitude
needs only the position and is always available. The true state is never an input.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from expand3.env.geomag import igrf_field_ecef_nt
from expand3.vehicle.attitude import quat_to_dcm

F64 = NDArray[np.float64]


def onboard_field(r_eci_km: F64, rot_ecef_from_eci: F64, epoch_utc: datetime, q_meas: F64
                  ) -> tuple[F64, F64]:
    """(body-frame model field [nT] (n, 3), model magnitude [nT] (n,)) at navigation positions
    (n, 3), with the Earth rotation (n, 3, 3) the vehicle knows and the measured attitude (n, 4)
    from body to inertial (NaN without a fix)."""
    r_ecef = np.einsum("nij,nj->ni", rot_ecef_from_eci, r_eci_km)
    b_ecef = igrf_field_ecef_nt(r_ecef, epoch_utc)
    b_eci = np.einsum("nji,nj->ni", rot_ecef_from_eci, b_ecef)
    body = np.full_like(b_eci, np.nan)
    for i in np.flatnonzero(np.all(np.isfinite(q_meas), axis=1)):
        body[i] = quat_to_dcm(q_meas[i]).T @ b_eci[i]
    return body, np.linalg.norm(b_ecef, axis=1)
