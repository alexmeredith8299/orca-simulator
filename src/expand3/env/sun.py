"""Sun direction and distance in ECI (GCRS), vectorized with astropy (M1).

Spec: Model: electrical power and Model: thermal network use the Sun direction; the shadow
fraction s needs the Sun's distance. Geocentric apparent position (astropy `get_body`).
"""

from __future__ import annotations

from datetime import datetime

import astropy.units as u
import numpy as np
from astropy.coordinates import get_body
from numpy.typing import NDArray

from expand3.env.frames import quiet_astropy, times

F64 = NDArray[np.float64]


def sun_eci(epoch_utc: datetime, t_s: F64) -> tuple[F64, F64]:
    """Sun unit vector (N, 3) and distance [km] (N,) from Earth's center in GCRS."""
    with quiet_astropy():
        xyz = get_body("sun", times(epoch_utc, t_s)).cartesian.xyz.to_value(u.km)
    vec = np.asarray(xyz, dtype=np.float64).reshape(3, -1).T
    dist = np.linalg.norm(vec, axis=-1)
    return vec / dist[:, None], dist
