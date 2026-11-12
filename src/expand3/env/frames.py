"""Earth frames: ECI (GCRS) to ECEF (ITRS) rotations and the Earth pole (M1).

ECI is GCRS and ECEF is ITRS. Rotations come from ERFA's IAU 2006/2000A celestial-to-terrestrial
matrix with astropy's time scales and IERS Earth orientation, vectorized over all requested times,
which is the same transform astropy's GCRS -> ITRS applies. astropy runs once per run, never inside
the integrator. IERS tables are used offline (bundled data, no downloads) so runs are reproducible;
times past the tables use astropy's predictions or edge values.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

import astropy.units as u
import erfa
import numpy as np
from astropy.time import Time
from astropy.utils import iers
from numpy.typing import NDArray

from expand3.units import DEG_PER_HOUR_ANGLE, HOURS_PER_DAY

F64 = NDArray[np.float64]

iers.conf.auto_download = False
# The bundled predictions age past astropy's 30 day limit, which would stop offline runs with an
# error. Only the error is lifted; the tables, and so the results, are unchanged.
iers.conf.auto_max_age = None
iers.conf.iers_degraded_accuracy = "warn"


@contextmanager
def quiet_astropy() -> Iterator[None]:
    """Silence ERFA 'dubious year' and IERS-range warnings for future epochs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", erfa.ErfaWarning)
        warnings.filterwarnings("ignore", module=r"astropy\..*")
        yield


def times(epoch_utc: datetime, t_s: F64) -> Time:
    """astropy Time for epoch + t_s seconds (UTC)."""
    return Time(epoch_utc, scale="utc") + np.asarray(t_s, dtype=np.float64) * u.s


def eci_to_ecef(epoch_utc: datetime, t_s: F64) -> F64:
    """(N, 3, 3) rotation matrices with r_ecef = M @ r_eci (GCRS -> ITRS)."""
    with quiet_astropy():
        t = times(epoch_utc, t_s)
        tt, ut1 = t.tt, t.ut1
        xp, yp = iers.earth_orientation_table.get().pm_xy(t)
        m = erfa.c2t06a(
            tt.jd1, tt.jd2, ut1.jd1, ut1.jd2, xp.to_value(u.rad), yp.to_value(u.rad)
        )
    return np.asarray(m, dtype=np.float64).reshape(-1, 3, 3)


def pole_eci(epoch_utc: datetime) -> F64:
    """Earth's rotation pole (ITRS +z) in ECI at the epoch. Fixed over a run (precession over
    72 h is about 0.0004 deg)."""
    m = eci_to_ecef(epoch_utc, np.zeros(1))[0]
    return np.asarray(m[2], dtype=np.float64)


def mean_sun_lon_ecef_rad(epoch_utc: datetime, t_s: F64) -> F64:
    """ECEF longitude of the mean Sun. Mean solar time at Greenwich is UT1, so the mean Sun's
    Greenwich hour angle is (UT1 - 12 h) x 15 deg/h and its longitude is minus that."""
    with quiet_astropy():
        ut1 = times(epoch_utc, t_s).ut1
    day_frac = np.mod(np.mod(ut1.jd1 - 0.5, 1.0) + ut1.jd2, 1.0)
    gha_deg = (day_frac * HOURS_PER_DAY - HOURS_PER_DAY / 2) * DEG_PER_HOUR_ANGLE
    return np.asarray(np.deg2rad(-gha_deg), dtype=np.float64)
