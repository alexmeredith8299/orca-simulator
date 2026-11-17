"""True magnetic field: IGRF-14 plus a common disturbance (extended faults, Phase C).

Not in the spec, which gives only the IGRF magnitude for the SAA (`vehicle/cdh.py`). The vector
comes from the same ppigrf call. The real field is noisier than IGRF (external currents, storms),
so the truth a magnetometer senses is IGRF plus a disturbance that is smooth in space and time and
the same for every vehicle at the same place and time. A vehicle's onboard model is IGRF alone,
so the disturbance is a common-mode error that one vehicle cannot tell from a sensor bias.

The disturbance is a sum of K waves in the Earth-fixed frame, dB(x, t) = sum a_k cos(k_k . x -
w_k t + phi_k), each polarised perpendicular to its wave vector. Wavelengths and periods are
log-uniform between the parameter bounds; the amplitude makes the rms of |dB| equal rms_nT. The
draws come from the named stream `draw.field`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import ppigrf
from numpy.typing import NDArray

from expand3.params import ParamView
from expand3.timebase import rng
from expand3.units import S_PER_HOUR

F64 = NDArray[np.float64]


def igrf_field_ecef_nt(r_ecef_km: F64, epoch_utc: datetime) -> F64:
    """IGRF-14 field vector [nT] in the Earth-fixed frame at ECEF positions (n, 3) [km], from
    ppigrf's geocentric components (up, south, east)."""
    r = np.atleast_2d(np.asarray(r_ecef_km, dtype=np.float64))
    rad = np.linalg.norm(r, axis=-1)
    colat = np.arccos(np.clip(r[:, 2] / rad, -1.0, 1.0))
    lon = np.arctan2(r[:, 1], r[:, 0])
    date = epoch_utc.astimezone(UTC).replace(tzinfo=None)
    b_r, b_t, b_p = ppigrf.igrf_gc(rad, np.rad2deg(colat), np.rad2deg(lon), date)
    up = r / rad[:, None]
    south = np.stack([np.cos(colat) * np.cos(lon), np.cos(colat) * np.sin(lon),
                      -np.sin(colat)], axis=1)
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], axis=1)
    out: F64 = (np.reshape(b_r, (-1, 1)) * up + np.reshape(b_t, (-1, 1)) * south
                + np.reshape(b_p, (-1, 1)) * east)
    return out


@dataclass(frozen=True)
class FieldDisturbance:
    """K waves: wave vectors [rad/km], angular rates [rad/s], phases and vector amplitudes [nT]."""

    k_rad_km: F64
    omega_rad_s: F64
    phase_rad: F64
    amp_nT: F64

    @classmethod
    def draw(cls, view: ParamView, seed: int) -> FieldDisturbance:
        g = rng(seed, "draw.field")
        n = int(view.get("field_disturbance.modes"))
        direction = g.normal(size=(n, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        lam = np.exp(g.uniform(np.log(view.get("field_disturbance.wavelength_min_km")),
                               np.log(view.get("field_disturbance.wavelength_max_km")), n))
        period = S_PER_HOUR * np.exp(g.uniform(np.log(view.get("field_disturbance.period_min_h")),
                                               np.log(view.get("field_disturbance.period_max_h")),
                                               n))
        phase = g.uniform(0.0, 2.0 * np.pi, n)
        pol = np.cross(direction, g.normal(size=(n, 3)))  # perpendicular to the wave vector
        pol /= np.linalg.norm(pol, axis=1, keepdims=True)
        amp = view.get("field_disturbance.rms_nT") * np.sqrt(2.0 / n)
        sign = np.where(g.random(n) < 0.5, -1.0, 1.0)  # waves travel either way
        return cls(2.0 * np.pi / lam[:, None] * direction, sign * 2.0 * np.pi / period, phase,
                   amp * pol)

    def at(self, r_ecef_km: F64, t_s: F64) -> F64:
        """The disturbance [nT] (n, 3) at ECEF positions (n, 3) [km] and times (n,) [s]."""
        arg = (np.asarray(r_ecef_km, dtype=np.float64) @ self.k_rad_km.T
               - np.asarray(t_s, dtype=np.float64)[:, None] * self.omega_rad_s + self.phase_rad)
        out: F64 = np.cos(arg) @ self.amp_nT
        return out
