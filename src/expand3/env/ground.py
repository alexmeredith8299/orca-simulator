"""Ground stations: WGS84 sites, elevation of each vehicle, visibility and passes (M1).

Spec: Parameters > Scenario and environment (Svalbard, Fairbanks, Wallops; 5 deg elevation mask).
Sites sit on the WGS84 ellipsoid at zero height. Elevation is geometric (no refraction), measured
from the local geodetic horizontal.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from numpy.typing import NDArray

from expand3.params import ParamView

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class Station:
    name: str
    ecef_km: F64
    up: F64


def stations(view: ParamView) -> list[Station]:
    out = []
    for gs in view.params.scenario.ground_stations:
        loc = EarthLocation.from_geodetic(gs.lon_deg * u.deg, gs.lat_deg * u.deg, 0 * u.m)
        xyz = np.array([loc.x.to_value(u.km), loc.y.to_value(u.km), loc.z.to_value(u.km)])
        lat, lon = np.deg2rad(gs.lat_deg), np.deg2rad(gs.lon_deg)
        up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
        out.append(Station(gs.name, xyz, up))
    return out


def elevation_deg(r_ecef_km: F64, station: Station) -> F64:
    """Elevation [deg] of positions r_ecef_km (..., 3) above the station's geodetic horizon."""
    rho = np.asarray(r_ecef_km) - station.ecef_km
    sin_el = (rho @ station.up) / np.linalg.norm(rho, axis=-1)
    return np.asarray(np.rad2deg(np.arcsin(np.clip(sin_el, -1.0, 1.0))), dtype=np.float64)


def pass_intervals(t_s: F64, visible: NDArray[np.bool_]) -> list[tuple[float, float]]:
    """(first, last) sample time of each contiguous run of visible samples."""
    v = np.concatenate([[False], np.asarray(visible, dtype=bool), [False]])
    edges = np.flatnonzero(np.diff(v.astype(np.int8)))
    starts, ends = edges[0::2], edges[1::2] - 1
    return [(float(t_s[a]), float(t_s[b])) for a, b in zip(starts, ends, strict=True)]
