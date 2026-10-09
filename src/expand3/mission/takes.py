"""Time-critical imaging takes: take time, target, constraints (M3).

Spec: Scenario > Mission thread ("the target sits under the vehicle's ground track at the first
point in the take window where the take constraints hold"); Parameters > Scenario (take window
after 30 h; off-nadir at most 45 deg; Sun at least 10 deg above the horizon at the target).

The take time is the first control tick at or after the window start where the Sun is at least
the minimum elevation at the vehicle's geodetic sub-satellite point. The target is that point on
the WGS84 ellipsoid, fixed in ECEF; off-nadir is then near zero at the take start. Off-nadir is
measured from geocentric nadir. Sun elevation is geometric, from the target's geodetic horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from numpy.typing import NDArray

from expand3.params import ParamView
from expand3.units import S_PER_HOUR

F64 = NDArray[np.float64]


@dataclass
class TakeRecord:
    """One vehicle's take: times [s from epoch], fixed target, and per-take-tick constraint
    values."""

    sat: str
    t_take_s: float
    t_end_s: float
    target_ecef_km: F64
    target_lat_deg: float
    target_lon_deg: float
    slew_in_start_s: float = 0.0
    slew_out_end_s: float = 0.0
    tick_times_s: list[float] = field(default_factory=list)
    sun_elevation_deg: list[float] = field(default_factory=list)
    off_nadir_deg: list[float] = field(default_factory=list)
    boresight_error_rad: list[float] = field(default_factory=list)


def subsatellite_point(r_ecef_km: F64) -> tuple[F64, F64, F64, F64]:
    """Geodetic sub-satellite point on WGS84 (ECEF km), its local up, latitude and longitude."""
    r = np.atleast_2d(r_ecef_km)
    loc = EarthLocation.from_geocentric(r[:, 0] * u.km, r[:, 1] * u.km, r[:, 2] * u.km)
    ground = EarthLocation.from_geodetic(loc.lon, loc.lat, 0 * u.m)
    point = np.stack([ground.x.to_value(u.km), ground.y.to_value(u.km), ground.z.to_value(u.km)],
                     axis=-1)
    lat = np.asarray(loc.lat.deg, dtype=np.float64)
    lon = np.asarray(loc.lon.wrap_at(180 * u.deg).deg, dtype=np.float64)
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    up = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)
    return point, up, lat, lon


def sun_elevation_deg(point_ecef_km: F64, up: F64, sun_ecef_km: F64) -> F64:
    """Geometric Sun elevation [deg] above the local horizon at ECEF points."""
    d = np.asarray(sun_ecef_km) - np.asarray(point_ecef_km)
    s = np.sum(d * up, axis=-1) / np.linalg.norm(d, axis=-1)
    return np.asarray(np.rad2deg(np.arcsin(np.clip(s, -1, 1))), dtype=np.float64)


def off_nadir_deg(r_eci_km: F64, target_eci_km: F64) -> F64:
    """Angle [deg] between the line of sight to the target and geocentric nadir."""
    los = np.asarray(target_eci_km) - np.asarray(r_eci_km)
    nadir = -np.asarray(r_eci_km)
    norms = np.linalg.norm(los, axis=-1) * np.linalg.norm(nadir, axis=-1)
    c = np.sum(los * nadir, axis=-1) / norms
    return np.asarray(np.rad2deg(np.arccos(np.clip(c, -1, 1))), dtype=np.float64)


def find_take_tick(
    view: ParamView, t_s: F64, r_ecef_km: F64, sun_ecef_km: F64
) -> int | None:
    """Index of the first tick at or after the take window start where the take constraints hold
    for a target at the sub-satellite point, or None."""
    start = view.get("scenario.take_window_start_h") * S_PER_HOUR
    idx = np.flatnonzero(np.asarray(t_s) >= start)
    if idx.size == 0:
        return None
    point, up, _, _ = subsatellite_point(np.asarray(r_ecef_km)[idx])
    sun_ok = sun_elevation_deg(point, up, np.asarray(sun_ecef_km)[idx]) >= view.get(
        "scenario.take_constraints.min_sun_elevation_at_target_deg"
    )
    nadir_ok = off_nadir_deg(np.asarray(r_ecef_km)[idx], point) <= view.get(
        "scenario.take_constraints.max_off_nadir_deg"
    )
    hits = np.flatnonzero(sun_ok & nadir_ok)
    return int(idx[hits[0]]) if hits.size else None
