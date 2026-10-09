"""M3: sub-satellite point, Sun elevation, off-nadir, and the take-time rule."""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import EarthLocation

from expand3.mission.takes import (
    find_take_tick,
    off_nadir_deg,
    subsatellite_point,
    sun_elevation_deg,
)
from expand3.params import Params, ParamView


def test_subsatellite_point_is_geodetic() -> None:
    rng = np.random.default_rng(4)
    r = rng.normal(size=(20, 3))
    r = 6903.0 * r / np.linalg.norm(r, axis=1, keepdims=True)
    point, up, lat, lon = subsatellite_point(r)
    loc = EarthLocation.from_geocentric(*(r.T * u.km))
    np.testing.assert_allclose(lat, loc.lat.deg, atol=1e-9)
    np.testing.assert_allclose(lon, loc.lon.wrap_at(180 * u.deg).deg, atol=1e-9)
    ground = EarthLocation.from_geodetic(loc.lon, loc.lat, 0 * u.m)
    np.testing.assert_allclose(point, np.stack([ground.x, ground.y, ground.z], -1).to_value(u.km),
                               atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(up, axis=1), 1.0)
    # the satellite lies along the local vertical of its geodetic sub-point
    d = r - point
    np.testing.assert_allclose(np.cross(d / np.linalg.norm(d, axis=1, keepdims=True), up), 0,
                               atol=1e-9)


def test_sun_elevation_and_off_nadir() -> None:
    point = np.array([[6378.137, 0, 0]])
    up = np.array([[1.0, 0, 0]])
    assert sun_elevation_deg(point, up, np.array([[1.5e8, 0, 0]]))[0] == pytest.approx(90)
    assert sun_elevation_deg(point, up, np.array([[0, 1.5e8, 0]]))[0] == pytest.approx(0, abs=0.01)
    r = np.array([[6903.137, 0, 0]])
    assert off_nadir_deg(r, point)[0] == pytest.approx(0)
    side = np.array([[6378.137, 525.0, 0]])
    assert off_nadir_deg(r, side)[0] == pytest.approx(45, abs=0.01)


def test_take_tick_rule(params: Params) -> None:
    view = ParamView(params, "SAT-2")
    t = np.arange(0, 40 * 3600 + 1, 30.0)
    # vehicle over the equator at lon 0; Sun moves so its elevation there rises through 10 deg
    r = np.tile([6903.137, 0, 0], (len(t), 1))
    elev = np.deg2rad(np.linspace(-80, 20, len(t)))  # crosses 10 deg at 36 h
    sun = 1.5e8 * np.stack([np.sin(elev), np.cos(elev), np.zeros_like(elev)], -1)
    n = find_take_tick(view, t, r, sun)
    assert n is not None
    assert t[n] >= 30 * 3600
    point, up, _, _ = subsatellite_point(r)
    se = sun_elevation_deg(point, up, sun)
    assert se[n] >= 10
    assert np.all(se[(t >= 30 * 3600) & (t < t[n])] < 10)
    # never satisfied -> None
    assert find_take_tick(view, t, r, -sun * 0 + np.array([-1.5e8, 0, 0])) is None
