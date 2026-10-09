"""M1: J2 nodal regression is sun-synchronous and the plane keeps LTAN 22:30 over 72 h."""

import numpy as np
import pytest

from expand3.constellation import ConstellationRun
from expand3.env.frames import mean_sun_lon_ecef_rad
from expand3.params import Params
from expand3.timebase import SeedDraws
from expand3.units import DEG_PER_HOUR_ANGLE, HOURS_PER_DAY, S_PER_DAY
from expand3.vehicle.orbit import equatorial_basis, rv_to_elements

pytestmark = pytest.mark.slow


def analytic_raan_rate_rad_s(params: Params) -> float:
    c, o = params.scenario.constants, params.scenario.orbit
    a, e, i = o.a_km, o.e, np.deg2rad(o.i_deg)
    n = np.sqrt(c.mu_km3_s2 / a**3)
    p = a * (1 - e**2)
    return float(-1.5 * n * c.J2 * (c.R_E_km / p) ** 2 * np.cos(i))


def test_analytic_rate_is_sun_synchronous(params: Params) -> None:
    rate_deg_day = np.rad2deg(analytic_raan_rate_rad_s(params)) * S_PER_DAY
    assert rate_deg_day == pytest.approx(360 / 365.2422, rel=1e-3)  # 0.9856 deg/day


def test_raan_drift_matches_j2_secular_rate(params: Params, run72: ConstellationRun) -> None:
    mu = params.scenario.constants.mu_km3_s2
    basis = equatorial_basis(run72.pole_eci, np.array([1.0, 0.0, 0.0]))
    for k in range(len(run72.sats)):
        el = rv_to_elements(run72.r_km[:, k], run72.v_km_s[:, k], mu, basis)
        raan = np.unwrap(el.raan_rad)
        rate = np.polyfit(run72.t_s, raan, 1)[0]
        assert rate == pytest.approx(analytic_raan_rate_rad_s(params), rel=0.02)


def test_nodal_period(params: Params, run72: ConstellationRun) -> None:
    mu = params.scenario.constants.mu_km3_s2
    a = params.scenario.orbit.a_km
    basis = equatorial_basis(run72.pole_eci, np.array([1.0, 0.0, 0.0]))
    el = rv_to_elements(run72.r_km[:, 0], run72.v_km_s[:, 0], mu, basis)
    u = np.unwrap(el.u_rad)
    assert np.all(np.diff(u) > 0)  # argument of latitude advances monotonically
    t_rev = 2 * np.pi / np.polyfit(run72.t_s, u, 1)[0]
    assert t_rev == pytest.approx(2 * np.pi * np.sqrt(a**3 / mu), rel=5e-3)


def test_ltan_holds(draws: SeedDraws, run72: ConstellationRun) -> None:
    lon_ms = mean_sun_lon_ecef_rad(draws.epoch_utc, run72.t_s)
    h = np.cross(run72.r_km, run72.v_km_s)
    node = np.cross(run72.pole_eci, h)  # ascending node direction, (N, K, 3)
    node_ecef = np.einsum("nij,nkj->nki", run72.eci_to_ecef, node)
    lon_an = np.arctan2(node_ecef[..., 1], node_ecef[..., 0])
    hours = HOURS_PER_DAY / 2 + np.rad2deg(lon_an - lon_ms[:, None]) / DEG_PER_HOUR_ANGLE
    ltan = np.mod(hours, HOURS_PER_DAY)
    ltdn = np.mod(ltan + HOURS_PER_DAY / 2, HOURS_PER_DAY)
    assert np.abs(ltan - 22.5).max() < 3 / 60  # within 3 minutes
    assert np.abs(ltdn - 10.5).max() < 3 / 60
