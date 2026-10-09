"""M1: the shadow model's eclipse fraction per orbit matches the beta-angle formula."""

import numpy as np
import pytest

from expand3.env.eclipse import shadow_fraction
from expand3.env.sun import sun_eci
from expand3.params import Params
from expand3.timebase import SeedDraws
from expand3.vehicle.orbit import initial_states, kepler_propagate


def test_eclipse_fraction_vs_beta(params: Params, draws: SeedDraws) -> None:
    c = params.scenario.constants
    y0 = initial_states(params, ["SAT-1"], draws.epoch_utc)[0]
    r0, v0 = y0[:3], y0[3:]
    a = params.scenario.orbit.a_km
    period = 2 * np.pi * np.sqrt(a**3 / c.mu_km3_s2)
    t = np.arange(0.0, period, 1.0)
    r, _ = kepler_propagate(r0, v0, t, c.mu_km3_s2)
    sun_u, sun_d = sun_eci(draws.epoch_utc, np.array([0.0]))
    s = shadow_fraction(r, sun_u[0], sun_d[0], c.R_E_km, c.R_sun_km)

    h_hat = np.cross(r0, v0) / np.linalg.norm(np.cross(r0, v0))
    beta = np.arcsin(h_hat @ sun_u[0])
    h = a - c.R_E_km
    assert abs(beta) < np.arcsin(c.R_E_km / a)  # this orbit does see eclipses
    expected = np.arccos(np.sqrt(h**2 + 2 * c.R_E_km * h) / (a * np.cos(beta))) / np.pi
    assert np.mean(1 - s) == pytest.approx(expected, rel=0.01)
    assert set(np.unique(s[(s == 0) | (s == 1)])) == {0.0, 1.0}
