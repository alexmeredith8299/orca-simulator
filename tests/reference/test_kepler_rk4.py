"""M1: RK4 is fourth order against the analytic Kepler solution, and the element conversions
round-trip."""

import numpy as np
import pytest

from expand3.params import Params
from expand3.timebase import SeedDraws
from expand3.vehicle.orbit import (
    Forces,
    OrbitModel,
    elements_to_rv,
    kepler_propagate,
    propagate,
    rv_to_elements,
)

MU = 398600.4418


def two_body(params: Params, draws: SeedDraws) -> OrbitModel:
    return OrbitModel.from_params(
        params, ["SAT-1"], draws, pole=np.array([0.0, 0.0, 1.0]), forces=Forces(False, False)
    )


def test_elements_round_trip() -> None:
    r, v = elements_to_rv(7000.0, 0.01, 1.7, 0.3, 0.9, 2.1, MU)
    el = rv_to_elements(r, v, MU)
    assert float(el.a_km) == pytest.approx(7000.0)
    assert float(el.e) == pytest.approx(0.01)
    assert float(el.i_rad) == pytest.approx(1.7)
    assert float(el.raan_rad) == pytest.approx(0.3)
    assert float(el.argp_rad) == pytest.approx(0.9)
    assert float(el.nu_rad) == pytest.approx(2.1)


def test_kepler_period() -> None:
    a = 6903.137
    r0, v0 = elements_to_rv(a, 0.0001, 1.7, 0.3, 0.0, 0.0, MU)
    period = 2 * np.pi * np.sqrt(a**3 / MU)
    r1, v1 = kepler_propagate(r0, v0, np.array([period]), MU)
    np.testing.assert_allclose(r1[0], r0, atol=1e-6)
    np.testing.assert_allclose(v1[0], v0, atol=1e-9)


def test_rk4_fourth_order(params: Params, draws: SeedDraws) -> None:
    model = two_body(params, draws)
    r0, v0 = elements_to_rv(6903.137, 0.0001, np.deg2rad(97.497), 0.4, 0.0, 0.0, MU)
    y0 = np.concatenate([r0, v0])[None, :]
    horizon = 6000.0  # about one orbit
    errors = []
    for dt in (60.0, 30.0):
        n = int(horizon / dt)
        out = propagate(model, y0, dt, n, n)
        r_ref, _ = kepler_propagate(r0, v0, np.array([horizon]), MU)
        errors.append(np.linalg.norm(out[-1, 0, :3] - r_ref[0]))
    ratio = errors[0] / errors[1]
    assert 13 < ratio < 19, (errors, ratio)
