"""M2: wheel allocation, saturation and torquing power. Spec: Model: actuators."""

import numpy as np
import pytest

from expand3.params import Params, ParamView
from expand3.vehicle.wheels import WheelArray

BUSES = [("SAT-1", 10.0), ("SAT-5", 8.0)]


@pytest.fixture(params=["SAT-1", "SAT-5"])
def wheels(request: pytest.FixtureRequest, params: Params) -> WheelArray:
    return WheelArray(ParamView(params, request.param))


def test_geometry(wheels: WheelArray) -> None:
    assert wheels.A.shape == (3, 4)
    np.testing.assert_allclose(np.linalg.norm(wheels.A, axis=0), 1.0)
    np.testing.assert_allclose(np.rad2deg(np.arccos(wheels.A[2])), 54.7)
    assert np.linalg.matrix_rank(wheels.A) == 3


def test_unsaturated_allocation_is_exact_and_min_norm(wheels: WheelArray) -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        tau = rng.normal(size=3) * 0.02
        u = wheels.allocate(tau)
        assert np.all(np.abs(u) < wheels.tau_max)
        np.testing.assert_allclose(wheels.A @ u, tau, atol=1e-15)
        # min norm: u lies in the row space of A (orthogonal to the null space)
        null = np.linalg.svd(wheels.A)[2][-1]
        assert abs(u @ null) < 1e-15


def test_clipping(wheels: WheelArray) -> None:
    u = wheels.allocate(np.array([5.0, -3.0, 2.0]))
    assert np.all(np.abs(u) <= wheels.tau_max)
    assert np.isclose(np.abs(u).max(), wheels.tau_max)


def test_saturation_holds_h_max(wheels: WheelArray) -> None:
    h = np.zeros(4)
    tau = np.array([0.0, 0.0, 0.05])
    dt = 1.0
    delivered = []
    for _ in range(2000):
        u = wheels.allocate(tau)
        h_new = wheels.advance(h, u, dt)
        delivered.append(wheels.body_torque(h, h_new, dt))
        h = h_new
        assert np.all(np.abs(h) <= wheels.h_max + 1e-12)
    np.testing.assert_allclose(np.abs(h), wheels.h_max)
    np.testing.assert_allclose(delivered[0], tau, atol=1e-15)
    np.testing.assert_allclose(delivered[-1], 0.0, atol=1e-15)


@pytest.mark.parametrize(("sat", "watts"), BUSES)
def test_torquing_power(params: Params, sat: str, watts: float) -> None:
    w = WheelArray(ParamView(params, sat))
    assert w.torquing_power_w(np.zeros(4)) == 0
    assert w.torquing_power_w(np.array([0.01, 0.0, -0.02, 0.0])) == pytest.approx(2 * watts)
    assert w.torquing_power_w(w.allocate(np.array([0.01, 0.02, 0.03]))) == pytest.approx(4 * watts)


def test_k_and_c_come_from_paramview(params: Params) -> None:
    view = ParamView(params, "SAT-3")
    view.overrides["wheels.k[1]"] = 0.5
    view.overrides["wheels.c[0]"] = 0.02
    w = WheelArray(view)
    np.testing.assert_array_equal(w.k, [1, 0.5, 1, 1])
    np.testing.assert_array_equal(w.c, [0.02, 0, 0, 0])
