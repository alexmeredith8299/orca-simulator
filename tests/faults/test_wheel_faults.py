"""M2: F3 (bearing friction) and F4 (torque constant) shapes through ParamView overrides."""

import numpy as np
import pytest

from expand3.mission.slew import slew
from expand3.params import Params, ParamView
from expand3.vehicle.attitude import AttitudeState, inertia, quat_from_axis_angle, step
from expand3.vehicle.wheels import WheelArray


def test_f3_friction_decays_h1_exponentially(params: Params) -> None:
    view = ParamView(params, "SAT-3")
    view.overrides["wheels.c[0]"] = 0.02
    w = WheelArray(view)
    h0 = np.array([2.0, -1.0, 0.5, 1.5])
    h, dt = h0.copy(), 1.0
    for n in range(1, 301):
        h = w.advance(h, np.zeros(4), dt)
        assert h[0] == pytest.approx(h0[0] * np.exp(-0.02 * n * dt), rel=1e-6)
        np.testing.assert_array_equal(h[1:], h0[1:])
    # friction torque bleeds wheel 1 momentum into the body: tau_w = A (c o h) on average
    inertia_b = inertia(view)
    s = AttitudeState(np.array([1.0, 0, 0, 0]), np.zeros(3), h0.copy())
    s1, _, tau = step(inertia_b, w, s, np.zeros(3), dt)
    np.testing.assert_allclose(tau, -w.A @ (s1.h - h0) / dt)
    assert np.linalg.norm(s1.omega) > 0


def test_f4_halves_wheel2_torque(params: Params) -> None:
    nominal = WheelArray(ParamView(params, "SAT-3"))
    view = ParamView(params, "SAT-3")
    view.overrides["wheels.k[1]"] = 0.5
    faulty = WheelArray(view)
    u = np.array([0.0, 0.05, 0.0, 0.0])
    h = np.zeros(4)
    t_nom = nominal.body_torque(h, nominal.advance(h, u, 1.0), 1.0)
    t_f4 = faulty.body_torque(h, faulty.advance(h, u, 1.0), 1.0)
    np.testing.assert_allclose(t_f4, 0.5 * t_nom)
    np.testing.assert_allclose(t_nom, nominal.A[:, 1] * 0.05)


def test_f4_slew_still_completes_but_works_harder(params: Params) -> None:
    q0 = np.array([1.0, 0, 0, 0])
    q1 = quat_from_axis_angle(np.array([1.0, 1.0, 0.3]) / np.linalg.norm([1.0, 1.0, 0.3]), 1.5)
    nominal = slew(ParamView(params, "SAT-3"), q0, q1)
    view = ParamView(params, "SAT-3")
    view.overrides["wheels.k[1]"] = 0.5
    faulty = slew(view, q0, q1)
    assert nominal.converged
    assert faulty.converged
    assert faulty.peak_tracking_error_deg > 3 * nominal.peak_tracking_error_deg
    assert faulty.duration_s >= nominal.duration_s
    effort_nom = np.abs(nominal.u).sum()
    effort_f4 = np.abs(faulty.u).sum()
    assert effort_f4 != pytest.approx(effort_nom, rel=1e-3)
