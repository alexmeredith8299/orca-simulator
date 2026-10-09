"""M2: quaternion conventions (scalar first, body to inertial)."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from expand3.vehicle.attitude import (
    dcm_to_quat,
    quat_angle_rad,
    quat_conj,
    quat_derivative,
    quat_from_axis_angle,
    quat_mul,
    quat_normalize,
    quat_rotate,
    quat_to_dcm,
)

quats = st.lists(st.floats(-1, 1), min_size=4, max_size=4).filter(
    lambda v: np.linalg.norm(v) > 0.1
).map(lambda v: quat_normalize(np.array(v)))


def test_convention_body_to_inertial() -> None:
    # +90 deg about inertial z maps body +x to inertial +y.
    q = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    np.testing.assert_allclose(q, [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    np.testing.assert_allclose(quat_rotate(q, np.array([1.0, 0, 0])), [0, 1, 0], atol=1e-15)
    np.testing.assert_allclose(quat_to_dcm(q) @ [1.0, 0, 0], [0, 1, 0], atol=1e-15)


@given(quats, quats)
def test_algebra(p: np.ndarray, q: np.ndarray) -> None:
    m = quat_to_dcm(q)
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(m) == pytest.approx(1.0)
    q2 = dcm_to_quat(m)
    assert quat_angle_rad(q, q2) < 1e-7
    np.testing.assert_allclose(quat_to_dcm(quat_mul(p, q)), quat_to_dcm(p) @ m, atol=1e-12)
    assert quat_angle_rad(quat_mul(q, quat_conj(q)), np.array([1.0, 0, 0, 0])) < 1e-7


def test_kinematics_constant_rate() -> None:
    # q_dot = 1/2 q x [0, w] with body rate w about body z for 1 s, integrated finely.
    w = np.array([0.0, 0.0, 0.3])
    q = np.array([1.0, 0, 0, 0])
    dt = 1e-4
    for _ in range(10000):
        q = quat_normalize(q + dt * quat_derivative(q, w))
    assert quat_angle_rad(q, quat_from_axis_angle(np.array([0, 0, 1.0]), 0.3)) < 1e-5


def test_one_second_step_matches_fine_step(params: object) -> None:
    from expand3.params import ParamView
    from expand3.vehicle.attitude import AttitudeState, inertia, step
    from expand3.vehicle.wheels import WheelArray

    view = ParamView(params, "SAT-1")  # type: ignore[arg-type]
    i_b, w = inertia(view), WheelArray(view)
    s0 = AttitudeState(np.array([1.0, 0, 0, 0]), np.array([0.08, -0.05, 0.1]),
                       np.array([0.5, -0.3, 0.2, 0.1]))
    tau = np.array([0.01, -0.02, 0.015])
    out = {}
    for dt in (1.0, 0.5, 0.01):
        s = s0
        for _ in range(round(30 / dt)):
            s, _, _ = step(i_b, w, s, tau, dt)
        out[dt] = s
    # 30 s at up to 0.18 rad/s: the 1 s step is within 1e-5 rad (0.0006 deg) of a fine step,
    # and the error is RK4 truncation (fourth order).
    err1 = quat_angle_rad(out[1.0].q, out[0.01].q)
    err_half = quat_angle_rad(out[0.5].q, out[0.01].q)
    assert err1 < 1e-5
    assert 12 < err1 / err_half < 20
    np.testing.assert_allclose(out[1.0].omega, out[0.01].omega, atol=1e-6)


def test_gravity_gradient() -> None:
    from expand3.vehicle.attitude import gravity_gradient_torque

    i_b = np.diag([16.0, 15.0, 11.0])
    r = np.array([6903.0, 0, 0])
    q = np.array([1.0, 0, 0, 0])  # nadir along body -x, a principal axis: no torque
    np.testing.assert_allclose(gravity_gradient_torque(i_b, r, q, 398600.4418), 0, atol=1e-18)
    q = quat_from_axis_angle(np.array([0, 0, 1.0]), np.deg2rad(10))
    tau = gravity_gradient_torque(i_b, r, q, 398600.4418)
    n = 398600.4418 / 6903.0**3
    # x (16) is not the minimum axis of the x-y pair, so the torque turns x further off nadir
    assert tau[2] == pytest.approx(3 * n * (16 - 15) * np.sin(np.deg2rad(20)) / 2, rel=1e-9)
