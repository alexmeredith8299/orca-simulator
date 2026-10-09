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
