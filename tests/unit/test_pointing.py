"""M2: pointing law. Spec: Model: orbit and attitude (pointing law)."""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from expand3.mission.pointing import burn, sun_hold, take
from expand3.params import ParamView, load_params
from expand3.vehicle.attitude import quat_rotate

PARAMS = load_params(Path(__file__).resolve().parents[2] / "params")
X, Z = np.array([1.0, 0, 0]), np.array([0.0, 0, 1])

unit = (
    st.lists(st.floats(-1, 1), min_size=3, max_size=3)
    .filter(lambda v: np.linalg.norm(v) > 0.1)
    .map(lambda v: np.array(v) / np.linalg.norm(v))
)


def angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b))


def rodrigues(v: np.ndarray, axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    c, s = np.cos(angles)[:, None], np.sin(angles)[:, None]
    return v * c + np.cross(axis, v) * s + axis * (axis @ v) * (1 - c)


@settings(max_examples=200)
@given(unit, unit)
def test_sun_hold(sun: np.ndarray, normal: np.ndarray) -> None:
    if abs(sun @ normal) > 0.999:
        return
    q = sun_hold(ParamView(PARAMS, "SAT-1"), sun, normal)
    assert angle(quat_rotate(q, Z), sun) < 1e-9
    # roll convention: +x is the orbit normal projected onto the plane normal to the Sun
    proj = normal - (normal @ sun) * sun
    assert angle(quat_rotate(q, X), proj) < 1e-9


@pytest.mark.parametrize("sat", ["SAT-2", "SAT-5"])
@settings(max_examples=100, deadline=None)
@given(a=unit, sun=unit)
def test_burn_matches_brute_force(sat: str, a: np.ndarray, sun: np.ndarray) -> None:
    if abs(a @ sun) > 0.999:
        return
    view = ParamView(PARAMS, sat)
    q = burn(view, a, sun)
    b = view.array("thruster.axis_b")
    assert angle(quat_rotate(q, b), a) < 1e-9
    closed_form = angle(quat_rotate(q, Z), sun)
    rolls = np.deg2rad(np.arange(0.0, 360.0, 0.005))
    z_scan = rodrigues(quat_rotate(q, Z), a, rolls)
    brute = np.arccos(np.clip(z_scan @ sun, -1, 1)).min()
    assert abs(np.rad2deg(closed_form - brute)) < 0.01


def test_take_boresight_hits_target() -> None:
    view = ParamView(PARAMS, "SAT-3")
    rng = np.random.default_rng(2)
    for _ in range(50):
        m = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        if np.linalg.det(m) < 0:
            m[:, 0] *= -1
        r = rng.normal(size=3)
        r = 6903.0 * r / np.linalg.norm(r)
        target_ecef = m @ (r * 6378.0 / 6903.0 + rng.normal(size=3) * 300)
        q = take(view, r, target_ecef, m, Z)
        los = m.T @ target_ecef - r
        assert angle(quat_rotate(q, -Z), los) < 1e-6
