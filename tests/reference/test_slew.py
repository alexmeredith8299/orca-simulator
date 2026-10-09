"""M2: rest-to-rest slews converge and take the bang-coast-bang time."""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from expand3.mission.slew import plan_slew, slew
from expand3.params import ParamView, load_params
from expand3.vehicle.attitude import quat_angle_rad, quat_from_axis_angle, quat_normalize

PARAMS = load_params(Path(__file__).resolve().parents[2] / "params")

quats = (
    st.lists(st.floats(-1, 1), min_size=4, max_size=4)
    .filter(lambda v: np.linalg.norm(v) > 0.1)
    .map(lambda v: quat_normalize(np.array(v)))
)


@pytest.mark.parametrize("sat", ["SAT-1", "SAT-5"])
@settings(max_examples=40, deadline=None)
@given(q0=quats, q1=quats)
def test_random_slews_converge(sat: str, q0: np.ndarray, q1: np.ndarray) -> None:
    view = ParamView(PARAMS, sat)
    res = slew(view, q0, q1)
    assert res.converged
    assert res.final_error_deg < 0.1
    assert res.final_rate_rad_s < 1e-4
    assert np.rad2deg(quat_angle_rad(res.q[-1], q1)) < 0.1
    np.testing.assert_allclose(np.linalg.norm(res.q, axis=1), 1.0, atol=1e-12)
    assert np.all(np.abs(res.h) <= view.get("wheels.h_max_Nms") + 1e-12)
    assert np.all(np.abs(res.u) <= view.get("wheels.tau_max_Nm") + 1e-12)
    if res.plan.angle_rad > np.deg2rad(10):
        assert res.duration_s == pytest.approx(res.plan.predicted_s, rel=0.10)


@pytest.mark.parametrize("sat", ["SAT-1", "SAT-5"])
def test_ninety_degree_slew(sat: str) -> None:
    view = ParamView(PARAMS, sat)
    q0 = np.array([1.0, 0, 0, 0])
    q1 = quat_from_axis_angle(np.array([1.0, 0, 0]), np.pi / 2)
    plan = plan_slew(view, q0, q1)
    res = slew(view, q0, q1)
    print(f"\n{sat}: 90 deg about +x, plan {plan.predicted_s:.1f} s, flew {res.duration_s:.1f} s")
    assert res.converged
    assert res.duration_s == pytest.approx(plan.predicted_s, rel=0.10)
    assert len(res.t_s) == len(res.h) == len(res.u)
    assert res.power_W.max() > 0
