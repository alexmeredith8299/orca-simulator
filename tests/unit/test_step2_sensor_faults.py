"""Extended faults of Step 2, sensor group (docs/extended-faults-plan.md): st_dropout,
st_false_star, st_stuck and gnss_outage act on the reading of one sensor and nothing else."""

import numpy as np

from expand3.params import Params, ParamView
from expand3.sensors import gnss, star_tracker
from expand3.timebase import rng
from expand3.vehicle.attitude import quat_conj, quat_mul

N = 400
ARCSEC_DEG = 3600.0


def tracker(params: Params, overrides: dict[str, float], sat: str = "SAT-2"):  # type: ignore[no-untyped-def]
    view = ParamView(params, sat)
    view.overrides.update(overrides)
    t = np.arange(N) * 30.0
    q = np.tile([1.0, 0, 0, 0], (N, 1))
    sun = np.tile([0.0, 0, 1], (N, 1))
    r = np.tile([0.0, 0, -1e6], (N, 1))  # far from the Earth, Sun on +z: always in view
    return star_tracker(rng(1, f"noise.star_tracker.{sat}"), view, t, q, r, sun)


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dq = np.array([quat_mul(quat_conj(x), y) for x, y in zip(a, b, strict=True)])
    dq = np.where(dq[:, :1] < 0, -dq, dq)
    return np.rad2deg(2 * np.linalg.norm(dq[:, 1:], axis=1))


def test_the_fault_paths_exist_and_are_off(params: Params) -> None:
    view = ParamView(params, "SAT-2")
    for path in ("sensor_faults.st.dropout", "sensor_faults.st.offset_deg",
                 "sensor_faults.st.stuck", "sensor_faults.gnss.outage"):
        assert view.get(path) == 0.0
    for fid, sat, shapes in [("st_dropout", "SAT-3", ["step", "intermittent"]),
                             ("st_false_star", "SAT-2", ["step"]),
                             ("st_stuck", "SAT-4", ["step"]),
                             ("gnss_outage", "SAT-4", ["step", "intermittent"])]:
        f = params.faults.extended[fid]
        assert f.vehicle == sat
        assert (f.shapes or [f.profile]) == shapes


def test_a_dropout_removes_every_fix(params: Params) -> None:
    base = tracker(params, {})
    out = tracker(params, {"sensor_faults.st.dropout": 1.0})
    assert base.available.all()
    assert not out.available.any()
    assert np.isnan(out.q_meas).all()
    np.testing.assert_array_equal(out.q_true, base.q_true)


def test_a_false_star_offsets_every_valid_fix_by_the_set_angle(params: Params) -> None:
    base = tracker(params, {})
    out = tracker(params, {"sensor_faults.st.offset_deg": 0.5})
    assert out.available.all()  # the fix is still marked valid
    off = angle_deg(base.q_meas, out.q_meas)
    np.testing.assert_allclose(off, 0.5, atol=2.0 / ARCSEC_DEG)  # the tracker's own noise is ~2"
    # one fixed rotation: the axis does not change from fix to fix
    rel = [quat_mul(quat_conj(a), b) for a, b in zip(base.q_meas, out.q_meas, strict=True)]
    axes = np.array([r[1:] / np.linalg.norm(r[1:]) for r in rel])
    assert np.abs(axes - axes.mean(axis=0)).max() < 0.05


def test_a_stuck_tracker_repeats_its_last_fix_and_keeps_its_availability(params: Params) -> None:
    view = ParamView(params, "SAT-2")
    view.schedule["sensor_faults.st.stuck"] = lambda t: float(t >= 6000.0)
    t = np.arange(N) * 30.0
    q = np.tile([1.0, 0, 0, 0], (N, 1))
    sun = np.tile([0.0, 0, 1], (N, 1))
    r = np.tile([0.0, 0, -1e6], (N, 1))
    out = star_tracker(rng(1, "noise.star_tracker.SAT-2"), view, t, q, r, sun)
    after = t >= 6000.0
    assert np.ptp(out.q_meas[after], axis=0).max() == 0.0
    assert np.ptp(out.q_meas[~after], axis=0).max() > 0.0
    np.testing.assert_array_equal(out.q_meas[after][0], out.q_meas[~after][-1])
    assert out.available.all()


def test_a_gnss_outage_leaves_nan_fixes_on_the_same_grid(params: Params) -> None:
    view = ParamView(params, "SAT-4")
    view.schedule["sensor_faults.gnss.outage"] = lambda t: float(t >= 6000.0)
    t = np.arange(N) * 30.0
    r = np.tile([7000.0, 0, 0], (N, 1))
    v = np.tile([0.0, 7.5, 0], (N, 1))
    pos, vel = gnss(rng(1, "noise.gnss.SAT-4"), view, t, r, v)
    base_pos, _ = gnss(rng(1, "noise.gnss.SAT-4"), ParamView(params, "SAT-4"), t, r, v)
    out = pos.t_s >= 6000.0
    np.testing.assert_array_equal(pos.t_s, base_pos.t_s)
    assert np.isnan(pos.meas[out]).all()
    assert np.isnan(vel.meas[out]).all()
    np.testing.assert_array_equal(pos.meas[~out], base_pos.meas[~out])  # same draws before
    np.testing.assert_array_equal(pos.truth, base_pos.truth)
