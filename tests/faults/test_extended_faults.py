"""Extended faults (docs/extended-faults-plan.md, Phase A): the table, the extra onset shapes, the
seeded draw over all 46 faults, and the sensor fault effects. The effects here are checked on
synthetic arrays; paired mission runs are in test_extended_runs.py."""

import numpy as np
import pytest

from expand3.faults import (
    Injector,
    draw_fault,
    draw_fault_extended,
    episodes,
    onset,
)
from expand3.params import Params, ParamView
from expand3.sensors import gnss, imu
from expand3.timebase import rng

TICK = 30.0
IMU_FAULTS = ["gyro_bias", "gyro_scale", "gyro_noise", "gyro_stuck"]
GNSS_FAULTS = ["gnss_jump", "gnss_noise"]
DONE = [*IMU_FAULTS, *GNSS_FAULTS]  # extended with each step of Phase A


def test_extended_table_is_separate_from_the_spec_ten(params: Params) -> None:
    assert sorted(params.faults.faults) == sorted(f"F{i}" for i in range(1, 11))
    assert set(DONE) <= set(params.faults.extended)
    assert not set(params.faults.extended) & set(params.faults.faults)
    for fid in DONE:
        f = params.faults.extended[fid]
        assert f.vehicle in params.scenario.bus_of
        assert f.profile == (f.shapes or [f.profile])[0]


def test_every_vehicle_is_hit_by_some_extended_fault_by_the_end_of_phase_a(
        params: Params) -> None:
    hit = {f.vehicle for f in params.faults.extended.values()}
    assert hit <= set(params.scenario.bus_of)  # and, once Phase A is complete, equal to it


def test_exponential_profile_is_one_minus_exp(params: Params) -> None:
    f = params.faults.extended["gyro_noise"].model_copy(update={"profile": "exponential",
                                                                "t_r_s": 3600.0})
    t_arm = 1000.0
    t = np.array([0.0, 999.0, 1000.0, 1000.0 + 3600.0, 1000.0 + 7200.0])
    s = [onset(f, t_arm, float(x)) for x in t]
    assert s[:3] == [0.0, 0.0, 0.0]
    assert s[3] == pytest.approx(1 - np.exp(-1))
    assert s[4] == pytest.approx(1 - np.exp(-2))


def test_episodes_alternate_from_an_on_episode_at_arm_and_depend_on_the_seed(
        params: Params) -> None:
    f = params.faults.extended["gyro_noise"]
    assert f.t_on_s is not None
    assert f.t_off_s is not None
    a = episodes(f, 5000.0, rng(1, "draw.episodes"), horizon_s=2.0e6)
    again = episodes(f, 5000.0, rng(1, "draw.episodes"), horizon_s=2.0e6)
    other = episodes(f, 5000.0, rng(2, "draw.episodes"), horizon_s=2.0e6)
    np.testing.assert_array_equal(a, again)
    assert not np.array_equal(a[:5], other[:5])
    assert a[0] == 5000.0
    assert np.all(np.diff(a) > 0)
    on = np.sum(a[1::2] - a[0::2][: len(a[1::2])])
    total = a[-1] - a[0]
    assert on / total == pytest.approx(f.t_on_s / (f.t_on_s + f.t_off_s), rel=0.15)


def test_intermittent_onset_is_one_inside_episodes_only(params: Params) -> None:
    f = params.faults.extended["gyro_noise"].model_copy(update={"profile": "intermittent"})
    ep = np.array([5000.0, 5600.0, 7000.0, 7300.0])
    assert [onset(f, 5000.0, t, ep) for t in (4999.0, 5000.0, 5599.0, 5600.0, 6999.0, 7000.0,
                                               7299.0, 7300.0, 9000.0)] == [
        0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]


def test_draw_over_all_46_is_reproducible_and_leaves_the_spec_draw_alone(
        params: Params) -> None:
    ids = {draw_fault_extended(params, s)[0] for s in range(2000)}
    assert ids & set(params.faults.faults)
    assert ids & set(params.faults.extended)
    assert len(ids) == len(params.faults.faults) + len(params.faults.extended)
    assert draw_fault_extended(params, 7) == draw_fault_extended(params, 7)
    for seed in range(20):  # the spec draw is untouched: same streams, same ten
        ids10 = sorted(params.faults.faults, key=lambda f: int(f[1:]))
        assert draw_fault(params, seed)[0] == ids10[int(
            rng(seed, "draw.fault").integers(len(ids10)))]
        # the lead stream is shared, so a seed has one lead whichever set it draws from
        assert draw_fault_extended(params, seed)[1] == draw_fault(params, seed)[1]


def test_draw_picks_a_listed_shape(params: Params) -> None:
    for seed in range(200):
        fid, _, profile = draw_fault_extended(params, seed)
        f = params.fault(fid)
        assert profile in (f.shapes or [f.profile])


def test_injector_schedules_an_extended_fault(params: Params) -> None:
    inj = Injector(params, "gyro_bias", t0_s=50_000.0, lead_s=8000.0, tick_s=TICK,
                   profile="step", seed=3)
    f = params.faults.extended["gyro_bias"]
    view = ParamView(params, f.vehicle)
    inj.install(view)
    path = f.paths[0]
    assert view.at(0.0).get(path) == 0.0
    assert view.at(inj.t_arm_s).get(path) == pytest.approx(f.set_to)
    lab = inj.label(np.arange(0.0, 60_000.0, TICK))
    assert lab.fault_id == "gyro_bias"
    assert lab.profile == "step"


# ---- IMU effects on the measurement ---------------------------------------------------------
def _imu(params: Params, schedule: dict[str, float] | None = None,
         arm: float = 3000.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    view = ParamView(params, "SAT-1")
    for path, value in (schedule or {}).items():
        view.schedule[path] = lambda t, v=value, p=path: (
            v if t >= arm else params_nominal[p])
    t = np.arange(0.0, 6000.0, TICK)
    omega = np.stack([1e-3 * np.sin(t / 700.0), 2e-3 * np.cos(t / 900.0), 3e-4 + 0 * t], axis=1)
    out = imu(rng(5, "noise.imu.SAT-1"), view, t, omega)
    return t, omega, out.meas


params_nominal = {"sensor_faults.imu.bias_rad_s": 0.0, "sensor_faults.imu.scale": 1.0,
                  "sensor_faults.imu.noise_mult": 1.0, "sensor_faults.imu.stuck": 0.0}


def test_nominal_imu_is_truth_plus_the_same_noise_as_before(params: Params) -> None:
    t, omega, meas = _imu(params)
    view = ParamView(params, "SAT-1")
    noise = rng(5, "noise.imu.SAT-1").normal(0.0, view.get("signals.imu.sigma_rad_s"),
                                             (len(t), 3))
    np.testing.assert_array_equal(meas, omega + noise)


def test_bias_scale_and_noise_multiplier_act_after_arm_only(params: Params) -> None:
    t, omega, base = _imu(params)
    n = len(t)
    after = t >= 3000.0
    _, _, b = _imu(params, {"sensor_faults.imu.bias_rad_s": 5e-4})
    np.testing.assert_allclose((b - base)[after], 5e-4, atol=1e-15)
    np.testing.assert_array_equal(b[~after], base[~after])
    _, _, s = _imu(params, {"sensor_faults.imu.scale": 1.25})
    np.testing.assert_allclose((s - base)[after], 0.25 * omega[after], atol=1e-15)
    _, _, m = _imu(params, {"sensor_faults.imu.noise_mult": 5.0})
    noise = base - omega
    np.testing.assert_allclose((m - omega)[after], 5.0 * noise[after], atol=1e-15)
    assert n == len(m)


def test_stuck_gyro_holds_the_last_healthy_sample(params: Params) -> None:
    t, _, base = _imu(params)
    _, _, st = _imu(params, {"sensor_faults.imu.stuck": 1.0})
    after = t >= 3000.0
    last = int(np.flatnonzero(~after)[-1])
    np.testing.assert_array_equal(st[after], np.tile(base[last], (int(after.sum()), 1)))
    np.testing.assert_array_equal(st[~after], base[~after])


# ---- GNSS effects on the measurement ----------------------------------------------------------
def _gnss(params: Params, schedule: dict[str, float] | None = None, arm: float = 3000.0
          ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    view = ParamView(params, "SAT-2")
    for path, value in (schedule or {}).items():
        view.schedule[path] = lambda t, v=value, p=path: (
            v if t >= arm else gnss_nominal[p])
    t = np.arange(0.0, 6000.0, TICK)
    r = np.stack([7000.0 + 0 * t, 10.0 * t, 0 * t], axis=1)
    v = np.stack([0 * t, 7.5 + 0 * t, 0 * t], axis=1)
    pos, vel = gnss(rng(5, "noise.gnss.SAT-2"), view, t, r, v)
    return pos.t_s, (pos.meas - pos.truth, vel.meas - vel.truth)


gnss_nominal = {"sensor_faults.gnss.offset_m": 0.0, "sensor_faults.gnss.noise_mult": 1.0}


def test_gnss_offset_is_added_to_position_only_after_arm(params: Params) -> None:
    t, (dr0, dv0) = _gnss(params)
    _, (dr, dv) = _gnss(params, {"sensor_faults.gnss.offset_m": 10.0})
    after = t >= 3000.0
    np.testing.assert_allclose((dr - dr0)[after], 10.0e-3, atol=1e-12)  # km
    np.testing.assert_array_equal(dr[~after], dr0[~after])
    np.testing.assert_array_equal(dv, dv0)


def test_gnss_noise_multiplier_scales_position_and_velocity_after_arm(params: Params) -> None:
    t, (dr0, dv0) = _gnss(params)
    _, (dr, dv) = _gnss(params, {"sensor_faults.gnss.noise_mult": 5.0})
    after = t >= 3000.0
    np.testing.assert_allclose(dr[after], 5.0 * dr0[after], atol=1e-15)
    np.testing.assert_allclose(dv[after], 5.0 * dv0[after], atol=1e-15)
    np.testing.assert_array_equal(dr[~after], dr0[~after])
