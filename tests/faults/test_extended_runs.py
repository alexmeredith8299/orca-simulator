"""Extended faults (Phase A) in paired mission runs: forced fault vs no fault, same seed. A
sensor fault must leave the truth of every vehicle and the measurements of the others untouched,
and show its signature in the faulted sensor after arm only."""

import numpy as np
import pytest

from expand3.faults import FaultConfig
from expand3.mission.thread import MissionRun, run_mission
from expand3.params import Params
from expand3.units import S_PER_HOUR

pytestmark = pytest.mark.slow
SEED = 20270
LEAD_S = 10_000.0
DURATION_S = 36 * S_PER_HOUR
IMU_FAULTS = ["gyro_bias", "gyro_scale", "gyro_noise", "gyro_stuck"]
GNSS_FAULTS = ["gnss_jump", "gnss_noise"]


@pytest.fixture(scope="module")
def nominal(params: Params) -> MissionRun:
    return run_mission(params, SEED, duration_s=DURATION_S, fault=FaultConfig(mode="none"))


@pytest.fixture(scope="module")
def faulted(params: Params) -> dict[str, MissionRun]:
    return {
        fid: run_mission(params, SEED, duration_s=DURATION_S,
                         fault=FaultConfig(mode="forced", fault_id=fid, lead_s=LEAD_S))
        for fid in [*IMU_FAULTS, *GNSS_FAULTS]
    }


@pytest.mark.parametrize("fid", IMU_FAULTS)
def test_gyro_faults_change_only_the_faulted_imu_after_arm(
        params: Params, nominal: MissionRun, faulted: dict[str, MissionRun], fid: str) -> None:
    run = faulted[fid]
    assert run.fault is not None
    sat = params.faults.extended[fid].vehicle
    assert (run.fault.fault_id, run.fault.vehicle) == (fid, sat)
    for other in run.sats:  # truth is sensor-independent: nothing physical moves
        np.testing.assert_array_equal(run.omega[:, run.sats.index(other)],
                                      nominal.omega[:, nominal.sats.index(other)])
        if other != sat:
            np.testing.assert_array_equal(run.signals[other].imu.meas,
                                          nominal.signals[other].imu.meas)
    a, b = run.signals[sat].imu, nominal.signals[sat].imu
    before = a.t_s < run.fault.t_arm_s
    np.testing.assert_array_equal(a.meas[before], b.meas[before])
    assert not np.array_equal(a.meas[~before], b.meas[~before])
    np.testing.assert_array_equal(a.truth, b.truth)


def test_signatures(params: Params, nominal: MissionRun, faulted: dict[str, MissionRun]) -> None:
    def err(fid: str) -> np.ndarray:
        run = faulted[fid]
        sat = params.faults.extended[fid].vehicle
        s = run.signals[sat].imu
        after = s.t_s >= run.fault.t_arm_s  # type: ignore[union-attr]
        return (s.meas - s.truth)[after]

    assert err("gyro_bias").mean() == pytest.approx(5e-4, rel=0.05)  # 10 sigma, all axes
    sigma = params.links.signals["imu"].model_extra["sigma_rad_s"]  # type: ignore[index]
    e = err("gyro_noise")
    assert e[: len(e) // 4].std() < 3.0 * sigma  # the ramp has barely started
    assert e[-len(e) // 8:].std() > 4.0 * sigma  # and is at or near x5 at the end
    stuck = faulted["gyro_stuck"].signals["SAT-3"].imu
    after = stuck.t_s >= faulted["gyro_stuck"].fault.t_arm_s  # type: ignore[union-attr]
    assert np.ptp(stuck.meas[after], axis=0).max() == 0.0


@pytest.mark.parametrize("fid", GNSS_FAULTS)
def test_gnss_faults_change_only_the_faulted_fixes_after_arm(
        params: Params, nominal: MissionRun, faulted: dict[str, MissionRun], fid: str) -> None:
    run = faulted[fid]
    assert run.fault is not None
    sat = params.faults.extended[fid].vehicle
    for other in run.sats:  # the fixes are sensors: truth and the crosslink states are untouched
        k = run.sats.index(other)
        np.testing.assert_array_equal(run.r_km[:, k], nominal.r_km[:, k])
        if other != sat:
            np.testing.assert_array_equal(run.signals[other].gnss_r.meas,
                                          nominal.signals[other].gnss_r.meas)
    assert run.crosslink is not None
    assert nominal.crosslink is not None
    np.testing.assert_array_equal(run.crosslink.r_meas_km, nominal.crosslink.r_meas_km)
    a, b = run.signals[sat].gnss_r, nominal.signals[sat].gnss_r
    before = a.t_s < run.fault.t_arm_s
    np.testing.assert_array_equal(a.meas[before], b.meas[before])
    assert not np.array_equal(a.meas[~before], b.meas[~before])


def test_gnss_signatures(params: Params, faulted: dict[str, MissionRun]) -> None:
    run = faulted["gnss_jump"]
    s = run.signals["SAT-2"].gnss_r
    after = s.t_s >= run.fault.t_arm_s  # type: ignore[union-attr]
    e_m = (s.meas - s.truth)[after] * 1000.0
    assert e_m.mean() == pytest.approx(10.0, abs=0.5)  # every axis, 1.5 m noise
    run = faulted["gnss_noise"]
    s = run.signals["SAT-3"].gnss_r
    e_m = (s.meas - s.truth) * 1000.0
    after = s.t_s >= run.fault.t_arm_s  # type: ignore[union-attr]
    assert e_m[~after].std() == pytest.approx(1.5, rel=0.15)
    assert e_m[after][-len(e_m[after]) // 8:].std() > 4.0 * 1.5  # near x5 at the end of the ramp
