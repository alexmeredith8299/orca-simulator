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
RADIO_FAULTS = ["tx_fade"]
MAG_FAULTS = ["mag_bias", "mag_noise", "mag_coupling"]


@pytest.fixture(scope="module")
def nominal(params: Params) -> MissionRun:
    return run_mission(params, SEED, duration_s=DURATION_S, fault=FaultConfig(mode="none"))


@pytest.fixture(scope="module")
def faulted(params: Params) -> dict[str, MissionRun]:
    return {
        fid: run_mission(params, SEED, duration_s=DURATION_S,
                         fault=FaultConfig(mode="forced", fault_id=fid, lead_s=LEAD_S))
        for fid in [*IMU_FAULTS, *GNSS_FAULTS, *RADIO_FAULTS, *MAG_FAULTS]
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


def test_nav_solution_follows_the_fixes_not_the_truth(
        nominal: MissionRun, faulted: dict[str, MissionRun]) -> None:
    """Every vehicle has a navigation solution at each fix. Nominally it is closer to the truth
    than the fix is; under gnss_jump the faulted vehicle's solution carries the offset, and the
    other vehicles' solutions are untouched."""
    for sat in nominal.sats:
        s = nominal.signals[sat]
        np.testing.assert_array_equal(s.nav_r.t_s, s.gnss_r.t_s)
        fix = np.linalg.norm(s.gnss_r.meas - s.gnss_r.truth, axis=1)[60:]
        nav = np.linalg.norm(s.nav_r.meas - s.nav_r.truth, axis=1)[60:]
        assert nav.mean() < fix.mean()
    run = faulted["gnss_jump"]
    for other in run.sats:
        if other != "SAT-2":
            np.testing.assert_array_equal(run.signals[other].nav_r.meas,
                                          nominal.signals[other].nav_r.meas)
    s = run.signals["SAT-2"]
    late = s.nav_r.t_s >= run.fault.t_arm_s + 3600.0  # type: ignore[union-attr]
    off = (s.nav_r.meas - s.nav_r.truth)[late].mean(axis=0) * 1000.0
    assert np.all(off > 5.0)  # of the 10 m on each axis


def test_tx_fade_lowers_the_senders_readings_after_arm(
        params: Params, nominal: MissionRun, faulted: dict[str, MissionRun]) -> None:
    """The fade changes only RSSI: the sender's outgoing readings drop by its depth after arm,
    nothing else on the link or in the vehicle moves, and no signal of the sender reports it."""
    run = faulted["tx_fade"]
    assert run.fault is not None
    sat = params.faults.extended["tx_fade"].vehicle
    assert run.crosslink is not None
    assert nominal.crosslink is not None
    depth = params.faults.extended["tx_fade"].set_to
    for j, (a, _b) in enumerate(run.crosslink.pairs):
        ok = run.crosslink.delivered[:, j]
        diff = (run.crosslink.rssi_dBm - nominal.crosslink.rssi_dBm)[ok, j]
        after = run.crosslink.t_s[ok] >= run.fault.t_arm_s + 3600.0
        if a == sat:
            assert diff[after].mean() < -0.8 * depth  # the step has reached full depth
            assert np.allclose(diff[run.crosslink.t_s[ok] < run.fault.t_arm_s], 0.0)
        else:
            assert np.allclose(diff, 0.0)
    np.testing.assert_array_equal(run.crosslink.delivered, nominal.crosslink.delivered)
    np.testing.assert_array_equal(run.crosslink.r_meas_km, nominal.crosslink.r_meas_km)


def _resid_abs(run: MissionRun, sat: str) -> np.ndarray:
    """Measured field magnitude minus the vehicle's model magnitude, per magnetometer sample."""
    sig = run.signals[sat]
    return np.linalg.norm(sig.mag.meas, axis=1) - sig.mag_model_abs.meas


def test_the_true_field_carries_a_disturbance_common_to_the_cluster(
        params: Params, nominal: MissionRun) -> None:
    """The sensor reads the true field to its own noise, the model lacks the disturbance, and
    that part of the residual is shared by cluster mates and not by the other cluster."""
    sigma = params.links.signals["magnetometer"].model_extra["sigma_nT"]  # type: ignore[index]
    for sat in nominal.sats:
        sig = nominal.signals[sat]
        assert np.abs(np.nanstd(sig.mag.meas - sig.mag.truth) / sigma - 1.0) < 0.05
        assert np.isnan(sig.mag_model.meas).any()  # no tracker fix on some samples
    rms = np.sqrt(np.mean(_resid_abs(nominal, "SAT-1") ** 2))
    assert 100.0 < rms < 600.0  # about the disturbance rms along the field
    a, b, c = (_resid_abs(nominal, s) for s in ("SAT-1", "SAT-2", "SAT-5"))
    assert np.corrcoef(a, b)[0, 1] > 0.5
    assert abs(np.corrcoef(a, c)[0, 1]) < np.corrcoef(a, b)[0, 1]


@pytest.mark.parametrize("fid", MAG_FAULTS)
def test_mag_faults_change_only_the_faulted_magnetometer_after_arm(
        params: Params, nominal: MissionRun, faulted: dict[str, MissionRun], fid: str) -> None:
    run = faulted[fid]
    assert run.fault is not None
    sat = params.faults.extended[fid].vehicle
    for other in run.sats:  # the field and the vehicle are the same; only the sensor differs
        np.testing.assert_array_equal(run.signals[other].mag.truth,
                                      nominal.signals[other].mag.truth)
        if other != sat:
            np.testing.assert_array_equal(run.signals[other].mag.meas,
                                          nominal.signals[other].mag.meas)
    a, b = run.signals[sat].mag, nominal.signals[sat].mag
    before = a.t_s < run.fault.t_arm_s
    np.testing.assert_array_equal(a.meas[before], b.meas[before])
    assert not np.array_equal(a.meas[~before], b.meas[~before])


def test_mag_fault_signatures(params: Params, nominal: MissionRun,
                              faulted: dict[str, MissionRun]) -> None:
    run = faulted["mag_bias"]
    sat = params.faults.extended["mag_bias"].vehicle
    d = run.signals[sat].mag.meas - nominal.signals[sat].mag.meas
    late = run.signals[sat].mag.t_s >= run.fault.t_arm_s  # type: ignore[union-attr]
    np.testing.assert_allclose(d[late], 300.0, atol=1e-9)
    run = faulted["mag_noise"]
    sat = params.faults.extended["mag_noise"].vehicle
    d = run.signals[sat].mag.meas - nominal.signals[sat].mag.meas
    late = run.signals[sat].mag.t_s >= run.fault.t_arm_s  # type: ignore[union-attr]
    assert d[late][-len(d[late]) // 8:].std() > 4.0 * d[~late].std() + 1.0  # near x5 at the end
    run = faulted["mag_coupling"]
    sat = params.faults.extended["mag_coupling"].vehicle
    d = (run.signals[sat].mag.meas - nominal.signals[sat].mag.meas)[:, 0]
    i = run.signals[sat].mag_model_abs.t_s
    amps = np.interp(i, run.t_s, run.eps[sat].i_A)
    late = i >= run.fault.t_arm_s  # type: ignore[union-attr]
    np.testing.assert_allclose(d[late], 100.0 * amps[late], atol=1e-6)
