"""M7: paired runs (forced fault vs forced no fault, same seed): isolation and signatures."""

import numpy as np
import pytest

from expand3.faults import FaultConfig
from expand3.mission.thread import MissionRun, run_mission
from expand3.params import Params
from expand3.units import S_PER_HOUR

pytestmark = pytest.mark.slow
SEED = 20270
LEAD_S = 10_000.0  # arms ~2.8 h before the take; SAT-4's burn (23.4-28.9 h) is still running
DURATION_S = 36 * S_PER_HOUR
FAULTS = [f"F{i}" for i in range(1, 11)]


@pytest.fixture(scope="module")
def nominal(params: Params) -> MissionRun:
    return run_mission(params, SEED, duration_s=DURATION_S, fault=FaultConfig(mode="none"))


@pytest.fixture(scope="module")
def faulted(params: Params) -> dict[str, MissionRun]:
    return {
        fid: run_mission(params, SEED, duration_s=DURATION_S,
                         fault=FaultConfig(mode="forced", fault_id=fid, lead_s=LEAD_S))
        for fid in FAULTS
    }


def arm_index(run: MissionRun) -> int:
    assert run.fault is not None
    return int(np.flatnonzero(run.t_s >= run.fault.t_arm_s)[0])


def test_labels(params: Params, faulted: dict[str, MissionRun], nominal: MissionRun) -> None:
    assert nominal.fault is None
    print()
    for fid, run in faulted.items():
        lab = run.fault
        assert lab is not None
        spec = params.faults.faults[fid]
        assert (lab.fault_id, lab.vehicle) == (fid, spec.vehicle)
        assert lab.lead_s == LEAD_S
        take = nominal.takes[spec.vehicle].t_take_s
        assert lab.t0_s == pytest.approx(take - LEAD_S)
        assert lab.t_arm_s == run.t_s[run.t_s >= lab.t0_s][0]
        assert np.all(lab.s[run.t_s < lab.t_arm_s] == 0)
        assert lab.s[-1] == 1.0
        for path in lab.paths:
            assert len(lab.theta[path]) == len(run.t_s)
        print(f"{fid} on {lab.vehicle}: arms at {lab.t_arm_s / 3600:.3f} h")


@pytest.mark.parametrize("fid", FAULTS)
def test_isolation(params: Params, faulted: dict[str, MissionRun], nominal: MissionRun,
                   fid: str) -> None:
    run = faulted[fid]
    sat = params.faults.faults[fid].vehicle
    k = run.sats.index(sat)
    n_arm = arm_index(run)
    for j, other in enumerate(run.sats):
        if other == sat:
            continue
        np.testing.assert_array_equal(run.r_km[:, j], nominal.r_km[:, j])
        np.testing.assert_array_equal(run.q[:, j], nominal.q[:, j])
        np.testing.assert_array_equal(run.eps[other].v_V, nominal.eps[other].v_V)
        np.testing.assert_array_equal(run.thermal[other].T_K, nominal.thermal[other].T_K)
        np.testing.assert_array_equal(run.saa[other].upsets, nominal.saa[other].upsets)
        np.testing.assert_array_equal(run.signals[other].gnss_r.meas,
                                      nominal.signals[other].gnss_r.meas)
        np.testing.assert_array_equal(run.signals[other].soc_estimate.meas,
                                      nominal.signals[other].soc_estimate.meas)
    pre = slice(0, n_arm)
    np.testing.assert_array_equal(run.r_km[pre, k], nominal.r_km[pre, k])
    np.testing.assert_array_equal(run.q[pre, k], nominal.q[pre, k])
    np.testing.assert_array_equal(run.eps[sat].v_V[pre], nominal.eps[sat].v_V[pre])
    np.testing.assert_array_equal(run.thermal[sat].T_K[pre], nominal.thermal[sat].T_K[pre])
    np.testing.assert_array_equal(run.saa[sat].upsets[pre], nominal.saa[sat].upsets[pre])
    a, b = run.crosslink, nominal.crosslink
    assert a is not None
    assert b is not None
    for j, (snd, rcv) in enumerate(a.pairs):
        if sat not in (snd, rcv):
            np.testing.assert_array_equal(a.delivered[:, j], b.delivered[:, j])
            np.testing.assert_array_equal(a.r_meas_km[:, j], b.r_meas_km[:, j])
        else:
            np.testing.assert_array_equal(a.delivered[pre, j], b.delivered[pre, j])


def test_f1_voltage_sag_grows(faulted: dict[str, MissionRun], nominal: MissionRun) -> None:
    run = faulted["F1"]
    assert run.fault is not None
    f, n = run.eps["SAT-2"], nominal.eps["SAT-2"]
    np.testing.assert_array_equal(f.E_J, n.E_J)  # R sets the voltage only
    np.testing.assert_array_equal(f.soc, n.soc)
    ocv = n.v_V - n.i_A * 0.08  # same SoC, so the same OCV in both runs
    np.testing.assert_allclose(f.v_V - f.i_A * run.fault.theta["eps.R_pack_ohm"], ocv, atol=1e-9)
    dis = (n.i_A < -0.5) & (run.t_s >= run.fault.t_arm_s)
    ratio = (ocv - f.v_V)[dis] / (ocv - n.v_V)[dis]
    t_dis = run.t_s[dis] - run.fault.t_arm_s
    assert ratio.max() > 2.8  # ~x3 once the 6 h ramp completes
    assert ratio[t_dis > 5 * 3600].mean() > ratio[t_dis < 2 * 3600].mean()
    print(f"\nF1: discharge sag x{ratio[t_dis < 1800].mean():.2f} in the first 30 min, "
          f"x{ratio[t_dis > 6 * 3600].mean():.2f} after the ramp")


def test_f2_string_open(faulted: dict[str, MissionRun], nominal: MissionRun) -> None:
    run = faulted["F2"]
    n_arm = arm_index(run)
    f, n = run.eps["SAT-2"], nominal.eps["SAT-2"]
    assert np.all(f.p_string_W[n_arm:, 0] == 0)
    sunny = n.p_string_W[n_arm:, 0] > 0
    np.testing.assert_allclose(f.p_string_W[n_arm:, 1:][sunny], n.p_string_W[n_arm:, 1:][sunny],
                               rtol=1e-3)
    drop = (n.p_gen_W - f.p_gen_W)[n_arm:][sunny]
    np.testing.assert_allclose(drop, n.p_string_W[n_arm:, 0][sunny], rtol=1e-3)


def wheel_residual(run: MissionRun, sat: str, wheel: int, k_w: float) -> tuple[np.ndarray, ...]:
    """Per slew step: dh + k u dt, which equals -c h dt under the spec's discrete update."""
    out_h, out_r = [], []
    for res in run.slews[sat]:
        h, u = res.h, res.u
        dh = np.diff(h[:, wheel])
        out_r.append(dh + k_w * u[1:, wheel] * 1.0)
        out_h.append(h[:-1, wheel])
    return np.concatenate(out_h), np.concatenate(out_r)


def test_f3_bearing_friction(faulted: dict[str, MissionRun], nominal: MissionRun) -> None:
    run = faulted["F3"]
    h, resid = wheel_residual(run, "SAT-3", 0, 1.0)
    np.testing.assert_allclose(resid, -0.02 * h, atol=1e-12)
    _, resid_n = wheel_residual(nominal, "SAT-3", 0, 1.0)
    np.testing.assert_allclose(resid_n, 0, atol=1e-12)
    a, b = run.slews["SAT-3"][0], nominal.slews["SAT-3"][0]
    assert a.peak_tracking_error_deg != b.peak_tracking_error_deg


def test_f4_torque_constant(faulted: dict[str, MissionRun], nominal: MissionRun) -> None:
    run = faulted["F4"]
    _, resid = wheel_residual(run, "SAT-3", 1, 0.5)
    np.testing.assert_allclose(resid, 0, atol=1e-12)  # wheel 2 delivers 0.5 u
    a, b = run.slews["SAT-3"][0], nominal.slews["SAT-3"][0]
    assert a.peak_tracking_error_deg > 10 * b.peak_tracking_error_deg


@pytest.mark.parametrize(("fid", "prop_ratio", "tol"), [("F5", 1.0, 1e-6), ("F6", 0.7, 1e-3)])
def test_f5_f6_burn(faulted: dict[str, MissionRun], nominal: MissionRun, fid: str,
                    prop_ratio: float, tol: float) -> None:
    run = faulted[fid]
    k = run.sats.index("SAT-4")
    n_arm = arm_index(run)
    burn_f, burn_n = run.burns["SAT-4"], nominal.burns["SAT-4"]
    assert burn_n.burn_start_s is not None
    assert burn_n.burn_end_s is not None
    assert burn_n.burn_start_s < run.t_s[n_arm] < burn_n.burn_end_s  # burn overlaps the arm
    dv_after_f = run.dv_m_s[-1, k] - run.dv_m_s[n_arm, k]
    dv_after_n = nominal.dv_m_s[-1, k] - nominal.dv_m_s[n_arm, k]
    assert dv_after_f == pytest.approx(0.7 * dv_after_n, rel=1e-9)
    prop_f = run.propellant_kg[n_arm, k] - run.propellant_kg[-1, k]
    prop_n = nominal.propellant_kg[n_arm, k] - nominal.propellant_kg[-1, k]
    assert prop_f == pytest.approx(prop_ratio * prop_n, rel=tol)
    assert burn_f.dv_m_s < burn_n.dv_m_s
    print(f"\n{fid}: burn dv {burn_n.dv_m_s:.3f} -> {burn_f.dv_m_s:.3f} m/s")


def test_f7_thermal(faulted: dict[str, MissionRun], nominal: MissionRun) -> None:
    run = faulted["F7"]
    f, n = run.thermal["SAT-2"], nominal.thermal["SAT-2"]
    last = run.t_s >= run.t_s[-1] - 4 * 3600
    drop = (n.T_K[last] - f.T_K[last]).mean(axis=0)
    ip = f.names.index("bus_px")
    assert drop[ip] == drop.max()
    assert drop[ip] > 0.5
    print(f"\nF7: bus_px cools {drop[ip]:.2f} K, bus_core {drop[f.names.index('bus_core')]:.2f} K")


def test_f8_crosslink(faulted: dict[str, MissionRun]) -> None:
    run = faulted["F8"]
    n_arm = arm_index(run)
    log = run.crosslink
    assert log is not None
    for j, (a, b) in enumerate(log.pairs):
        post = log.sent[n_arm:, j]
        rate = log.delivered[n_arm:, j][post].mean()
        if "SAT-2" in (a, b):
            assert abs(rate - 0.1) < 4 * np.sqrt(0.09 / post.sum())
        else:
            assert rate == 1.0
    age = log.age_s[("SAT-1", "SAT-2")]
    assert age[n_arm:].max() > 120
    assert age[:n_arm].max() == 0


def test_f9_upset_rate(params: Params, faulted: dict[str, MissionRun]) -> None:
    run = faulted["F9"]
    n_arm = arm_index(run)
    saa = run.saa["SAT-2"]
    inside = saa.in_saa[n_arm:]
    expect = (1 / 86400 + 50 / 1800) * 30.0 * inside.sum()
    got = saa.upsets[n_arm:][inside].sum()
    assert abs(got - expect) < 4 * np.sqrt(expect)
    print(f"\nF9: {got} upsets inside the SAA after arm (expected {expect:.1f})")


def test_f10_dark_current(faulted: dict[str, MissionRun], nominal: MissionRun) -> None:
    f, n = faulted["F10"].takes["SAT-2"], nominal.takes["SAT-2"]
    assert f.payload_K == n.payload_K
    assert f.dark_current_e_s_px == pytest.approx(4 * n.dark_current_e_s_px)


def test_faulted_72h_runtime(params: Params) -> None:
    run = run_mission(params, SEED, duration_s=72 * S_PER_HOUR,
                      fault=FaultConfig(mode="forced", fault_id="F1"))
    print(f"\nfaulted 72 h run: {run.wall_s:.1f} s")
    assert run.wall_s < 60
