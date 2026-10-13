"""M6: sensor noise models, star tracker geometry, SoC estimator, tank gauge.
Spec: Model: sensors; Signals table."""

import numpy as np
import pytest

from expand3.params import Params, ParamView
from expand3.sensors import (
    every,
    gnss,
    imu,
    pack_current,
    pack_voltage,
    soc_estimate,
    star_tracker,
    star_tracker_available,
    string_power,
    tank_gauge,
    thermistor_noise,
    tracker_basis,
)
from expand3.timebase import rng
from expand3.vehicle.attitude import quat_conj, quat_mul

N = 200_000
ARCSEC = np.deg2rad(1 / 3600)


def check_stats(err: np.ndarray, mean: float, sigma: float) -> None:
    """Sample mean and std within 4-sigma confidence bounds."""
    n = err.size
    assert abs(err.mean() - mean) < 4 * sigma / np.sqrt(n)
    assert abs(err.std() - sigma) < 4 * sigma / np.sqrt(2 * n)


def test_every() -> None:
    t = np.arange(0, 300.0, 30.0)
    np.testing.assert_array_equal(every(t, 60), [True, False] * 5)
    assert every(t, 30).all()


def test_gnss_noise(params: Params) -> None:
    view = ParamView(params, "SAT-2")
    t = np.arange(N) * 30.0
    r = np.tile([7000.0, 0, 0], (N, 1))
    v = np.tile([0.0, 7.5, 0], (N, 1))
    pos, vel = gnss(rng(1, "noise.gnss.SAT-2"), view, t, r, v)
    np.testing.assert_array_equal(pos.t_s, t[::2])
    check_stats((pos.meas - pos.truth).ravel(), 0.0, 1.5e-3)  # km
    check_stats((vel.meas - vel.truth).ravel(), 0.0, 5e-5)  # km/s


def test_imu_power_thermistor_noise(params: Params) -> None:
    view = ParamView(params, "SAT-1")
    t = np.arange(N) * 30.0
    w = imu(rng(1, "noise.imu.SAT-1"), view, t, np.zeros((N, 3)))
    check_stats((w.meas - w.truth).ravel(), 0.0, 5e-5)
    v = pack_voltage(rng(1, "noise.voltage.SAT-1"), view, t, np.full(N, 31.0))
    check_stats(v.meas - v.truth, 0.0, 0.04)
    i = pack_current(rng(1, "noise.current.SAT-1"), view, t, np.full(N, 3.0))
    check_stats(i.meas - i.truth, 0.05, 0.02)  # bias recovered
    p_true = np.abs(np.random.default_rng(0).normal(100, 40, (N, 4))) + 1
    p = string_power(rng(1, "noise.string_power.SAT-1"), view, t, p_true)
    check_stats((p.meas / p.truth - 1).ravel(), 0.0, 0.01)  # multiplicative
    zero = string_power(rng(1, "noise.string_power.SAT-1"), view, t, np.zeros((N, 4)))
    assert np.all(zero.meas == 0)
    noise = thermistor_noise(rng(1, "noise.thermistor.SAT-1"), view, N, 15)
    check_stats(noise.ravel(), 0.0, 0.5)


def test_tank_gauge(params: Params) -> None:
    view = ParamView(params, "SAT-4")
    t = np.arange(N) * 30.0
    prop = np.linspace(8.0, 7.9, N)
    burning = np.zeros(N, dtype=bool)
    burning[1000:150_000] = True
    g = tank_gauge(rng(1, "noise.tank.SAT-4"), view, t, prop, burning)
    np.testing.assert_array_equal(g.t_s, t[burning])
    check_stats(g.meas - g.truth, 0.0, 0.1)
    np.testing.assert_array_equal(g.truth, prop[burning])


def rotvec(dq: np.ndarray) -> np.ndarray:
    dq = dq if dq[0] >= 0 else -dq
    return 2 * dq[1:]


@pytest.mark.parametrize(("sat", "cross", "roll"), [("SAT-1", 2.0, 20.0), ("SAT-5", 3.0, 25.0)])
def test_star_tracker_errors(params: Params, sat: str, cross: float, roll: float) -> None:
    view = ParamView(params, sat)
    basis = tracker_basis(view)
    np.testing.assert_allclose(basis[:, 2], [0, -np.cos(np.deg2rad(8)), -np.sin(np.deg2rad(8))],
                               atol=1e-12)
    n = 40_000
    t = np.arange(n) * 30.0
    q = np.tile([1.0, 0, 0, 0], (n, 1))
    # Sun along +z body (idle), vehicle far from Earth along +x: always available
    sun = np.tile([0.0, 0, 1], (n, 1))
    r = np.tile([0.0, 0, -1e6], (n, 1))
    st = star_tracker(rng(1, f"noise.star_tracker.{sat}"), view, t, q, r, sun)
    assert st.available.all()
    err_body = np.array([rotvec(quat_mul(quat_conj(a), b)) for a, b in zip(st.q_true, st.q_meas,
                                                                              strict=True)])
    err_tr = err_body @ basis  # components along tracker x, y, z (boresight)
    check_stats(err_tr[:, 0], 0, cross * ARCSEC)
    check_stats(err_tr[:, 1], 0, cross * ARCSEC)
    check_stats(err_tr[:, 2], 0, roll * ARCSEC)


def test_star_tracker_exclusion(params: Params) -> None:
    view = ParamView(params, "SAT-1")
    b = tracker_basis(view)[:, 2]  # boresight in body; identity attitude, so also in ECI
    q = np.array([[1.0, 0, 0, 0]])
    perp = np.cross(b, [1.0, 0, 0])
    perp /= np.linalg.norm(perp)

    def tilt(deg: float) -> np.ndarray:
        a = np.deg2rad(deg)
        return np.cos(a) * b + np.sin(a) * perp

    def available(sun: np.ndarray, r: np.ndarray) -> bool:
        return bool(star_tracker_available(view, q, r[None], sun[None])[0])

    # Sun cone (30 deg); Earth far away directly behind the tracker
    behind = b * 1e7
    assert not available(tilt(29.9), behind)
    assert available(tilt(30.1), behind)
    # Earth cone (25 deg from the limb) at 6903 km, Sun directly behind the tracker
    r_mag = 6903.137
    rho = np.rad2deg(np.arcsin(6378.137 / r_mag))
    for margin, ok in ((24.9, False), (25.1, True)):
        to_earth = tilt(rho + margin)
        assert available(-b, -to_earth * r_mag) is ok
    assert not available(-b, -b * r_mag)  # boresight on the Earth's center


def test_soc_estimator_drift_and_rereference(params: Params) -> None:
    view = ParamView(params, "SAT-1")
    e_cap = view.get("eps.E_cap_J")
    n = 2000
    t = np.arange(n) * 30.0
    v_true = np.full(n, 30.4)
    i_true = np.full(n, 2.0)  # charging, not at rest, mid SoC: never re-referenced
    v_meas = v_true + np.random.default_rng(3).normal(0, 0.04, n)
    i_meas = i_true + 0.05 + np.random.default_rng(4).normal(0, 0.02, n)
    est = soc_estimate(rng(1, "noise.soc.SAT-1"), view, t, v_meas, i_meas, 0.5)
    count_true = 0.5 + np.cumsum(np.r_[0, (v_true * i_true)[:-1]]) * 30.0 / e_cap
    err = est - count_true
    drift = err - err[0]
    expected = np.cumsum(np.r_[0, np.full(n - 1, 0.05 * 30.4)]) * 30.0 / e_cap
    assert abs(drift[-1] - expected[-1]) < 0.1 * expected[-1]
    assert abs(err[0]) < 4 * 0.02
    # re-reference: steep region and at rest -> pulled to OCV^-1(V_meas)
    v_hi = np.full(10, 33.2)  # OCV^-1 = 0.95
    i_rest = np.full(10, 0.1)
    est_hi = soc_estimate(rng(1, "noise.soc.SAT-1"), view, t[:10], v_hi, i_rest, 0.93)
    assert est_hi[-1] == pytest.approx(0.95, abs=1e-3)
    # steep but not at rest: no re-reference
    est_busy = soc_estimate(rng(1, "noise.soc.SAT-1"), view, t[:10], v_hi, np.full(10, 3.0), 0.93)
    assert abs(est_busy[1] - est_busy[0] - 33.2 * 3.0 * 30 / e_cap) < 1e-9
    # at rest but mid-curve: no re-reference
    est_mid = soc_estimate(rng(1, "noise.soc.SAT-1"), view, t[:10], np.full(10, 30.4), i_rest,
                           0.5)
    assert abs(est_mid[1] - est_mid[0] - 30.4 * 0.1 * 30 / e_cap) < 1e-9


def test_dummy_stream_changes_nothing(params: Params) -> None:
    view = ParamView(params, "SAT-2")
    t = np.arange(100) * 30.0
    r = np.tile([7000.0, 0, 0], (100, 1))
    a, _ = gnss(rng(9, "noise.gnss.SAT-2"), view, t, r, r)
    rng(9, "noise.dummy.SAT-2").normal(size=1000)
    b, _ = gnss(rng(9, "noise.gnss.SAT-2"), view, t, r, r)
    np.testing.assert_array_equal(a.meas, b.meas)
