"""Magnetometer and the onboard field model (Phase C, docs/extended-faults-plan.md). Not in the
spec: a three-axis sensor with Gaussian noise, and a model of the field (IGRF at the vehicle's
navigation position, rotated by its star-tracker attitude) that is the only reference the
vehicle has for what it should read."""

from datetime import UTC, datetime

import numpy as np

from expand3.env.geomag import igrf_field_ecef_nt
from expand3.params import Params, ParamView
from expand3.sensors import magnetometer
from expand3.timebase import rng
from expand3.vehicle.magref import onboard_field

N = 100_000
EPOCH = datetime(2027, 3, 1, tzinfo=UTC)
BIAS, MULT, COUPLING = ("sensor_faults.mag.bias_nT", "sensor_faults.mag.noise_mult",
                        "sensor_faults.mag.coupling_nT_per_A")


def _read(params: Params, **overrides: float):
    view = ParamView(params, "SAT-1")
    view.overrides.update(overrides)
    t = np.arange(N) * 30.0
    b = np.tile([20000.0, -5000.0, 30000.0], (N, 1))
    return magnetometer(rng(1, "noise.mag.SAT-1"), view, t, b, np.full(N, 2.0))


def test_noise_is_gaussian_at_the_set_sigma_and_unbiased(params: Params) -> None:
    s = _read(params)
    sigma = params.links.signals["magnetometer"].model_extra["sigma_nT"]  # type: ignore[index]
    err = (s.meas - s.truth).ravel()
    assert abs(err.mean()) < 4 * sigma / np.sqrt(err.size)
    assert abs(err.std() / sigma - 1.0) < 0.02
    np.testing.assert_array_equal(s.t_s, (np.arange(N) * 30.0)[::2])  # the 60 s grid


def test_bias_noise_multiplier_and_coupling_act_as_named(params: Params) -> None:
    base = _read(params)
    sigma = float((base.meas - base.truth).std())
    biased = _read(params, **{BIAS: 300.0})
    np.testing.assert_allclose((biased.meas - biased.truth).mean(axis=0), 300.0,
                               atol=4 * sigma / np.sqrt(biased.t_s.size))
    noisy = _read(params, **{MULT: 5.0})
    assert abs((noisy.meas - noisy.truth).std() / sigma - 5.0) < 0.1
    coupled = _read(params, **{COUPLING: 100.0})  # 100 nT per A at 2 A
    np.testing.assert_allclose((coupled.meas - coupled.truth).mean(axis=0), 200.0,
                               atol=4 * sigma / np.sqrt(coupled.t_s.size))


def test_onboard_field_is_igrf_in_the_body_frame_and_blind_without_a_tracker_fix() -> None:
    r_eci = np.array([[6903.0, 0.0, 0.0], [0.0, 6903.0, 0.0]])
    rot = np.tile(np.eye(3), (2, 1, 1))
    q = np.array([[1.0, 0.0, 0.0, 0.0], [np.nan, np.nan, np.nan, np.nan]])  # no fix on tick 1
    body, mag = onboard_field(r_eci, rot, EPOCH, q)
    want = igrf_field_ecef_nt(r_eci, EPOCH)
    np.testing.assert_allclose(body[0], want[0], rtol=1e-9)  # identity attitude
    assert np.isnan(body[1]).all()
    np.testing.assert_allclose(mag, np.linalg.norm(want, axis=1), rtol=1e-9)
