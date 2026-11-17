"""True magnetic field = IGRF-14 plus a common disturbance; the vehicle's onboard model is IGRF
alone. Not in the spec (docs/extended-faults-plan.md, Phase C). The disturbance is a smooth random
field, the same for every vehicle at the same place and time, so cluster mates feel nearly the
same one and it cannot be told from a magnetometer bias by one vehicle alone."""

from datetime import UTC, datetime

import numpy as np

from expand3.env.geomag import FieldDisturbance, igrf_field_ecef_nt
from expand3.params import Params, ParamView
from expand3.vehicle.cdh import field_magnitude_nt

EPOCH = datetime(2027, 3, 1, tzinfo=UTC)
R_KM = 6903.0


def _ring(n: int, lon0: float = 0.0, lat0: float = 0.0) -> np.ndarray:
    lon = np.deg2rad(lon0 + np.linspace(0.0, 360.0, n, endpoint=False))
    lat = np.deg2rad(lat0)
    return R_KM * np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                            np.full(n, np.sin(lat))], axis=1)


def test_igrf_vector_has_the_magnitude_the_saa_uses() -> None:
    r = np.concatenate([_ring(40), _ring(40, lat0=60.0), _ring(40, lat0=-45.0)])
    b = igrf_field_ecef_nt(r, EPOCH)
    assert b.shape == (120, 3)
    np.testing.assert_allclose(np.linalg.norm(b, axis=1), field_magnitude_nt(r, EPOCH), rtol=1e-9)
    assert 20000.0 < np.linalg.norm(b, axis=1).min() < np.linalg.norm(b, axis=1).max() < 60000.0


def _field(params: Params, seed: int = 4) -> FieldDisturbance:
    return FieldDisturbance.draw(ParamView(params, "SAT-1"), seed)


def test_disturbance_has_the_set_rms_and_zero_mean(params: Params) -> None:
    f = _field(params)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20000, 3))
    x = R_KM * x / np.linalg.norm(x, axis=1, keepdims=True)
    t = rng.uniform(0.0, 72 * 3600.0, 20000)
    d = f.at(x, t)
    rms = ParamView(params, "SAT-1").get("field_disturbance.rms_nT")
    assert abs(np.sqrt((np.linalg.norm(d, axis=1) ** 2).mean()) / rms - 1.0) < 0.2
    assert np.abs(d.mean(axis=0)).max() < 0.1 * rms


def test_disturbance_is_the_same_field_for_everyone_and_depends_on_the_seed(
        params: Params) -> None:
    x, t = _ring(50), np.full(50, 5000.0)
    np.testing.assert_array_equal(_field(params).at(x, t), _field(params).at(x, t))
    assert not np.allclose(_field(params, 4).at(x, t), _field(params, 5).at(x, t))


def test_nearby_vehicles_feel_nearly_the_same_disturbance_and_distant_ones_do_not(
        params: Params) -> None:
    f = _field(params)
    rng = np.random.default_rng(1)
    base = rng.normal(size=(4000, 3))
    base = R_KM * base / np.linalg.norm(base, axis=1, keepdims=True)
    t = rng.uniform(0.0, 72 * 3600.0, 4000)
    ref = f.at(base, t)

    def corr(shift_km: float) -> float:
        other = base + np.array([0.0, 0.0, 0.0])
        other = other + shift_km * np.cross(base, [0.0, 0.0, 1.0]) / np.linalg.norm(
            np.cross(base, [0.0, 0.0, 1.0]), axis=1, keepdims=True)
        got = f.at(other, t)
        return float((ref * got).sum() / np.sqrt((ref**2).sum() * (got**2).sum()))

    assert corr(100.0) > 0.9
    assert corr(10000.0) < 0.5


def test_disturbance_drifts_in_time(params: Params) -> None:
    f = _field(params)
    x = _ring(200)
    a, b = f.at(x, np.zeros(200)), f.at(x, np.full(200, 12 * 3600.0))
    c = float((a * b).sum() / np.sqrt((a**2).sum() * (b**2).sum()))
    assert c < 0.5
