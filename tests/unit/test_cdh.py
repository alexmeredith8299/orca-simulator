"""M6: SAA and C&DH, imager dark current. Spec: Model: sensors, SAA and C&DH (F9), imager (F10)."""

from datetime import UTC, datetime

import astropy.units as u
import numpy as np
import ppigrf
import pytest
from astropy.coordinates import EarthLocation

from expand3.params import Params, ParamView
from expand3.timebase import rng
from expand3.vehicle.cdh import edge_computer_off, field_magnitude_nt, upsets
from expand3.vehicle.payload import dark_current_e_s_px

EPOCH = datetime(2027, 8, 17, tzinfo=UTC)


def ecef(lat: float, lon: float, h_km: float) -> np.ndarray:
    loc = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, h_km * u.km)
    return np.array([[loc.x.to_value(u.km), loc.y.to_value(u.km), loc.z.to_value(u.km)]])


@pytest.mark.parametrize(("lat", "lon", "inside"), [(-26.0, -50.0, True), (60.0, 100.0, False)])
def test_field_matches_ppigrf(lat: float, lon: float, inside: bool) -> None:
    b = field_magnitude_nt(ecef(lat, lon, 525.0), EPOCH)[0]
    be, bn, bu = ppigrf.igrf(lon, lat, 525.0, EPOCH.replace(tzinfo=None))
    ref = float(np.sqrt(be**2 + bn**2 + bu**2).ravel()[0])
    assert b == pytest.approx(ref, rel=1e-6)
    assert bool(b < 22_000) is inside


def test_edge_computer_timing() -> None:
    t = np.arange(20) * 30.0
    in_saa = np.zeros(20, dtype=bool)
    in_saa[5:9] = True  # inside for ticks 5..8, leaves at tick 9 (t = 270 s)
    off = edge_computer_off(t, in_saa, 60.0)
    expected = np.zeros(20, dtype=bool)
    expected[5:11] = True  # off inside, and through 270 s and 300 s; back on at 330 s
    np.testing.assert_array_equal(off, expected)


def test_upset_rates(params: Params) -> None:
    view = ParamView(params, "SAT-2")
    n, dt = 400_000, 30.0
    in_saa = np.zeros(n, dtype=bool)
    in_saa[: n // 4] = True
    counts, resets = upsets(rng(1, "upsets.SAT-2"), view, in_saa, dt)
    lam_out, lam_in = 1 / 86400, 1 / 1800
    for mask, lam in ((~in_saa, lam_out), (in_saa, lam_out + lam_in)):
        expect = lam * dt * mask.sum()
        assert abs(counts[mask].sum() - expect) < 4 * np.sqrt(expect)
    total = counts.sum()
    assert abs(resets.sum() - 0.05 * total) < 4 * np.sqrt(total * 0.05 * 0.95)
    assert np.all(resets <= counts)
    view.overrides["cdh_upsets.lambda_in_per_s"] = 50 * lam_in
    counts50, _ = upsets(rng(2, "upsets.SAT-2"), view, in_saa[: n // 40], dt)
    expect = (lam_out + 50 * lam_in) * dt * (n // 40)
    assert abs(counts50.sum() - expect) < 4 * np.sqrt(expect)


def test_dark_current(params: Params) -> None:
    view = ParamView(params, "SAT-2")
    assert dark_current_e_s_px(view, 293.15) == pytest.approx(10.0)
    assert dark_current_e_s_px(view, 293.15 + 6.5) == pytest.approx(20.0)
    assert dark_current_e_s_px(view, 293.15 - 13.0) == pytest.approx(2.5)
    view.overrides["imager_dark_current.D0_e_s_px"] = 40.0
    assert dark_current_e_s_px(view, 299.65) == pytest.approx(4 * 20.0)
