"""M6: intra-cluster crosslink. Spec: Model: sensors, crosslink (F8)."""

import numpy as np
import pytest

from expand3.links import crosslink, line_of_sight
from expand3.params import Params, ParamView

SATS = ["SAT-1", "SAT-2", "SAT-3", "SAT-4", "SAT-5", "SAT-6"]


def test_line_of_sight() -> None:
    a = np.array([[7000.0, 0, 0]])
    assert line_of_sight(a, np.array([[7000.0, 500, 0]]), 6378.137)[0]
    assert not line_of_sight(a, np.array([[-7000.0, 0, 0]]), 6378.137)[0]
    assert not line_of_sight(a, np.array([[0.0, 7000, 0]]), 6378.137)[0]  # chord dips to 4950 km


def geometry(n: int, sep_km: float) -> tuple[np.ndarray, np.ndarray]:
    """Clusters on opposite sides of the Earth; members sep_km apart along track."""
    r = np.zeros((n, 6, 3))
    v = np.zeros((n, 6, 3))
    for k in range(3):
        r[:, k] = [6903.0, k * sep_km, 0]
        r[:, 3 + k] = [-6903.0, k * sep_km, 0]
    v[..., 2] = 7.6
    return r, v


def views(params: Params, p_sat2: float = 0.0) -> dict[str, ParamView]:
    out = {s: ParamView(params, s) for s in SATS}
    if p_sat2:
        out["SAT-2"].overrides["crosslink.drop_probability"] = p_sat2
    return out


def test_clusters_range_and_noise(params: Params) -> None:
    n = 4000
    t = np.arange(n) * 30.0
    r, v = geometry(n, 700.0)  # SAT-1..SAT-3 spans 1400 km; SAT-1 to SAT-3 within range
    log = crosslink(5, params, views(params), SATS, t, r, v)
    pairs = set(log.pairs)
    assert ("SAT-1", "SAT-4") not in pairs
    assert all((a[-1] in "123") == (b[-1] in "123") for a, b in log.pairs)
    assert log.sent.all()  # all within 1400 km and in sight
    assert log.delivered.all()  # p = 0 delivers all
    j = log.pairs.index(("SAT-1", "SAT-2"))
    err_r = log.r_meas_km[:, j] - r[:, 0]
    err_v = log.v_meas_km_s[:, j] - v[:, 0]
    assert abs(err_r.std() - 0.020) < 4 * 0.020 / np.sqrt(2 * err_r.size)
    assert abs(err_v.std() - 2e-5) < 4 * 2e-5 / np.sqrt(2 * err_v.size)
    far = crosslink(5, params, views(params), SATS, t, *geometry(n, 800.0))
    j13 = far.pairs.index(("SAT-1", "SAT-3"))
    assert not far.sent[:, j13].any()  # 1600 km: out of range
    assert not far.delivered[:, j13].any()


def test_f8_drop_probability(params: Params) -> None:
    n = 20_000
    t = np.arange(n) * 30.0
    r, v = geometry(n, 300.0)
    log = crosslink(5, params, views(params, p_sat2=0.9), SATS, t, r, v)
    for j, (a, b) in enumerate(log.pairs):
        rate = log.delivered[:, j].mean()
        if "SAT-2" in (a, b):
            assert abs(rate - 0.1) < 4 * np.sqrt(0.09 / n)
        else:
            assert rate == 1.0


def test_partner_age(params: Params) -> None:
    n = 2000
    t = np.arange(n) * 30.0
    r, v = geometry(n, 300.0)
    log = crosslink(5, params, views(params, p_sat2=0.5), SATS, t, r, v)
    j = log.pairs.index(("SAT-2", "SAT-1"))
    age = log.age_s[("SAT-1", "SAT-2")]
    got = log.delivered[:, j]
    assert np.all(age[got] == 0)
    for n_ in np.flatnonzero(~got[1:]) + 1:
        if np.isfinite(age[n_ - 1]):
            assert age[n_] == pytest.approx(age[n_ - 1] + 30.0)
    assert log.receipts["SAT-1"].max() <= 2
