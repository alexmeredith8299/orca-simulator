"""Learned residuals, peer differencing and the unsupervised baselines on synthetic batches."""

from pathlib import Path

import numpy as np
import pytest
from detect_synth import ARM_S, FAULT_VEHICLE, make_batch

from expand3.detect.base import Alarm
from expand3.detect.config import DetectConfig
from expand3.detect.data import SeedData
from expand3.detect.features import context, ewma, since_change
from expand3.detect.glr import slope_glr
from expand3.detect.harness import run_detectors
from expand3.detect.models import Ridge, poly_terms
from expand3.detect.peer import match_lag

PARAMS_DIR = Path(__file__).resolve().parents[2] / "params"
TRAIN = ["none"] * 5 + ["F1", "F7", "F2", "F9", "none"]
TEST = ["F1", "F7", "F2", "F9", "none"]
CFG = DetectConfig(h_floor=12.0, learn_window=60)


@pytest.fixture(scope="module")
def batch(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[int], list[int]]:
    out = tmp_path_factory.mktemp("synth2")
    seeds = make_batch(out, [*TRAIN, *TEST])
    return out, seeds[: len(TRAIN)], seeds[len(TRAIN):]


def run(batch: tuple[Path, list[int], list[int]], names: list[str]) -> dict[str, list[Alarm]]:
    out, tune, test = batch
    alarms, _ = run_detectors(out, names, tune, test, CFG, PARAMS_DIR)
    return {n: [a for a in alarms if a.detector == n] for n in names}


def post_arm(alarms: list[Alarm], seed: int, fault: str, fault_id: str) -> list[Alarm]:
    return [a for a in alarms if a.seed == seed and a.sat_id == FAULT_VEHICLE[fault]
            and a.t_s >= ARM_S and a.fault_id == fault_id]


def test_ridge_recovers_a_polynomial_and_ignores_scale() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 2)) * [1.0, 100.0]
    y = 2 + x[:, 0] ** 2 - 3 * x[:, 0] * x[:, 1] / 100
    model = Ridge(1e-3, degree=2).fit(x, y)
    assert np.abs(model.predict(x) - y).max() < 1e-2
    assert poly_terms(x, 3).shape == (500, 9)


def test_ewma_and_event_timers() -> None:
    step = np.concatenate([np.zeros(50), np.ones(500)])
    y = ewma(step, tau_s=300.0, dt_s=30.0)
    assert y[-1] == pytest.approx(1.0, abs=1e-3)
    assert y[60] == pytest.approx(1 - np.exp(-10 * 30.0 / 300.0), abs=0.05)
    sun = np.array([1, 1, 0, 0, 0, 1], dtype=float)
    assert since_change(sun, 30.0, 1e9, 0.0).tolist() == [0, 0, 0, 30, 60, 0]
    assert since_change(sun, 30.0, 45.0, 0.0).tolist()[-2:] == [45.0, 0.0]


def test_slope_glr_sees_a_resistance_not_a_mean_shift() -> None:
    rng = np.random.default_rng(1)
    current = np.tile([4.0, -4.0], 200)
    resid = 0.3 * current + 0.1 * rng.standard_normal(400)  # mean zero, slope 0.3
    assert slope_glr(resid, current, 0.1, 100).max() > 1000
    assert slope_glr(0.1 * rng.standard_normal(400), current, 0.1, 100).max() < 20


def test_match_lag_finds_the_phase_offset() -> None:
    a = (np.sin(2 * np.pi * np.arange(1000) / 190.0) > 0).astype(float)
    assert match_lag(a[7:], a[:-7], 300) == -7  # a[i] resembles b[i + 7]
    assert match_lag(a[:-5], a[5:], 300) == 5
    assert match_lag(a, a, 300) == 0


def test_measurement_reader_refuses_truth(batch: tuple[Path, list[int], list[int]]) -> None:
    d = SeedData(batch[0] / "seed_00000")
    assert not [c for c in d.measurements("SAT-2") if c.endswith("_true")]
    with pytest.raises(ValueError, match="truth"):
        d.table("truth")
    with pytest.raises(ValueError, match="truth"):
        d.table("measurements", ["T_bus_px_K_true"])


def test_context_uses_no_node_temperature(batch: tuple[Path, list[int], list[int]]) -> None:
    d = SeedData(batch[0] / "seed_00000")
    _, x, names = context(d, "SAT-2", CFG)
    assert x.shape[1] == len(names)
    assert not [n for n in names if n.startswith("T_")]
    assert np.isfinite(x).all()


@pytest.mark.parametrize("kind", ["ridge", "hgb"])
def test_learned_thermal_and_eps(batch: tuple[Path, list[int], list[int]], kind: str) -> None:
    if kind == "hgb":
        pytest.importorskip("sklearn")
    out = run(batch, [f"lrn_thermal_{kind}", f"lrn_eps_{kind}"])
    _, _, test = batch
    th, eps = out[f"lrn_thermal_{kind}"], out[f"lrn_eps_{kind}"]
    assert post_arm(th, test[1], "F7", "F7")
    assert post_arm(eps, test[0], "F1", "F1")
    assert post_arm(eps, test[2], "F2", "F2")
    quiet = [a for a in th + eps if a.seed == test[4]]
    assert not quiet


def test_peer_detectors(batch: tuple[Path, list[int], list[int]]) -> None:
    out = run(batch, ["peer_thermal", "peer_eps", "peer_saa"])
    _, _, test = batch
    assert post_arm(out["peer_thermal"], test[1], "F7", "F7")
    assert post_arm(out["peer_eps"], test[0], "F1", "F1")
    assert post_arm(out["peer_eps"], test[2], "F2", "F2")
    assert post_arm(out["peer_saa"], test[3], "F9", "F9")
    assert not [a for al in out.values() for a in al if a.seed == test[4]]
    # only the targets are scored, never the reference
    assert {a.sat_id for al in out.values() for a in al} <= {"SAT-2", "SAT-3"}


def test_unsupervised_detects_without_naming_a_fault(
        batch: tuple[Path, list[int], list[int]]) -> None:
    pytest.importorskip("sklearn")
    out = run(batch, ["unsup_pca", "unsup_iforest"])
    _, _, test = batch
    for name in ("unsup_pca", "unsup_iforest"):
        hits = [a for a in out[name] if a.seed == test[1] and a.sat_id == "SAT-2"
                and a.t_s >= ARM_S]
        assert hits, name
        assert hits[0].fault_id.endswith(":thermal") or ":" in hits[0].fault_id
        false = [a for a in out[name] if a.seed == test[4]]
        assert len(false) <= 3
