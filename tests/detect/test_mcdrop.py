"""The Monte Carlo dropout forecasters: windows, pickling, uncertainty and synthetic faults."""

import pickle
from pathlib import Path

import numpy as np
import pytest
from detect_synth import ARM_S, FAULT_VEHICLE, make_batch

from expand3.detect.config import DetectConfig
from expand3.detect.harness import run_detectors

pytest.importorskip("torch")

from expand3.detect.mcdrop import McdModel, _windows

PARAMS_DIR = Path(__file__).resolve().parents[2] / "params"
CFG = DetectConfig(h_floor=12.0, mcd_epochs=8, mcd_hidden=16, mcd_lags=8, mcd_samples=10,
                   mcd_window=6, mcd_threads=2, mcd_batch=128, mcd_device="cpu")
TRAIN = ["none"] * 6 + ["F1", "F7", "none", "none"]
TEST = ["F1", "F7", "none"]


def test_windows_are_oldest_first_and_clamped_at_the_series_start() -> None:
    x = np.arange(20, dtype=np.float32)[:, None]
    start = np.array([0] * 10 + [10] * 10)  # two series of ten rows
    w = _windows(x, np.array([3, 11, 14]), start, 5)
    assert w[0, :, 0].tolist() == [0.0, 0.0, 1.0, 2.0, 3.0]
    assert w[1, :, 0].tolist() == [10.0, 10.0, 10.0, 10.0, 11.0]  # never reads the first series
    assert w[2, :, 0].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]


def test_model_learns_a_smooth_signal_and_reports_uncertainty() -> None:
    rng = np.random.default_rng(0)
    series, targets, seeds = [], [], []
    for s in range(10):
        t = np.arange(300.0)
        x = np.stack([np.sin(t / 20.0), np.cos(t / 33.0)], axis=1).astype(np.float32)
        series.append(x)
        targets.append(2.0 * x[:, :1] + 0.5 * x[:, 1:] + 0.05 * rng.normal(size=(300, 1)))
        seeds.append(s)
    model = McdModel(CFG, ["y"]).fit(series, targets, seeds)
    mean, std = model.predict(series[1], seed=1)
    assert np.abs(mean[:, 0] - targets[1][:, 0]).mean() < 0.3
    assert np.all(std > 0.0)
    assert 0.8 < model.coverage <= 1.0
    # pickling goes through the weights and the stochastic passes are reproducible
    clone: McdModel = pickle.loads(pickle.dumps(model))
    again, _ = clone.predict(series[1], seed=1)
    assert np.allclose(mean, again, atol=1e-5)
    other, _ = model.predict(series[1], seed=2)
    assert not np.allclose(mean, other)


@pytest.mark.slow
def test_f7_and_f1_scores_rise_after_arm(tmp_path: Path) -> None:
    seeds = make_batch(tmp_path, [*TRAIN, *TEST])
    tune, test = seeds[: len(TRAIN)], seeds[len(TRAIN):]
    keep: dict[str, object] = {}
    run_detectors(tmp_path, ["mcd_thermal", "mcd_eps"], tune, test, CFG, PARAMS_DIR,
                  keep=keep)
    by_seed = keep["traces"]
    for seed, kind, name in ((test[0], "F1", "mcd_eps"), (test[1], "F7", "mcd_thermal")):
        traces = by_seed[seed][0][name]  # type: ignore[index]
        tr = next(t for t in traces if t.sat_id == FAULT_VEHICLE[kind])
        before, after = tr.score[tr.t_s < ARM_S], tr.score[tr.t_s >= ARM_S]
        assert after.max() > 3.0 * max(before.max(), 1.0), (kind, before.max(), after.max())
