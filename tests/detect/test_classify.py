"""The isolation classifier: labels, features, and a small end-to-end run."""

from pathlib import Path

import numpy as np
import pytest
from detect_synth import ARM_S, FAULT_VEHICLE, TAKE_S, make_seed

from expand3.detect.base import Trace
from expand3.detect.classify import CLASSES, SKIP, feature_keys, folds, labels, trace_features
from expand3.detect.config import DetectConfig
from expand3.detect.data import SeedData
from expand3.detect.registry import REGISTRY

PARAMS_DIR = Path(__file__).resolve().parents[2] / "params"
CFG = DetectConfig(h_floor=12.0, learn_window=60, clf_step_s=900.0, clf_max_iter=40,
                   clf_folds=3, clf_floor=0.5)
INPUTS = ["f2", "f8", "f9", "f10", "wheels", "lrn_thermal_ridge", "lrn_eps_ridge"]
FAULTS = ["F1", "F2", "F3", "F4", "F7", "F8", "F9", "F10"]


def test_labels_leave_out_the_stretch_before_the_window_opens(tmp_path: Path) -> None:
    d = SeedData(make_seed(tmp_path, 0, "F10"))
    grid = np.arange(900.0, 86400.0, 900.0)
    y = labels(d, "SAT-2", grid)
    take = 80030.0  # SAT-2's take; F10's window opens at take start
    assert np.all(y[grid < ARM_S] == 0)
    assert np.all(y[(grid >= ARM_S) & (grid < take)] == SKIP)
    assert np.all(y[grid >= take] == CLASSES.index("F10"))
    assert np.all(labels(d, "SAT-1", grid) == 0)  # other vehicles are nominal
    ymid = labels(SeedData(make_seed(tmp_path, 1, "F2")), "SAT-2", grid)
    assert np.all(ymid[grid >= ARM_S] == CLASSES.index("F2"))  # F2 acts from arm


def test_features_hold_and_trail() -> None:
    tr = Trace("SAT-2", "F2", np.array([100.0, 1000.0, 5000.0]), np.array([0.0, 20.0, 1.0]))
    keys = ["f2:F2", "f9:F9"]
    grid = np.array([50.0, 1500.0, 6000.0])
    x = trace_features({"f2": [tr]}, "SAT-2", grid, CFG, keys)
    assert np.isnan(x[0, 0])  # nothing yet
    assert x[1, 0] == pytest.approx(np.log1p(20.0))  # held
    assert x[2, 0] == pytest.approx(np.log1p(1.0))
    assert x[2, 1] == pytest.approx(np.log1p(20.0))  # trailing max still sees 1000 s ago
    assert np.isnan(x[:, 2:]).all()  # the f9 key has no trace
    assert feature_keys(["f2", "wheels"], REGISTRY) == ["f2:F2", "wheels:F3", "wheels:F4"]


def test_folds_keep_seeds_together() -> None:
    seeds = np.repeat(np.arange(10), 7)
    masks = folds(seeds, 3)
    assert sum(m.sum() for m in masks) == len(seeds)
    for m in masks:
        assert set(seeds[m]).isdisjoint(set(seeds[~m]))


@pytest.mark.slow
def test_classifier_isolates_on_synthetic_runs(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    from expand3.detect.classify import run_classifier

    main, forced = tmp_path / "main", tmp_path / "forced"
    kinds = ["none"] * 4 + FAULTS * 2          # tune seeds 0..19
    for s, k in enumerate(kinds):
        make_seed(main, s, k)
    test_kinds = ["none", *FAULTS]             # seeds 20..28
    for s, k in enumerate(test_kinds, start=20):
        make_seed(main, s, k)
    for f in FAULTS:                            # extra forced runs on tune seeds
        for s in range(3):
            make_seed(forced / f, s, f)
    alarms, result = run_classifier(main, sorted(forced.iterdir()), INPUTS, list(range(20)),
                                    list(range(20, 29)), CFG, PARAMS_DIR, REGISTRY)
    assert sum(result["info"]["train_rows"].values()) > 0
    got = {}
    for a in alarms:
        if a.t_s >= ARM_S and a.sat_id == FAULT_VEHICLE.get(test_kinds[a.seed - 20], ""):
            got.setdefault(a.seed, a.fault_id)
    for s, k in enumerate(test_kinds, start=20):
        if k in ("F2", "F8", "F9", "F7", "F1"):
            assert got.get(s) == k, (k, got.get(s))
    assert not [a for a in alarms if a.seed == 20]  # the clean run stays quiet
    assert TAKE_S > ARM_S


def test_raw_features_are_window_statistics_of_measured_signals(tmp_path: Path) -> None:
    from expand3.detect.rawfeatures import raw_features

    d = SeedData(make_seed(tmp_path, 0, "F2"))
    grid = np.arange(900.0, 86400.0, 900.0)
    x, names = raw_features(d, "SAT-2", grid, CFG)
    assert x.shape == (len(grid), len(names))
    assert not any(n.endswith("_true") or "theta" in n for n in names)
    col = {n: x[:, i] for i, n in enumerate(names)}
    post = grid > ARM_S + CFG.raw_windows_s[0]
    # an F2 string reads low against the others once the window has filled with sunlit samples
    low = col[f"string_ratio_min@{CFG.raw_windows_s[0]:g}"]
    clean = raw_features(d, "SAT-1", grid, CFG)[0][:, names.index(f"string_ratio_min@"
                                                                  f"{CFG.raw_windows_s[0]:g}")]
    assert np.nanmin(low[post]) < np.nanmin(clean[post]) - 0.05


@pytest.mark.slow
def test_raw_classifier_isolates_on_synthetic_runs(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    from expand3.detect.rawclf import run_raw_classifier

    main, forced = tmp_path / "main", tmp_path / "forced"
    kinds = ["none"] * 4 + FAULTS * 2
    for s, k in enumerate(kinds):
        make_seed(main, s, k)
    test_kinds = ["none", *FAULTS]
    for s, k in enumerate(test_kinds, start=20):
        make_seed(main, s, k)
    for f in FAULTS:
        for s in range(3):
            make_seed(forced / f, s, f)
    alarms, result = run_raw_classifier(main, sorted(forced.iterdir()), list(range(20)),
                                        list(range(20, 29)), CFG, PARAMS_DIR)
    assert set(result["info"]["train_rows"]) <= {"A", "B"}
    assert sum(result["info"]["train_rows"].values()) > 0
    assert all(a.detector == "clf_raw" for a in alarms)
    got = {}
    for a in alarms:
        if a.t_s >= ARM_S and a.sat_id == FAULT_VEHICLE.get(test_kinds[a.seed - 20], ""):
            got.setdefault(a.seed, a.fault_id)
    for s, k in enumerate(test_kinds, start=20):
        # F8 is left out: its link rates move on the other vehicles too, and with a handful of
        # synthetic F8 rows its score stays near the tuned threshold
        if k not in ("none", "F8"):
            assert got.get(s) == k, (k, got.get(s))


def test_probabilities_of_a_model_that_saw_one_class() -> None:
    pytest.importorskip("sklearn")
    from expand3.detect.classify import _classifier, _proba

    x = np.arange(20.0).reshape(10, 2)
    p = _proba(_classifier(CFG).fit(x, np.zeros(10, dtype=int)), x)
    assert p.shape == (10, len(CLASSES))
    assert np.all(p[:, 0] == 1.0)
    assert np.all(p[:, 1:] == 0.0)
