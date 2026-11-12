"""Fault isolation with a gradient-boosted classifier on summary features of the residuals.

At decision times every `clf_step_s` the features of a vehicle are, for each residual statistic
(detector and fault id), its latest GLR score and its largest over the last `clf_lookback_s`,
on a log scale. Labels come from forced-fault runs (`expand3 run --fault Fk`) and the nominal tune
data: the faulted vehicle is labelled with its fault once the fault's window has opened (see
score.evidence_time), nominal before arm and on every other vehicle, and the stretch between arm
and the window opening is left out. Training uses tune seeds only. Scores for tune seeds are
out-of-fold by seed, so the false-alarm threshold is tuned on predictions the model did not
train on; test seeds use the model fitted on all training data.

The alarm score is 1 - P(nominal), and the alarm names the most likely fault.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from expand3.detect.base import Alarm, Trace
from expand3.detect.config import DetectConfig
from expand3.detect.data import SeedData, seed_dirs
from expand3.detect.glr import episodes, tune_threshold
from expand3.detect.harness import TraceSet, calibrate_nominal, compute_traces
from expand3.detect.score import evidence_time

F64 = NDArray[np.float64]
CLASSES = ("nominal", *[f"F{i}" for i in range(1, 11)])
SKIP = -1


def trace_features(traces: dict[str, list[Trace]], sat: str, grid: F64, cfg: DetectConfig,
                   keys: list[str]) -> F64:
    """(len(grid), 2 * len(keys)): held and trailing-max log1p scores; NaN where no trace."""
    out = np.full((len(grid), 2 * len(keys)), np.nan)
    for name, trs in traces.items():
        for tr in trs:
            if tr.sat_id != sat or len(tr.t_s) == 0:
                continue
            k = keys.index(f"{name}:{tr.fault_id}")
            hi = np.searchsorted(tr.t_s, grid, side="right")
            lo = np.searchsorted(tr.t_s, grid - cfg.clf_lookback_s, side="right")
            score = np.log1p(np.maximum(tr.score, 0.0))
            held = np.where(hi > 0, score[np.maximum(hi - 1, 0)], np.nan)
            recent = np.array([score[a:b].max() if b > a else np.nan for a, b in zip(lo, hi,
                                                                                    strict=True)])
            out[:, 2 * k] = np.fmax(out[:, 2 * k], held)
            out[:, 2 * k + 1] = np.fmax(out[:, 2 * k + 1], recent)
    return out


def labels(d: SeedData, sat: str, grid: F64) -> NDArray[np.int_]:
    """Class index per decision time (SKIP between arm and the window opening)."""
    out = np.zeros(len(grid), dtype=int)
    if sat != d.vehicle or d.t_arm_s is None:
        return out
    evidence = evidence_time(d, sat, d.fault_id, d.t_arm_s)
    after = grid >= d.t_arm_s
    out[after] = SKIP
    if evidence is not None:
        out[grid >= evidence] = CLASSES.index(d.fault_id)
    return out


def feature_keys(names: list[str], registry: dict[str, Any]) -> list[str]:
    return [f"{n}:{f}" for n in names for f in registry[n].fault_ids]


class Rows:
    """Decision-time rows of a set of runs."""

    def __init__(self) -> None:
        self.x: list[F64] = []
        self.y: list[NDArray[np.int_]] = []
        self.seed: list[NDArray[np.int_]] = []
        self.sat: list[NDArray[np.str_]] = []
        self.t: list[F64] = []
        self.tag: list[NDArray[np.str_]] = []

    def add(self, d: SeedData, traces: dict[str, list[Trace]], keys: list[str],
            cfg: DetectConfig, tag: str) -> None:
        grid = np.arange(cfg.clf_step_s, d.t_end_s, cfg.clf_step_s)
        for sat in d.sat_ids:
            self.x.append(trace_features(traces, sat, grid, cfg, keys))
            self.y.append(labels(d, sat, grid))
            n = len(grid)
            self.seed.append(np.full(n, d.seed))
            self.sat.append(np.full(n, sat, dtype="U8"))
            self.t.append(grid)
            self.tag.append(np.full(n, tag, dtype="U32"))

    def arrays(self) -> tuple[F64, NDArray[np.int_], NDArray[np.int_], NDArray[np.str_], F64,
                              NDArray[np.str_]]:
        return (np.concatenate(self.x), np.concatenate(self.y), np.concatenate(self.seed),
                np.concatenate(self.sat), np.concatenate(self.t), np.concatenate(self.tag))


def _classifier(cfg: DetectConfig) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=cfg.clf_max_iter, learning_rate=cfg.hgb_learning_rate,
        max_leaf_nodes=cfg.clf_max_leaf_nodes, class_weight="balanced",
        random_state=cfg.random_seed)


def _proba(model: Any, x: F64) -> F64:
    """Class probabilities in CLASSES order (zero for classes the model never saw)."""
    out = np.zeros((len(x), len(CLASSES)))
    seen = [int(c) for c in model.classes_]
    if len(seen) == 1:  # one class (a fold with no fault): sklearn still returns 2 columns
        out[:, seen[0]] = 1.0
    else:
        out[:, seen] = model.predict_proba(x)
    return out


def folds(seeds: NDArray[np.int_], k: int) -> list[NDArray[np.bool_]]:
    """k held-out masks that keep every row of a seed together."""
    unique = np.unique(seeds)
    return [np.isin(seeds, unique[i::k]) for i in range(k)]


def run_classifier(batch: Path, forced_dirs: list[Path], input_names: list[str],
                   tune: list[int], seeds: list[int], cfg: DetectConfig, params_dir: Path,
                   registry: dict[str, Any], workers: int = 1
                   ) -> tuple[list[Alarm], dict[str, Any]]:
    """Train on the tune seeds (main batch and the forced-fault batches) and alarm on `seeds`."""
    nominal = calibrate_nominal(batch, input_names, tune, cfg, params_dir, workers)
    keys = feature_keys(input_names, registry)
    main_traces: TraceSet = compute_traces(batch, nominal, sorted({*tune, *seeds}), cfg,
                                           params_dir, workers)
    rows = Rows()
    for s, (traces, _) in main_traces.items():
        rows.add(SeedData(seed_dirs(batch)[s]), traces, keys, cfg, "main")
    for fdir in forced_dirs:
        fseeds = [s for s in seed_dirs(fdir) if s in set(tune)]
        for s, (traces, _) in compute_traces(fdir, nominal, fseeds, cfg, params_dir,
                                             workers).items():
            rows.add(SeedData(seed_dirs(fdir)[s]), traces, keys, cfg, f"forced:{fdir.name}")
    x, y, seed, sat, t, tag = rows.arrays()
    group = np.full(len(x), "all", dtype="U8")
    alarms, info, models = fit_and_alarm("clf", (x, y, seed, sat, t, tag, group), tune, seeds,
                                         cfg)
    info.update(keys=keys)
    return alarms, {"info": info, "model": models["all"]}


RowArrays = tuple[F64, NDArray[np.int_], NDArray[np.int_], NDArray[np.str_], F64,
                  NDArray[np.str_], NDArray[np.str_]]


def fit_and_alarm(name: str, rows: RowArrays, tune: list[int], seeds: list[int],
                  cfg: DetectConfig) -> tuple[list[Alarm], dict[str, Any], dict[str, Any]]:
    """Fit one classifier per group (a bus, or all rows) on the tune rows and alarm on `seeds`.

    Rows are (x, y, seed, sat, t, tag, group). Tune seeds are scored out of fold by seed. The
    threshold of each group is tuned on its out-of-fold scores at the fault-free decision times
    of the main tune seeds; test seeds use the model fitted on all the group's training rows.
    """
    x, y, seed, sat, t, tag, group = rows
    prob = np.zeros((len(x), len(CLASSES)))
    models: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    per_class: dict[str, dict[str, int]] = {}
    dropped: dict[str, list[int]] = {}
    alarms: list[Alarm] = []
    for g in np.unique(group):
        here = group == g
        train = here & np.isin(seed, tune) & (y != SKIP)
        # a column with no value in the group (a thermistor the bus lacks) cannot be binned
        keep = ~np.all(np.isnan(x[train]), axis=0)
        xg = x[:, keep]
        dropped[str(g)] = [int(i) for i in np.flatnonzero(~keep)]
        for held in folds(seed[train], cfg.clf_folds):
            idx = np.flatnonzero(train)
            fit_rows, held_rows = idx[~held], idx[held]
            model = _classifier(cfg).fit(xg[fit_rows], y[fit_rows])
            every = np.flatnonzero(here & np.isin(seed, np.unique(seed[held_rows])))
            prob[every] = _proba(model, xg[every])
        final = _classifier(cfg).fit(xg[train], y[train])
        test_rows = here & np.isin(seed, seeds) & ~np.isin(seed, tune)
        prob[test_rows] = _proba(final, xg[test_rows])
        models[str(g)] = final
        per_class[str(g)] = {c: int(np.sum(y[train] == i)) for i, c in enumerate(CLASSES)}
        not_nominal = 1.0 - prob[:, 0]
        main = here & (tag == "main")
        null, days = [], 0.0
        for s in tune:
            for v in np.unique(sat[main & (seed == s)]):
                sel = main & (seed == s) & (sat == v)
                null.append(not_nominal[sel & (y == 0)])
                days += float(np.sum(sel & (y == 0))) * cfg.clf_step_s / 86400.0
        h = tune_threshold(null, days, cfg.target_far_per_vehicle_day, cfg.clf_floor)
        thresholds[str(g)] = h
        for s in seeds:
            for v in np.unique(sat[main & (seed == s)]):
                sel = np.flatnonzero(main & (seed == s) & (sat == v))
                for i in episodes(not_nominal[sel], h):
                    r = sel[i]
                    guess = CLASSES[1 + int(np.argmax(prob[r, 1:]))]
                    alarms.append(Alarm(s, name, str(v), float(t[r]), guess,
                                        float(not_nominal[r])))
    alarms.sort(key=lambda a: (a.seed, a.t_s, a.sat_id))
    info = {"threshold": thresholds, "classes": list(CLASSES),
            "train_rows": {g: sum(c.values()) for g, c in per_class.items()},
            "rows_per_class": per_class, "dropped_columns": dropped}
    return alarms, info, models


def save_model(out: Path, result: dict[str, Any], name: str = "clf") -> None:
    stem = "classifier" if name == "clf" else "classifier_raw"
    with (out / f"{stem}.pkl").open("wb") as f:
        pickle.dump(result, f)
