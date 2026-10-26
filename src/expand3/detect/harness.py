"""Run detectors over a batch: calibrate on the tune seeds, score every seed, emit alarms.

Per detector: `summarize` each tune seed and `fit` the nominal values, compute traces for every
seed, tune one threshold per fault id from the fault-free parts of the tune traces, and turn
threshold crossings into alarms. Seed jobs run in a spawn process pool.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from expand3.detect.base import Alarm, Detector, Trace
from expand3.detect.config import DetectConfig
from expand3.detect.data import SeedData, seed_dirs
from expand3.detect.glr import episodes, tune_threshold
from expand3.detect.registry import CLASSIFIER, FAMILIES, REGISTRY

FAMILY_NAMES = set(FAMILIES)
ALARM_FIELDS = ["seed", "detector", "sat_id", "t_s", "fault_id", "score", "detail"]


def detector_names(text: str) -> list[str]:
    """`all`, one name or a comma list."""
    names: list[str] = []
    for token in (n.strip() for n in text.split(",")):
        if token == "all":
            names += list(REGISTRY)
        elif token == CLASSIFIER:
            names.append(token)
        elif token in FAMILY_NAMES and token != "classifier":
            names += [n for n, cls in REGISTRY.items() if cls.family == token]
        else:
            names.append(token)
    unknown = [n for n in names if n not in REGISTRY and n != CLASSIFIER]
    if unknown:
        raise ValueError(f"unknown detector {unknown}; choose from {sorted(REGISTRY)} or all")
    return names


def _make(name: str, cfg: DetectConfig, params_dir: str | None) -> Detector:
    return REGISTRY[name](cfg, params_dir)


def _summarize_job(args: tuple[Path, list[str], DetectConfig, str | None]) -> dict[str, Any]:
    directory, names, cfg, params_dir = args
    d = SeedData(directory)
    return {n: _make(n, cfg, params_dir).summarize(d) for n in names}


def _trace_job(args: tuple[Path, dict[str, dict[str, Any]], DetectConfig, str | None]
               ) -> tuple[int, dict[str, list[Trace]], dict[str, Any]]:
    directory, nominal, cfg, params_dir = args
    d = SeedData(directory)
    traces = {n: _make(n, cfg, params_dir).traces(d, nom) for n, nom in nominal.items()}
    info = {"fault_free_days": {s: d.fault_free_days(s) for s in d.sat_ids},
            "t_end_s": d.t_end_s,
            "fault_free": {n: [d.fault_free(tr.sat_id, tr.t_s) for tr in trs]
                           for n, trs in traces.items()}}
    return d.seed, traces, info


def _map[T](fn: Callable[[Any], T], jobs: list[Any], workers: int) -> list[T]:
    if workers <= 1:
        return [fn(j) for j in jobs]
    # one thread per worker: gradient boosting would otherwise start a full OpenMP pool in each
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        return list(pool.map(fn, jobs))


def alarms_from_trace(seed: int, detector: str, tr: Trace, h: float) -> list[Alarm]:
    return [Alarm(seed, detector, tr.sat_id, float(tr.t_s[i]), tr.fault_id, float(tr.score[i]),
                  "" if tr.detail is None else str(tr.detail[i]))
            for i in episodes(tr.score, h)]


TraceSet = dict[int, tuple[dict[str, list[Trace]], dict[str, Any]]]


def check_seeds(batch: Path, seeds: list[int]) -> dict[int, Path]:
    dirs = seed_dirs(batch)
    missing = [s for s in set(seeds) if s not in dirs]
    if missing:
        raise FileNotFoundError(f"seeds without a completed run in {batch}: {sorted(missing)}")
    return dirs


def calibrate_nominal(batch: Path, names: list[str], tune: list[int], cfg: DetectConfig,
                      params_dir: Path | None, workers: int = 1) -> dict[str, dict[str, Any]]:
    """Fit each detector's nominal values on the tune seeds."""
    dirs = check_seeds(batch, tune)
    pdir = None if params_dir is None else str(params_dir)
    sums = _map(_summarize_job, [(dirs[s], names, cfg, pdir) for s in tune], workers)
    return {n: _make(n, cfg, pdir).fit([s[n] for s in sums]) for n in names}


def compute_traces(batch: Path, nominal: dict[str, dict[str, Any]], seeds: list[int],
                   cfg: DetectConfig, params_dir: Path | None, workers: int = 1) -> TraceSet:
    """Score traces (and the fault-free masks and exposure) of each seed, for every detector in
    `nominal`."""
    dirs = check_seeds(batch, seeds)
    pdir = None if params_dir is None else str(params_dir)
    results = _map(_trace_job, [(dirs[s], nominal, cfg, pdir) for s in sorted(set(seeds))],
                   workers)
    return {seed: (traces, info) for seed, traces, info in results}


def tune_thresholds(names: list[str], by_seed: TraceSet, tune: list[int], cfg: DetectConfig
                    ) -> tuple[dict[str, dict[str, float]], float]:
    """One threshold per detector and fault id from the fault-free parts of the tune traces,
    and the fault-free vehicle-days they came from."""
    out: dict[str, dict[str, float]] = {}
    days = 0.0
    for n in names:
        null: dict[str, list[Any]] = {f: [] for f in REGISTRY[n].fault_ids}
        days = 0.0
        for s in tune:
            traces, info = by_seed[s]
            days += sum(info["fault_free_days"].values())
            for tr, ok in zip(traces[n], info["fault_free"][n], strict=True):
                null[tr.fault_id].append(tr.score[ok])
        own = REGISTRY[n].h_floor
        floor = cfg.h_floor if own is None else own
        out[n] = {f: tune_threshold(scores, days, cfg.target_far_per_vehicle_day, floor)
                  for f, scores in null.items()}
    return out, days


def run_detectors(batch: Path, names: list[str], tune: list[int], seeds: list[int],
                  cfg: DetectConfig, params_dir: Path | None, workers: int = 1,
                  keep: dict[str, Any] | None = None) -> tuple[list[Alarm], dict[str, Any]]:
    """Calibrate on `tune` and detect on `seeds`. Returns the alarms and the calibration. When
    `keep` is given it receives the nominal values (`nominal`) and the traces (`traces`)."""
    check_seeds(batch, [*tune, *seeds])
    nominal = calibrate_nominal(batch, names, tune, cfg, params_dir, workers)
    by_seed = compute_traces(batch, nominal, [*tune, *seeds], cfg, params_dir, workers)
    thresholds, days = tune_thresholds(names, by_seed, tune, cfg)
    calibration: dict[str, Any] = {"config": cfg.as_dict(), "tune_seeds": sorted(tune),
                                   "nominal": nominal, "thresholds": thresholds,
                                   "tune_vehicle_days": days}
    alarms = [a for n in names for s in seeds for tr in by_seed[s][0][n]
              for a in alarms_from_trace(s, n, tr, thresholds[n][tr.fault_id])]
    alarms.sort(key=lambda a: (a.seed, a.t_s, a.sat_id, a.fault_id))
    if keep is not None:
        keep.update(nominal=nominal, traces=by_seed, thresholds=thresholds)
    return alarms, calibration


def write_alarms(path: Path, alarms: list[Alarm]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ALARM_FIELDS)
        w.writeheader()
        w.writerows(asdict(a) for a in alarms)


def read_alarms(path: Path) -> list[Alarm]:
    with path.open(newline="") as f:
        return [Alarm(int(r["seed"]), r["detector"], r["sat_id"], float(r["t_s"]), r["fault_id"],
                      float(r["score"]), r["detail"]) for r in csv.DictReader(f)]


def detect_batch(batch: Path, names: list[str], tune: list[int], seeds: list[int],
                 out: Path, cfg: DetectConfig, params_dir: Path | None, workers: int = 1
                 ) -> list[Alarm]:
    alarms, calibration = run_detectors(batch, names, tune, seeds, cfg, params_dir, workers)
    out.mkdir(parents=True, exist_ok=True)
    write_alarms(out / "alarms.csv", alarms)
    with (out / "nominal.pkl").open("wb") as f:
        pickle.dump(calibration["nominal"], f)
    (out / "calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True, default=_encode) + "\n")
    return alarms


def _encode(obj: Any) -> Any:
    """JSON for calibration values that are not plain: arrays as lists, models by name."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return f"<{type(obj).__name__}>"


__all__ = ["detect_batch", "detector_names", "read_alarms", "run_detectors",
           "write_alarms"]
