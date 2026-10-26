"""Fault isolation learned from the measured signals themselves.

`clf` classifies the scores of the other detectors. This classifier replaces them with the
signals the physics residuals read, summarised over trailing windows (`rawfeatures`), so every
nominal value and every sensitivity must be learned from the data. It is trained and alarmed the
same way as `clf` (decision times, labels, out-of-fold scoring, threshold; see `classify`), with
one model per bus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from expand3.detect.base import Alarm
from expand3.detect.classify import F64, RowArrays, fit_and_alarm, labels
from expand3.detect.config import DetectConfig
from expand3.detect.data import SeedData, seed_dirs
from expand3.detect.features import bus_of
from expand3.detect.harness import _map
from expand3.detect.rawfeatures import raw_features

NAME = "clf_raw"


def _rows_job(args: tuple[Path, str, DetectConfig, dict[str, str]]
              ) -> tuple[list[F64], list[Any], list[str]]:
    """Decision-time rows of every vehicle of one seed directory."""
    directory, tag, cfg, buses = args
    d = SeedData(directory)
    grid = np.arange(cfg.clf_step_s, d.t_end_s, cfg.clf_step_s)
    x: list[F64] = []
    rest: list[Any] = []
    names: list[str] = []
    for sat in d.sat_ids:
        f, names = raw_features(d, sat, grid, cfg)
        n = len(grid)
        x.append(f)
        rest.append((labels(d, sat, grid), np.full(n, d.seed), np.full(n, sat, dtype="U8"), grid,
                     np.full(n, tag, dtype="U32"), np.full(n, buses[sat], dtype="U8")))
    return x, rest, names


def run_raw_classifier(batch: Path, forced_dirs: list[Path], tune: list[int], seeds: list[int],
                       cfg: DetectConfig, params_dir: Path, workers: int = 1
                       ) -> tuple[list[Alarm], dict[str, Any]]:
    """Train on the tune seeds (main batch and forced-fault batches) and alarm on `seeds`."""
    buses = bus_of(str(params_dir))
    main = seed_dirs(batch)
    jobs = [(main[s], "main", cfg, buses) for s in sorted({*tune, *seeds})]
    for fdir in forced_dirs:
        found = seed_dirs(fdir)
        jobs += [(found[s], f"forced:{fdir.name}", cfg, buses) for s in sorted(found)
                 if s in set(tune)]
    results = _map(_rows_job, jobs, workers)
    names = results[0][2]
    x = np.concatenate([a for r in results for a in r[0]])
    cols = [np.concatenate([part[i] for r in results for part in r[1]]) for i in range(6)]
    y, seed, sat, t, tag, group = cols
    rows: RowArrays = (x, y, seed, sat, t, tag, group)
    alarms, info, models = fit_and_alarm(NAME, rows, tune, seeds, cfg)
    info["features"] = names
    return alarms, {"info": info, "model": models}
