"""Run entry point: seeded single runs and batch runs (M8).

One directory per seed (`seed_NNNNN`) holds the data files from `recorder` and a
`manifest.json`; the batch directory holds the batch `manifest.json`. A seed is complete when
its manifest exists, so a batch resumes by skipping complete seeds unless forced. Each seed is
written to a temporary directory first and renamed into place, so an interrupted seed never
looks complete. Manifests carry the provenance and all timing; the data files carry neither, so
they are byte-identical across runs and worker counts (CLAUDE.md Known gaps).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import multiprocessing
import os
import pickle
import platform
import shutil
import subprocess
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from expand3.faults import FaultConfig, draw_fault, draw_fault_extended
from expand3.mission.thread import Truth, compute_truth, reuses_baseline, run_mission
from expand3.params import load_params
from expand3.recorder import write_run
from expand3.timebase import draw_seed
from expand3.units import S_PER_HOUR

MANIFEST = "manifest.json"
_DEPENDENCIES = ("numpy", "scipy", "pyarrow", "pydantic", "PyYAML", "astropy", "pyerfa",
                 "ppigrf", "pandas")


def seed_dir(out: Path, seed: int) -> Path:
    return Path(out) / f"seed_{seed:05d}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def params_sha256(params_dir: Path) -> str:
    """Hash of every file under params/ (relative names and contents, sorted)."""
    h = hashlib.sha256()
    for path in sorted(Path(params_dir).rglob("*")):
        if path.is_file():
            h.update(str(path.relative_to(params_dir)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def git_info() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], check=True,
                                capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status)}


def versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("expand3", *_DEPENDENCIES):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _dump(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def code_sha256() -> str:
    """Hash of the simulator sources (everything but the detectors), so a stored truth is never
    read by code that would compute something else."""
    root = Path(__file__).parent
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "detect" not in path.relative_to(root).parts:
            h.update(path.relative_to(root).as_posix().encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def load_or_compute_truth(params_dir: Path, seed: int, duration_h: float | None,
                          cache: Path) -> Truth:
    """The fault-free truth of a seed, stored in `cache` and keyed by the params, the simulator
    code and the run length. Sensor-only faults and the no-fault run of a seed share it."""
    params = load_params(params_dir)
    key = hashlib.sha256(
        f"{params_sha256(params_dir)}|{code_sha256()}|{duration_h}".encode()).hexdigest()[:16]
    path = Path(cache) / f"truth_{seed:05d}_{key}.pkl"
    if path.exists():
        with path.open("rb") as fh:
            truth: Truth = pickle.load(fh)
        return truth
    duration_s = None if duration_h is None else duration_h * S_PER_HOUR
    truth = compute_truth(params, draw_seed(params, seed), duration_s, None, None)
    Path(cache).mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(truth, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.rename(path)
    return truth


def run_seed(params_dir: Path, seed: int, out: Path, fault: FaultConfig,
             duration_h: float | None = None, force: bool = False,
             truth_cache: Path | None = None) -> dict[str, Any]:
    """Run one seed end to end and write its directory. Returns its batch-manifest entry."""
    final = seed_dir(out, seed)
    if (final / MANIFEST).exists() and not force:
        return {"seed": seed, "status": "skipped"}
    started, t_start = _now(), time.perf_counter()
    params = load_params(params_dir)
    draws = draw_seed(params, seed)
    duration_s = None if duration_h is None else duration_h * S_PER_HOUR
    baseline = None
    if truth_cache is not None and reuses_baseline(params, seed, fault):
        baseline = load_or_compute_truth(params_dir, seed, duration_h, truth_cache)
    run = run_mission(params, seed, duration_s=duration_s, fault=fault, baseline=baseline)
    tmp = Path(out) / f".tmp_seed_{seed:05d}_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    files = write_run(run, params, tmp)
    fault_id, lead = draw_fault(params, seed)
    drawn_ext, _, drawn_shape = draw_fault_extended(params, seed)
    manifest = {
        "seed": seed,
        "fault_config": dataclasses.asdict(fault),
        "duration_h": duration_h,
        "draws": {
            "epoch_utc": draws.epoch_utc.isoformat(),
            "run_length_s": draws.run_length_s,
            "run_length_used_s": float(run.t_s[-1]),
            "rho0_kg_m3": draws.rho0_kg_m3,
            "H_km": draws.H_km,
            "burn_command_s": dict(sorted(draws.burn_command_s.items())),
            "burn_length_s": draws.burn_length_s,
            "fault_id": fault_id,
            "fault_lead_s": lead,
            "fault_id_extended": drawn_ext,
            "fault_shape_extended": drawn_shape,
        },
        "fault_applied": None if run.fault is None else {
            "fault_id": run.fault.fault_id, "vehicle": run.fault.vehicle,
            "profile": run.fault.profile, "lead_s": run.fault.lead_s,
            "t0_s": run.fault.t0_s, "t_arm_s": run.fault.t_arm_s,
        },
        "git": git_info(),
        "params_sha256": params_sha256(params_dir),
        "versions": versions(),
        "files": {p.name: _sha256(p) for p in files},
        "started_utc": started,
        "finished_utc": _now(),
        "wall_s": time.perf_counter() - t_start,
    }
    _dump(tmp / MANIFEST, manifest)
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    return {"seed": seed, "status": "ok", "wall_s": manifest["wall_s"]}


def _guarded(runner: Callable[..., dict[str, Any]], args: tuple[Any, ...]) -> dict[str, Any]:
    try:
        return runner(*args)
    except Exception as exc:  # a failing seed is recorded; the batch goes on
        return {"seed": args[1], "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}


def run_batch(params_dir: Path, seeds: list[int], out: Path, fault: FaultConfig,
              duration_h: float | None = None, workers: int = 1, force: bool = False,
              truth_cache: Path | None = None,
              runner: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run seeds (in parallel processes when workers > 1) and write the batch manifest."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runner = runner or run_seed
    started, t_start = _now(), time.perf_counter()
    extra = () if truth_cache is None else (Path(truth_cache),)
    jobs = [(Path(params_dir), s, out, fault, duration_h, force, *extra) for s in seeds]
    results: list[dict[str, Any]] = []
    if workers <= 1:
        results = [_guarded(runner, job) for job in jobs]
    else:
        # spawn, not fork: numpy/astropy threads make forking a multi-threaded parent unsafe
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = [pool.submit(_guarded, runner, job) for job in jobs]
            results = [f.result() for f in as_completed(futures)]
    results.sort(key=lambda r: int(r["seed"]))
    batch = {
        "out": str(out),
        "seeds": results,
        "counts": {k: sum(r["status"] == k for r in results) for k in ("ok", "skipped", "failed")},
        "fault_config": dataclasses.asdict(fault),
        "duration_h": duration_h,
        "git": git_info(),
        "params_sha256": params_sha256(params_dir),
        "versions": versions(),
        "workers": workers,
        "started_utc": started,
        "finished_utc": _now(),
        "wall_s": time.perf_counter() - t_start,
    }
    _dump(out / MANIFEST, batch)
    return batch
