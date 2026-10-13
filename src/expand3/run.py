"""Run entry point: seeded single runs and batch runs (M8)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from expand3.faults import FaultConfig


def seed_dir(out: Path, seed: int) -> Path:
    raise NotImplementedError


def run_seed(params_dir: Path, seed: int, out: Path, fault: FaultConfig,
             duration_h: float | None = None, force: bool = False) -> dict[str, Any]:
    raise NotImplementedError


def run_batch(params_dir: Path, seeds: list[int], out: Path, fault: FaultConfig,
              duration_h: float | None = None, workers: int = 1, force: bool = False,
              runner: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    raise NotImplementedError
