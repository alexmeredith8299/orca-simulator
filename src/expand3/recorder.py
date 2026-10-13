"""Run recorder: truth and telemetry channels written to Parquet (M8)."""

from __future__ import annotations

from pathlib import Path

from expand3.mission.thread import MissionRun
from expand3.params import Params

DATA_FILES = (
    "truth.parquet", "measurements.parquet", "wheel_steps.parquet", "burns.parquet",
    "burn_summary.parquet", "takes.parquet", "crosslink.parquet", "labels.json",
)


def write_run(run: MissionRun, params: Params, out_dir: Path) -> list[Path]:
    raise NotImplementedError
