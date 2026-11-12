"""Read-only view of one seed directory written by `expand3 run`."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray

F64 = NDArray[np.float64]
BoolArr = NDArray[np.bool_]
S_PER_DAY = 86400.0


def seed_dirs(batch: Path) -> dict[int, Path]:
    """Completed seeds of a batch (those with a manifest), by seed number."""
    return {int(d.name.split("_")[1]): d for d in sorted(batch.glob("seed_*"))
            if (d / "manifest.json").is_file()}


class SeedData:
    """Tables and labels of one seed. Tables are read on demand, only the columns asked for."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.seed = int(directory.name.split("_")[1])

    @cached_property
    def labels(self) -> dict[str, Any]:
        out: dict[str, Any] = json.loads((self.directory / "labels.json").read_text())
        return out

    @property
    def fault_id(self) -> str:
        return str(self.labels["fault"])

    @property
    def vehicle(self) -> str | None:
        v = self.labels.get("vehicle")
        return None if self.fault_id == "none" or v is None else str(v)

    @property
    def t_arm_s(self) -> float | None:
        t = self.labels.get("t_arm_s")
        return None if self.vehicle is None or t is None else float(t)

    def table(self, name: str, columns: list[str] | None = None) -> Any:
        """A table of the seed. Truth is not an input to any detector, so it cannot be read."""
        if name == "truth" or any(c.endswith("_true") for c in columns or []):
            raise ValueError("detectors read measurements, not truth")
        return pq.read_table(self.directory / f"{name}.parquet", columns=columns)

    def measurements(self, sat: str) -> dict[str, F64]:
        """Every measured column of one vehicle (`_meas`, gates, counts), in time order, with no
        `_true` column. Read once per seed and cached."""
        cache: dict[str, dict[str, F64]] = self.__dict__.setdefault("_meas_cache", {})
        if sat not in cache:
            table = pq.read_table(self.directory / "measurements.parquet")
            names = [c for c in table.column_names if c != "sat_id" and not c.endswith("_true")]
            sel = np.asarray(table.column("sat_id")) == sat
            cache[sat] = {c: np.asarray(table.column(c))[sel].astype(np.float64) for c in names}
        return cache[sat]

    def sat_columns(self, name: str, sat: str, columns: list[str]) -> dict[str, Any]:
        """Columns of one vehicle's rows, in time order. `sat_id` is always read."""
        table = self.table(name, ["sat_id", *columns])
        sel = np.asarray(table.column("sat_id")) == sat
        return {c: np.asarray(table.column(c))[sel] for c in columns}

    @cached_property
    def sat_ids(self) -> list[str]:
        ids = np.unique(np.asarray(self.table("measurements", ["sat_id"]).column("sat_id")))
        return [str(s) for s in ids]

    @cached_property
    def t_end_s(self) -> float:
        return float(np.max(np.asarray(self.table("measurements", ["t_s"]).column("t_s"))))

    def fault_free(self, sat: str, t_s: F64) -> BoolArr:
        """True where the data cannot carry the fault: any other vehicle, or before arm."""
        t = np.asarray(t_s, dtype=np.float64)
        if sat != self.vehicle or self.t_arm_s is None:
            return np.ones(t.shape, dtype=bool)
        return t < self.t_arm_s

    def fault_free_days(self, sat: str) -> float:
        """Vehicle-days of fault-free data for one vehicle."""
        if sat != self.vehicle or self.t_arm_s is None:
            return self.t_end_s / S_PER_DAY
        return min(self.t_arm_s, self.t_end_s) / S_PER_DAY

    def t_take_s(self, sat: str) -> float | None:
        """First take of a vehicle, or None."""
        table = self.table("takes", ["sat_id", "t_take_s"])
        sel = np.asarray(table.column("sat_id")) == sat
        t = np.asarray(table.column("t_take_s"))[sel]
        return float(t.min()) if t.size else None
