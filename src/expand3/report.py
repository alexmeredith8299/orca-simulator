"""Validation report from a batch directory: Markdown plus CSV tables (M8)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_report(batch_dir: Path, params_dir: Path | None = None) -> dict[str, Any]:
    raise NotImplementedError
