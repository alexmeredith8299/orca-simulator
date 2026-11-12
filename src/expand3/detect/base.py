"""Detector interface, score traces and alarms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from expand3.detect.config import DetectConfig
from expand3.detect.data import SeedData

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class Trace:
    """A detection statistic over time for one vehicle and one fault id.

    `detail` optionally names the component per sample (a string, a wheel); an alarm carries the
    detail at the sample where it fires.
    """

    sat_id: str
    fault_id: str
    t_s: F64
    score: F64
    detail: NDArray[np.str_] | None = None


@dataclass(frozen=True)
class Alarm:
    seed: int
    detector: str
    sat_id: str
    t_s: float
    fault_id: str
    score: float
    detail: str = ""


class Detector(ABC):
    """A detector reads a seed's measurement-side files and returns score traces.

    Calibration is in two steps over the tune seeds. `summarize` returns sufficient statistics of
    the seed's fault-free data and `fit` pools them into nominal values. `traces` then scores a
    seed against those nominal values. Thresholds are tuned afterwards from the traces.
    """

    name: str
    fault_ids: tuple[str, ...]
    family: str = "physics"
    # True when alarms do not name a fault (unsupervised); they are scored on detection only
    detection_only: bool = False
    # threshold floor for this detector's scores; None uses DetectConfig.h_floor
    h_floor: float | None = None

    def __init__(self, cfg: DetectConfig, params_dir: str | None = None) -> None:
        self.cfg = cfg
        self.params_dir = params_dir

    @abstractmethod
    def summarize(self, d: SeedData) -> dict[str, Any]: ...

    @abstractmethod
    def fit(self, summaries: list[dict[str, Any]]) -> dict[str, Any]: ...

    @abstractmethod
    def traces(self, d: SeedData, nominal: dict[str, Any]) -> list[Trace]: ...
