"""Fault injector F1 to F10: onset profiles, draws, ParamView schedules, labels (M7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from expand3.params import Fault, Params, ParamView

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class FaultConfig:
    mode: Literal["seeded", "none", "forced"] = "seeded"
    fault_id: str | None = None
    lead_s: float | None = None


@dataclass
class FaultLabel:
    fault_id: str
    name: str
    vehicle: str
    paths: list[str]
    magnitude: float
    kind: Literal["multiply", "set"]
    profile: str
    t0_s: float
    t_arm_s: float
    lead_s: float
    t_s: F64 = field(default_factory=lambda: np.zeros(0))
    s: F64 = field(default_factory=lambda: np.zeros(0))
    theta: dict[str, F64] = field(default_factory=dict)


def draw_fault(params: Params, seed: int) -> tuple[str, float]:
    raise NotImplementedError


def arm_time(t0_s: float, tick_s: float) -> float:
    raise NotImplementedError


def onset(fault: Fault, t_arm_s: float, t_s: float) -> float:
    raise NotImplementedError


def perturbed(fault: Fault, theta0: float, s: float) -> float:
    raise NotImplementedError


class Injector:
    def __init__(self, params: Params, fault_id: str, t0_s: float, lead_s: float,
                 tick_s: float) -> None:
        raise NotImplementedError

    def install(self, view: ParamView) -> None:
        raise NotImplementedError

    def label(self, t_s: F64) -> FaultLabel:
        raise NotImplementedError
