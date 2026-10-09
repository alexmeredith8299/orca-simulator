"""Simulation timebase: run clock, named random substreams and per-seed scenario draws (M1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from expand3.params import Draw, Params, ParamView


def rng(seed: int, name: str) -> np.random.Generator:
    raise NotImplementedError


def sample(draw: Draw, gen: np.random.Generator) -> float:
    raise NotImplementedError


@dataclass(frozen=True)
class SeedDraws:
    seed: int
    epoch_utc: datetime
    run_length_s: float
    rho0_kg_m3: float
    H_km: float


def draw_seed(params: Params, seed: int) -> SeedDraws:
    raise NotImplementedError


@dataclass(frozen=True)
class Clock:
    step_s: float
    tick_s: float
    run_length_s: float

    @classmethod
    def from_view(cls, view: ParamView, run_length_s: float) -> Clock:
        raise NotImplementedError

    @property
    def steps_per_tick(self) -> int:
        raise NotImplementedError

    @property
    def n_ticks(self) -> int:
        raise NotImplementedError

    def tick_times_s(self) -> NDArray[np.float64]:
        raise NotImplementedError
