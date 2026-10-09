"""Simulation timebase: run clock, named random substreams and per-seed scenario draws (M1).

Spec: Parameters > Scenario and environment (epoch, run length, integration step, control tick,
drag rho_0 and H). Every random number in a run comes from `rng(seed, name)`.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from expand3.params import Draw, Params, ParamView
from expand3.units import S_PER_HOUR

_KEY_WORDS = 4  # 128 bits of the name's SHA-256 go into the SeedSequence spawn key


def rng(seed: int, name: str) -> np.random.Generator:
    """Generator for the named substream of a run seed, e.g. `noise.gnss.SAT-2`, `draw.fault`.

    The stream depends only on (seed, name), so streams are independent of call order and adding
    a stream never shifts the draws of another.
    """
    digest = hashlib.sha256(name.encode()).digest()
    key = tuple(int.from_bytes(digest[4 * j : 4 * j + 4], "little") for j in range(_KEY_WORDS))
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=key)))


def sample(draw: Draw, gen: np.random.Generator) -> float:
    """One value from a `low`/`high` draw (uniform or log-uniform)."""
    if draw.low is None or draw.high is None:
        raise ValueError("sample() needs a draw with low and high")
    if draw.draw == "uniform":
        return float(gen.uniform(draw.low, draw.high))
    return float(math.exp(gen.uniform(math.log(draw.low), math.log(draw.high))))


@dataclass(frozen=True)
class SeedDraws:
    """The per-seed scenario draws used so far (M1). Later milestones add fields."""

    seed: int
    epoch_utc: datetime
    run_length_s: float
    rho0_kg_m3: float
    H_km: float


def _draw_epoch(draw: Draw, gen: np.random.Generator) -> datetime:
    if draw.year is None:
        raise ValueError("epoch draw needs a year")
    start = datetime(draw.year, 1, 1, tzinfo=UTC)
    span_s = (datetime(draw.year + 1, 1, 1, tzinfo=UTC) - start).total_seconds()
    return start + timedelta(seconds=float(gen.uniform(0.0, span_s)))


def draw_seed(params: Params, seed: int) -> SeedDraws:
    """Spec: Parameters > Scenario and environment. Epoch uniform over the year (UTC), run length
    uniform in hours, rho_0 log-uniform, H uniform; each from its own named substream."""
    s = params.scenario
    return SeedDraws(
        seed=seed,
        epoch_utc=_draw_epoch(s.epoch, rng(seed, "draw.epoch")),
        run_length_s=sample(s.run_length_h, rng(seed, "draw.run_length")) * S_PER_HOUR,
        rho0_kg_m3=sample(s.drag.rho0_kg_m3, rng(seed, "draw.rho0")),
        H_km=sample(s.drag.H_km, rng(seed, "draw.H")),
    )


@dataclass(frozen=True)
class Clock:
    """Fixed integration step inside a fixed control tick. Spec: Scenario and environment."""

    step_s: float
    tick_s: float
    run_length_s: float

    def __post_init__(self) -> None:
        ratio = self.tick_s / self.step_s
        if ratio < 1 or not math.isclose(ratio, round(ratio)):
            raise ValueError("control tick must be a whole number of integration steps")

    @classmethod
    def from_view(cls, view: ParamView, run_length_s: float) -> Clock:
        return cls(
            step_s=view.get("scenario.integration_step_s"),
            tick_s=view.get("scenario.control_tick_s"),
            run_length_s=run_length_s,
        )

    @property
    def steps_per_tick(self) -> int:
        return round(self.tick_s / self.step_s)

    @property
    def n_ticks(self) -> int:
        """Ticks at t = 0, tick, 2 tick, ... up to and including run_length_s."""
        return math.floor(self.run_length_s / self.tick_s + 1e-9) + 1

    def tick_times_s(self) -> NDArray[np.float64]:
        return np.arange(self.n_ticks, dtype=np.float64) * self.tick_s
