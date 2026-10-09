"""Station-keeping burns: command, uplink at the next pass, finite low-thrust burn (M3)."""

from __future__ import annotations

from dataclasses import dataclass, field

from expand3.params import Params, ParamView
from expand3.timebase import SeedDraws


@dataclass
class BurnRecord:
    sat: str
    command_time_s: float
    length_s: float
    accel_m_s2: float
    uplink_pass: tuple[float, float] | None = None
    burn_start_s: float | None = None
    burn_end_s: float | None = None
    suspended: list[tuple[float, float]] = field(default_factory=list)
    flown_s: float = 0.0
    dv_m_s: float = 0.0
    propellant_used_kg: float = 0.0


def accel_magnitude_m_s2(view: ParamView, length_s: float) -> float:
    raise NotImplementedError


def plan_burns(params: Params, draws: SeedDraws, step_s: float) -> dict[str, BurnRecord]:
    raise NotImplementedError


class BurnController:
    """Per-vehicle burn state machine, advanced at control ticks and queried every step."""

    def __init__(
        self, record: BurnRecord, tick_s: float, step_s: float,
        suspend: list[tuple[float, float]] | None = None,
    ) -> None:
        raise NotImplementedError

    def at_tick(self, t_s: float, visible: bool) -> None:
        raise NotImplementedError

    def active(self, t_s: float) -> bool:
        raise NotImplementedError

    def flew(self, dt: float) -> None:
        raise NotImplementedError
