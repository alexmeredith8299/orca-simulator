"""Station-keeping burns: command, uplink at the next pass, finite low-thrust burn (M3).

Spec: Scenario > Mission thread (SAT-2, SAT-3 and SAT-4 each get one station-keeping command at
the burn command time; the burn flies after the command uplinks at the next ground pass);
Model: actuators (finite low-thrust burn, zero-order-hold acceleration over the burn length).

Defaults where the spec is silent (CLAUDE.md Known gaps):
- The uplink pass is the first pass (any station) with a visible tick at or after the command
  time, so a command issued during a pass uplinks in that pass. The burn starts at the first tick
  after that pass ends (loss of signal).
- Burn length is one per-seed draw (scenario `station_keeping_burn_length_draw_h`), rounded to
  whole integration steps. The ordered acceleration is prograde with a constant magnitude sized
  for `station_keeping_dv_m_s` at the nominal eta_F.
- A burn is suspended over its vehicle's take slews and take, and resumes afterwards, so the full
  burn length is always flown.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from expand3.params import Params, ParamView
from expand3.timebase import SeedDraws


@dataclass
class BurnRecord:
    """Burn log for one vehicle (times in s from the epoch; truth delta-v and propellant)."""

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
    """Ordered acceleration magnitude for the station-keeping delta-v over the burn length at the
    vehicle's nominal (loaded, not fault-overridden) eta_F."""
    eta_nominal = view.params.bus(view.sat_id).thruster.eta_F
    return view.get("scenario.station_keeping_dv_m_s") / (eta_nominal * length_s)


def plan_burns(
    params: Params, draws: SeedDraws, step_s: float, views: dict[str, ParamView] | None = None
) -> dict[str, BurnRecord]:
    """One burn record per burn vehicle. `views` carries any per-vehicle overrides."""
    length = round(draws.burn_length_s / step_s) * step_s
    views = views or {}
    return {
        sat: BurnRecord(
            sat=sat,
            command_time_s=draws.burn_command_s[sat],
            length_s=length,
            accel_m_s2=accel_magnitude_m_s2(views.get(sat, ParamView(params, sat)), length),
        )
        for sat in params.scenario.burn_vehicles
    }


class BurnController:
    """Per-vehicle burn state machine, advanced at control ticks and queried every step."""

    def __init__(
        self, record: BurnRecord, tick_s: float, step_s: float,
        suspend: list[tuple[float, float]] | None = None,
    ) -> None:
        self.rec = record
        self.tick_s = tick_s
        self.step_s = step_s
        self.suspend = list(suspend or [])
        self._state = "wait"
        self._prev_visible = False
        self._run_start = 0.0
        self._pass_last = 0.0
        self._last_active = 0.0

    def at_tick(self, t_s: float, visible: bool) -> None:
        """Advance the command / uplink logic with this tick's ground visibility (any station)."""
        if visible and not self._prev_visible:
            self._run_start = t_s
        self._prev_visible = visible
        if self._state == "wait" and t_s >= self.rec.command_time_s and visible:
            self._state = "in_pass"
        if self._state == "in_pass":
            if visible:
                self._pass_last = t_s
            else:
                self.rec.uplink_pass = (self._run_start, self._pass_last)
                self.rec.burn_start_s = t_s
                self._state = "burn"

    def active(self, t_s: float) -> bool:
        """Whether the thruster fires over the integration step starting at t_s."""
        if self._state != "burn" or self.rec.burn_start_s is None:
            return False
        if t_s < self.rec.burn_start_s or self.rec.flown_s >= self.rec.length_s:
            return False
        for window in self.suspend:
            if window[0] <= t_s < window[1]:
                if window not in self.rec.suspended:
                    self.rec.suspended.append(window)
                return False
        self._last_active = t_s
        return True

    def flew(self, dt: float) -> None:
        """Account one fired step; closes the burn when the full length has been flown."""
        self.rec.flown_s += dt
        if self.rec.flown_s >= self.rec.length_s:
            self.rec.burn_end_s = self._last_active + dt
            self._state = "done"
