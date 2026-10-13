"""Fault injector F1 to F10: onset profiles, draws, ParamView schedules, labels (M7).

Spec: Faults. Each fault perturbs parameter(s) theta from theta0 through an onset profile s(t):

    theta(t) = theta0 (1 + (m - 1) s(t))          (multiply; F2 with m = 0 gives o_1 = 1 - s)
    theta(t) = theta0 + (set_to - theta0) s(t)    (F3 and F8, nominal zero: set directly)
    s(t) = 0 before arm; step: 1 from arm; ramp: min(1, (t - t_arm) / t_r)

The fault arms on the first control tick at or after t0 = (take start) - lead and stays on to
the end of the run. A cap (F7: 1.0) bounds the perturbed value. Everything comes from
params/faults.yaml; the injector only installs time schedules on the faulted vehicle's
ParamView, so model code holds no fault logic.

Defaults where the spec is silent (CLAUDE.md Known gaps): the ramp runs from the arm tick, not
from t0. The take start that sets t0 comes from the fault-free plan of the same seed, so a fault
cannot move its own onset. The draws use the streams `draw.fault` and `draw.fault_lead`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from expand3.params import Fault, Params, ParamView
from expand3.timebase import rng, sample

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class FaultConfig:
    """Which fault a run carries: the seeded draw (default), none, or a forced fault (and
    optionally a forced lead) for paired comparisons and tests."""

    mode: Literal["seeded", "none", "forced"] = "seeded"
    fault_id: str | None = None
    lead_s: float | None = None


@dataclass
class FaultLabel:
    """Labels recorded with a faulted run, with s(t) and theta(t) at every tick."""

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
    """Spec: one fault per seed, uniform over the ten, and the fault lead drawn per seed."""
    ids = sorted(params.faults.faults, key=lambda f: int(f[1:]))
    fid = ids[int(rng(seed, "draw.fault").integers(len(ids)))]
    return fid, sample(params.scenario.fault_lead_s, rng(seed, "draw.fault_lead"))


def arm_time(t0_s: float, tick_s: float) -> float:
    """First control tick at or after t0."""
    return math.ceil(t0_s / tick_s - 1e-9) * tick_s


def onset(fault: Fault, t_arm_s: float, t_s: float) -> float:
    """Spec: onset profile s(t)."""
    if t_s < t_arm_s:
        return 0.0
    if fault.profile == "step":
        return 1.0
    assert fault.t_r_s is not None
    return min(1.0, (t_s - t_arm_s) / fault.t_r_s)


def perturbed(fault: Fault, theta0: float, s: float) -> float:
    """Spec: theta(t) for one parameter at onset level s."""
    if fault.set_to is not None:
        value = theta0 + (fault.set_to - theta0) * s
    else:
        assert fault.m is not None
        value = theta0 * (1 + (fault.m - 1) * s)
    return min(value, fault.cap) if fault.cap is not None else value


class Injector:
    """One fault armed at a given t0 on the table's vehicle."""

    def __init__(self, params: Params, fault_id: str, t0_s: float, lead_s: float,
                 tick_s: float) -> None:
        self.fault_id = fault_id
        self.fault = params.faults.faults[fault_id]
        self.t0_s = t0_s
        self.lead_s = lead_s
        self.t_arm_s = arm_time(t0_s, tick_s)
        self._theta0 = {p: ParamView(params, self.fault.vehicle).base(p) for p in self.fault.paths}

    def s(self, t_s: float) -> float:
        return onset(self.fault, self.t_arm_s, t_s)

    def theta(self, path: str, t_s: float) -> float:
        return perturbed(self.fault, self._theta0[path], self.s(t_s))

    def install(self, view: ParamView) -> None:
        """Schedule the faulted path(s) on the faulted vehicle's view."""
        if view.sat_id != self.fault.vehicle:
            raise ValueError(f"{self.fault_id} applies to {self.fault.vehicle}, not {view.sat_id}")
        for path in self.fault.paths:
            view.schedule[path] = self._schedule_for(path)

    def _schedule_for(self, path: str) -> Callable[[float], float]:
        def theta_at(t_s: float) -> float:
            return self.theta(path, t_s)

        return theta_at

    def label(self, t_s: F64) -> FaultLabel:
        f = self.fault
        return FaultLabel(
            fault_id=self.fault_id,
            name=f.name,
            vehicle=f.vehicle,
            paths=list(f.paths),
            magnitude=float(f.set_to if f.set_to is not None else f.m),  # type: ignore[arg-type]
            kind="set" if f.set_to is not None else "multiply",
            profile=f.profile,
            t0_s=self.t0_s,
            t_arm_s=self.t_arm_s,
            lead_s=self.lead_s,
            t_s=np.asarray(t_s, dtype=np.float64),
            s=np.array([self.s(float(t)) for t in t_s]),
            theta={p: np.array([self.theta(p, float(t)) for t in t_s]) for p in f.paths},
        )
