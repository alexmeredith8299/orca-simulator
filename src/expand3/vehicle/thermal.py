"""Thermal network: node equation, topology, explicit Euler per control tick (M5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from expand3.params import ParamView
from expand3.vehicle.eps import Eps, EpsInputs, EpsState, EpsStep

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class ThermalEnv:
    shadow: float
    sun_body: F64
    nadir_body: F64
    r_km: float
    sun_dot_nadir: float


def thermal_env(q: F64, r_eci_km: F64, sun_eci: F64, shadow: float) -> ThermalEnv:
    raise NotImplementedError


@dataclass
class Network:
    names: list[str]
    C: F64
    alpha: F64
    eps: F64
    area_src: F64
    area_rad: F64
    normal: F64
    view: F64
    G: F64
    S: float
    a_alb: float
    q_IR: float
    R_E_km: float
    sigma: float

    def absorbed_w(self, env: ThermalEnv) -> tuple[F64, F64, F64]:
        raise NotImplementedError

    def rate(self, T: F64, Q: F64, env: ThermalEnv) -> F64:
        raise NotImplementedError

    def step(self, T: F64, Q: F64, env: ThermalEnv, dt: float) -> F64:
        raise NotImplementedError

    def linearized(self, T_ref: float) -> F64:
        raise NotImplementedError

    def stability_limit_s(self, T_ref: float) -> float:
        raise NotImplementedError


class Thermal:
    names: list[str]

    def __init__(self, view: ParamView) -> None:
        raise NotImplementedError

    def network(self) -> Network:
        raise NotImplementedError

    def initial_temps(self) -> F64:
        raise NotImplementedError

    def heat_vector(self, heat_w: dict[str, float]) -> F64:
        raise NotImplementedError

    def temps_map(self, T: F64) -> dict[str, float]:
        raise NotImplementedError

    def step(self, T: F64, heat_w: dict[str, float], env: ThermalEnv, dt: float) -> F64:
        raise NotImplementedError


@dataclass
class ThermalHistory:
    t_s: F64
    names: list[str]
    T_K: F64

    def node(self, name: str) -> F64:
        out: F64 = self.T_K[:, self.names.index(name)]
        return out


def step_power_and_thermal(
    eps: Eps,
    thermal: Thermal,
    eps_state: EpsState,
    T: F64,
    inputs: EpsInputs,
    env: ThermalEnv,
    dt: float,
) -> tuple[EpsStep, EpsState, F64]:
    raise NotImplementedError


def run_offline(
    thermal: Thermal, envs: list[ThermalEnv], heats: list[dict[str, float]], dt: float,
    T0: F64 | None = None,
) -> F64:
    raise NotImplementedError
