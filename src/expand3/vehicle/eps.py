"""Electrical power: array strings, loads, pack, terminal voltage, protections, heat (M4)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from expand3.params import ParamView

F64 = NDArray[np.float64]
B = NDArray[np.bool_]


@dataclass(frozen=True)
class EpsInputs:
    shadow: float
    sun_body: F64
    node_temps_K: Mapping[str, float]
    sensed_battery_K: float | None = None
    in_pass: bool = False
    take: bool = False
    wheel_power_W: float = 0.0
    edge_off: bool = False


@dataclass(frozen=True)
class EpsState:
    E_J: float
    uv_hold: bool = False
    inhibit: bool = False
    battery_heater: bool = False
    payload_heater: bool = False


@dataclass(frozen=True)
class EpsStep:
    p_string_W: F64
    p_gen_W: float
    loads_W: dict[str, float]
    demand_W: float
    p_acc_W: float
    p_sup_W: float
    p_pack_W: float
    curtailed_W: float
    unmet_W: float
    E_J: float
    E_next_J: float
    soc: float
    ocv_V: float
    v_V: float
    i_A: float
    disc_clamped: bool
    shed: bool
    uv_hold: bool
    inhibit: bool
    battery_heater: bool
    payload_heater: bool
    heat_W: dict[str, float]
    pack_loss_W: float
    array_conv_loss_W: float
    load_conv_loss_W: float


@dataclass
class EpsHistory:
    t_s: F64
    p_string_W: F64
    p_gen_W: F64
    loads_W: dict[str, F64]
    demand_W: F64
    p_pack_W: F64
    curtailed_W: F64
    unmet_W: F64
    E_J: F64
    soc: F64
    v_V: F64
    i_A: F64
    shed: B
    uv_hold: B
    inhibit: B
    battery_heater: B
    payload_heater: B
    disc_clamped: B
    heat_W: dict[str, F64] = field(default_factory=dict)


def placeholder_temperatures(view: ParamView) -> dict[str, float]:
    raise NotImplementedError


def history(t_s: F64, steps: list[EpsStep]) -> EpsHistory:
    raise NotImplementedError


class Eps:
    def __init__(self, view: ParamView) -> None:
        raise NotImplementedError

    def initial_state(self) -> EpsState:
        raise NotImplementedError

    def string_power_w(self, shadow: float, sun_body: F64, temps_k: Mapping[str, float]) -> F64:
        raise NotImplementedError

    def loads_w(self, state: EpsState, inputs: EpsInputs) -> dict[str, float]:
        raise NotImplementedError

    def usable_energy_j(self, battery_k: float) -> float:
        raise NotImplementedError

    def ocv_v(self, soc: float) -> float:
        raise NotImplementedError

    def step(self, state: EpsState, inputs: EpsInputs, dt: float) -> tuple[EpsStep, EpsState]:
        raise NotImplementedError
