"""Sensor models for the spec Signals table: GNSS, star tracker, IMU, EPS, thermal (M6)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from expand3.params import ParamView

F64 = NDArray[np.float64]
B = NDArray[np.bool_]


@dataclass
class Sampled:
    """A sampled signal: sample times, measurement and truth (rows aligned)."""

    t_s: F64
    meas: F64
    truth: F64


@dataclass
class StarTrackerSignal:
    t_s: F64
    available: B
    q_meas: F64
    q_true: F64


@dataclass
class VehicleSignals:
    gnss_r: Sampled
    gnss_v: Sampled
    star_tracker: StarTrackerSignal
    imu: Sampled
    pack_voltage: Sampled
    pack_current: Sampled
    string_power: Sampled
    node_temperature: Sampled
    node_names: list[str]
    soc_estimate: Sampled
    tank_gauge: Sampled
    burn_a_cmd_m_s2: F64
    burn_dv_m_s: F64
    burn_propellant_used_kg: F64
    heater_commanded: dict[str, B]
    heater_delivered: dict[str, B]
    gates: dict[str, B] = field(default_factory=dict)


def every(t_s: F64, rate_s: float) -> B:
    raise NotImplementedError


def gnss(rng: np.random.Generator, view: ParamView, t_s: F64, r_km: F64, v_km_s: F64
         ) -> tuple[Sampled, Sampled]:
    raise NotImplementedError


def tracker_basis(view: ParamView) -> F64:
    raise NotImplementedError


def star_tracker_available(view: ParamView, q: F64, r_eci_km: F64, sun_eci: F64) -> B:
    raise NotImplementedError


def star_tracker(rng: np.random.Generator, view: ParamView, t_s: F64, q: F64, r_eci_km: F64,
                 sun_eci: F64) -> StarTrackerSignal:
    raise NotImplementedError


def imu(rng: np.random.Generator, view: ParamView, t_s: F64, omega: F64) -> Sampled:
    raise NotImplementedError


def pack_voltage(rng: np.random.Generator, view: ParamView, t_s: F64, v: F64) -> Sampled:
    raise NotImplementedError


def pack_current(rng: np.random.Generator, view: ParamView, t_s: F64, i: F64) -> Sampled:
    raise NotImplementedError


def string_power(rng: np.random.Generator, view: ParamView, t_s: F64, p: F64) -> Sampled:
    raise NotImplementedError


def thermistor_noise(rng: np.random.Generator, view: ParamView, n_ticks: int, n_nodes: int) -> F64:
    raise NotImplementedError


def soc_estimate(rng: np.random.Generator, view: ParamView, t_s: F64, v_meas: F64, i_meas: F64,
                 soc_true0: float) -> F64:
    raise NotImplementedError


def tank_gauge(rng: np.random.Generator, view: ParamView, t_s: F64, propellant_kg: F64,
               burning: B) -> Sampled:
    raise NotImplementedError
