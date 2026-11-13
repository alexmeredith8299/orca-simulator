"""Sensor models for the spec Signals table: GNSS, star tracker, IMU, EPS, thermal (M6).

Spec: Model: sensors, crosslink, SAA, imager (Navigation); Signals table. Truth plus white
Gaussian noise unless stated; every signal keeps its truth beside it. Each sensor of each vehicle
draws from its own named stream (`noise.<sensor>.<sat>`), so adding a sensor never changes
another's draws.

Conventions (CLAUDE.md Known gaps):
- The star tracker boresight is the -y face direction tilted 8 deg toward -z ("below level").
  Its frame has z on the boresight and x along body +x. Errors are a small rotation in that
  frame, with cross-boresight sigma on x and y and roll sigma on z, applied on the body side.
  There is no fix with the Sun within the Sun cone of the boresight, or with the boresight within
  the Earth cone of the Earth's limb (or on the Earth).
- SoC estimate: E_est advances by V_meas I_meas dt (charging positive), starting from the true
  SoC plus N(0, 2 %). It is re-referenced to OCV^-1(V_meas) when the estimate is in a steep part
  of the curve (SoC < 0.1 or > 0.9) and the pack is near rest (|I_meas| < 0.5 A).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from expand3.params import ParamView
from expand3.units import M_PER_KM

F64 = NDArray[np.float64]
B = NDArray[np.bool_]

_ARCSEC_RAD = np.deg2rad(1.0 / 3600.0)


@dataclass
class Sampled:
    """A sampled signal: sample times, measurement and truth (rows aligned)."""

    t_s: F64
    meas: F64
    truth: F64


@dataclass
class StarTrackerSignal:
    """Attitude fixes at the tracker rate; q_meas is NaN where no fix is available."""

    t_s: F64
    available: B
    q_meas: F64
    q_true: F64


@dataclass
class VehicleSignals:
    """Every measured signal of one vehicle, each with its truth (Spec: Signals table)."""

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
    """Ticks on the sensor's sampling grid (multiples of rate_s from t = 0)."""
    out: B = np.isclose(np.mod(np.asarray(t_s, dtype=np.float64), rate_s), 0.0)
    return out


def gnss(rng: np.random.Generator, view: ParamView, t_s: F64, r_km: F64, v_km_s: F64
         ) -> tuple[Sampled, Sampled]:
    """Spec: GNSS position and velocity, truth + N(0, 1.5 m), N(0, 5 cm/s) per axis, every 60 s.
    Extended faults (gnss_jump, gnss_noise): a position offset on every axis, and the noise
    times its multiplier on position and velocity."""
    m = every(t_s, view.get("signals.gnss.rate_s"))
    ts = t_s[m]
    n = int(m.sum())
    dr = rng.normal(0.0, view.get("signals.gnss.pos_sigma_m") / M_PER_KM, (n, 3))
    dv = rng.normal(0.0, view.get("signals.gnss.vel_sigma_m_s") / M_PER_KM, (n, 3))
    mult = view.get_series("sensor_faults.gnss.noise_mult", ts)[:, None]
    offset = view.get_series("sensor_faults.gnss.offset_m", ts)[:, None] / M_PER_KM
    return (Sampled(ts, r_km[m] + dr * mult + offset, r_km[m].copy()),
            Sampled(ts, v_km_s[m] + dv * mult, v_km_s[m].copy()))


def tracker_basis(view: ParamView) -> F64:
    """3x3, columns = tracker x, y, z (z = boresight) in body axes."""
    from expand3.mission.pointing import axis_from_label

    st = view.params.bus(view.sat_id).star_tracker
    face = axis_from_label(st.face)
    tilt = np.deg2rad(view.get("star_tracker.tilt_below_level_deg"))
    down = np.array([0.0, 0.0, -1.0])
    z = np.cos(tilt) * face + np.sin(tilt) * down
    x = np.cross(face, down)
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 0 else np.array([1.0, 0.0, 0.0])
    if x @ np.array([1.0, 0.0, 0.0]) < 0:
        x = -x
    return np.column_stack([x, np.cross(z, x), z])


def _dcms(q: F64) -> F64:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def star_tracker_available(view: ParamView, q: F64, r_eci_km: F64, sun_eci: F64) -> B:
    """Spec: no attitude fix inside the Sun or Earth exclusion (Earth cone from the limb)."""
    b = np.einsum("nij,j->ni", _dcms(np.atleast_2d(q)), tracker_basis(view)[:, 2])
    r = np.atleast_2d(r_eci_km)
    rn = np.linalg.norm(r, axis=-1)
    to_earth = -r / rn[:, None]
    sun_angle = np.rad2deg(np.arccos(np.clip(np.sum(b * np.atleast_2d(sun_eci), -1), -1, 1)))
    earth_angle = np.rad2deg(np.arccos(np.clip(np.sum(b * to_earth, -1), -1, 1)))
    limb = np.rad2deg(np.arcsin(np.clip(view.get("scenario.constants.R_E_km") / rn, -1, 1)))
    out: B = (sun_angle > view.get("star_tracker.sun_exclusion_deg")) & (
        earth_angle - limb > view.get("star_tracker.earth_exclusion_deg"))
    return out


def star_tracker(rng: np.random.Generator, view: ParamView, t_s: F64, q: F64, r_eci_km: F64,
                 sun_eci: F64) -> StarTrackerSignal:
    """Spec: attitude fix with cross-boresight and roll noise, none inside the exclusions."""
    m = every(t_s, view.get("signals.star_tracker.rate_s"))
    n = int(m.sum())
    sig = _ARCSEC_RAD * np.array([view.get("star_tracker.noise_cross_arcsec"),
                                  view.get("star_tracker.noise_cross_arcsec"),
                                  view.get("star_tracker.noise_roll_arcsec")])
    phi_body = (rng.normal(size=(n, 3)) * sig) @ tracker_basis(view).T
    angle = np.linalg.norm(phi_body, axis=-1)
    axis = phi_body / np.where(angle > 0, angle, 1.0)[:, None]
    dq = np.concatenate([np.cos(angle / 2)[:, None], np.sin(angle / 2)[:, None] * axis], axis=1)
    qt = q[m]
    w1, v1 = qt[:, :1], qt[:, 1:]
    w2, v2 = dq[:, :1], dq[:, 1:]
    q_meas = np.concatenate([w1 * w2 - np.sum(v1 * v2, -1, keepdims=True),
                             w1 * v2 + w2 * v1 + np.cross(v1, v2)], axis=1)
    available = star_tracker_available(view, qt, r_eci_km[m], sun_eci[m])
    q_meas[~available] = np.nan
    return StarTrackerSignal(t_s[m], available, q_meas, qt.copy())


def hold_last_healthy(meas: F64, flag: F64) -> F64:
    """Where `flag` is above one half the reading stops updating: it repeats the last sample
    taken with the flag down (the first sample if it never was)."""
    out = np.array(meas, dtype=np.float64)
    stuck = flag > 0.5
    if not stuck.any():
        return out
    healthy = np.flatnonzero(~stuck)
    last = np.maximum(np.searchsorted(healthy, np.arange(len(out))) - 1, 0)
    src = healthy[last] if healthy.size else np.zeros(len(out), dtype=np.int64)
    out[stuck] = meas[src[stuck]]
    return out


def imu(rng: np.random.Generator, view: ParamView, t_s: F64, omega: F64) -> Sampled:
    """Spec: body rate + N(0, 5e-5 rad/s), every 30 s. Extended faults (gyro_bias, gyro_scale,
    gyro_noise, gyro_stuck): scale x truth + bias + the noise times its multiplier, then frozen."""
    m = every(t_s, view.get("signals.imu.rate_s"))
    ts = t_s[m]
    noise = rng.normal(0.0, view.get("signals.imu.sigma_rad_s"), (int(m.sum()), 3))
    bias = view.get_series("sensor_faults.imu.bias_rad_s", ts)[:, None]
    scale = view.get_series("sensor_faults.imu.scale", ts)[:, None]
    mult = view.get_series("sensor_faults.imu.noise_mult", ts)[:, None]
    meas = omega[m] * scale + bias + noise * mult
    meas = hold_last_healthy(meas, view.get_series("sensor_faults.imu.stuck", ts))
    return Sampled(ts, meas, omega[m].copy())


def pack_voltage(rng: np.random.Generator, view: ParamView, t_s: F64, v: F64) -> Sampled:
    """Spec: V + N(0, 0.04 V)."""
    m = every(t_s, view.get("signals.pack_voltage.rate_s"))
    noise = rng.normal(0.0, view.get("signals.pack_voltage.sigma_V"), int(m.sum()))
    return Sampled(t_s[m], v[m] + noise, v[m].copy())


def pack_current(rng: np.random.Generator, view: ParamView, t_s: F64, i: F64) -> Sampled:
    """Spec: I + 0.05 A bias + N(0, 0.02 A)."""
    m = every(t_s, view.get("signals.pack_current.rate_s"))
    noise = rng.normal(0.0, view.get("signals.pack_current.sigma_A"), int(m.sum()))
    return Sampled(t_s[m], i[m] + view.get("signals.pack_current.bias_A") + noise, i[m].copy())


def string_power(rng: np.random.Generator, view: ParamView, t_s: F64, p: F64) -> Sampled:
    """Spec: per-string power as modeled, times 1 + N(0, 1 %) current-sense noise."""
    m = every(t_s, view.get("signals.string_power.rate_s"))
    noise = rng.normal(0.0, view.get("signals.string_power.rel_sigma"), p[m].shape)
    return Sampled(t_s[m], p[m] * (1.0 + noise), p[m].copy())


def thermistor_noise(rng: np.random.Generator, view: ParamView, n_ticks: int, n_nodes: int) -> F64:
    """Spec: node temperature T + N(0, 0.5 K); drawn up front because EPS uses the sensed battery
    temperature inside the run."""
    return rng.normal(0.0, view.get("signals.node_temperature.sigma_K"), (n_ticks, n_nodes))


def soc_estimate(rng: np.random.Generator, view: ParamView, t_s: F64, v_meas: F64, i_meas: F64,
                 soc_true0: float) -> F64:
    """Spec: SoC estimate, coulomb count of measured V x I, initial error N(0, 2 %),
    re-referenced to OCV where the curve is steep."""
    e_cap = view.get("eps.E_cap_J")
    soc_curve, volts = view.array("eps.ocv_curve.soc"), view.array("eps.ocv_curve.volts")
    low, high = view.get("signals.soc_estimate.steep_soc_low"), view.get(
        "signals.soc_estimate.steep_soc_high")
    rest = view.get("signals.soc_estimate.rest_current_A")
    est = np.empty(len(t_s))
    soc = soc_true0 + rng.normal(0.0, view.get("signals.soc_estimate.initial_error_sigma"))
    for n in range(len(t_s)):
        if (soc < low or soc > high) and abs(i_meas[n]) < rest:
            soc = float(np.interp(v_meas[n], volts, soc_curve))
        est[n] = soc
        if n + 1 < len(t_s):
            soc += v_meas[n] * i_meas[n] * float(t_s[n + 1] - t_s[n]) / e_cap
    return est


def tank_gauge(rng: np.random.Generator, view: ParamView, t_s: F64, propellant_kg: F64,
               burning: B) -> Sampled:
    """Spec: burn record tank gauge, propellant remaining + N(0, 0.1 kg), while burning."""
    m = np.asarray(burning, dtype=bool) & every(t_s, view.get("signals.tank_gauge.rate_s"))
    noise = rng.normal(0.0, view.get("signals.tank_gauge.sigma_kg"), int(m.sum()))
    return Sampled(t_s[m], propellant_kg[m] + noise, propellant_kg[m].copy())
