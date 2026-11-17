"""The mission thread end to end: six vehicles, burns, takes, slews, pointing (M3).

Spec: Scenario > Mission thread; Model: orbit and attitude (pointing law); Model: actuators;
Simplifications (ideal pointing between slews, may use the true state).

Run order:
1. Precompute frames, Sun and station geometry at every control tick (astropy, once).
2. Integrate all six orbits with RK4 at the integration step. Burns run as a state machine
   per burn vehicle: command, uplink pass, burn from loss of signal, ZOH prograde thrust.
3. From the trajectory, find each take vehicle's take tick and target, and plan the slew in
   (ending at take start) and the slew out (back to idle at take end).
4. If a burn overlaps a take's slews and take, integrate again with the burn suspended over
   that window (on the integration-step grid), and confirm the take ticks did not move.
5. Build per-tick attitude (pointing law or slew history), wheel momentum, rates and gates.

Body rate between slews is the finite-difference rate of the ideal attitude from one tick to
the next, and wheel momentum is held constant between slews (CLAUDE.md Known gaps).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from expand3.env.eclipse import shadow_fraction
from expand3.env.frames import eci_to_ecef
from expand3.env.geomag import FieldDisturbance, igrf_field_ecef_nt
from expand3.env.ground import Station, elevation_deg, stations
from expand3.env.sun import sun_eci
from expand3.faults import FaultConfig, FaultLabel, Injector, draw_fault, draw_fault_extended
from expand3.links import CrosslinkLog, crosslink
from expand3.mission import pointing
from expand3.mission.burns import BurnController, BurnRecord, plan_burns
from expand3.mission.slew import SlewResult, plan_slew, slew
from expand3.mission.takes import (
    TakeRecord,
    find_take_tick,
    off_nadir_deg,
    subsatellite_point,
    sun_elevation_deg,
)
from expand3.params import Params, ParamView
from expand3.sensors import (
    Sampled,
    StarTrackerSignal,
    VehicleSignals,
    every,
    gnss,
    imu,
    magnetometer,
    pack_current,
    pack_voltage,
    soc_estimate,
    star_tracker,
    string_power,
    tank_gauge,
    thermistor_noise,
)
from expand3.timebase import Clock, SeedDraws, draw_seed, rng
from expand3.vehicle.attitude import quat_conj, quat_mul, quat_rotate, quat_to_dcm
from expand3.vehicle.cdh import SaaHistory, saa_history
from expand3.vehicle.eps import BATTERY_NODE, PAYLOAD_NODE, Eps, EpsHistory, EpsInputs, history
from expand3.vehicle.magref import onboard_field
from expand3.vehicle.nav import nav_model, navigate
from expand3.vehicle.orbit import Forces, OrbitModel, initial_states, rk4_step
from expand3.vehicle.payload import dark_current_e_s_px
from expand3.vehicle.thermal import (
    Thermal,
    ThermalHistory,
    orbit_heat,
    settled_initial_temps,
    step_power_and_thermal,
    thermal_env,
)
from expand3.vehicle.thruster import thrust_step

F64 = NDArray[np.float64]
B = NDArray[np.bool_]

_MAX_REPLANS = 3
_SLEW_TIME_ITERATIONS = 6


@dataclass(frozen=True)
class MissionRun:
    """Per-tick truth, indexed [tick, vehicle, ...]; pass gates are [tick, vehicle, station]."""

    draws: SeedDraws
    sats: list[str]
    station_names: list[str]
    t_s: F64
    r_km: F64
    v_km_s: F64
    q: F64
    omega: F64
    h_wheel: F64
    mass_kg: F64
    propellant_kg: F64
    a_cmd_m_s2: F64
    dv_m_s: F64
    gate_pass: B
    gate_take: B
    gate_slew: B
    gate_burn: B
    eci_to_ecef: F64
    sun_unit: F64
    sun_dist_km: F64
    shadow: F64
    elevation_deg: F64
    burns: dict[str, BurnRecord]
    takes: dict[str, TakeRecord]
    slews: dict[str, list[SlewResult]]
    wall_s: float
    eps: dict[str, EpsHistory] = field(default_factory=dict)
    thermal: dict[str, ThermalHistory] = field(default_factory=dict)
    saa: dict[str, SaaHistory] = field(default_factory=dict)
    signals: dict[str, VehicleSignals] = field(default_factory=dict)
    crosslink: CrosslinkLog | None = None
    fault: FaultLabel | None = None


@dataclass
class _Trajectory:
    r: F64
    v: F64
    mass: F64
    prop: F64
    a_cmd: F64
    dv: F64
    burning: B
    burns: dict[str, BurnRecord]


@dataclass
class _TakePlan:
    n_take: int
    target_ecef: F64
    lat: float
    lon: float
    t_take: float
    t_end: float
    q_slew_in_0: F64
    q_take_0: F64
    t_slew_in: float
    q_take_end: F64
    q_idle_end: F64
    window: tuple[float, float]


def _hermite(t: float, t0: float, h: float, r0: F64, v0: F64, r1: F64, v1: F64) -> tuple[F64, F64]:
    """Cubic Hermite position and velocity between two ticks."""
    s = (t - t0) / h
    h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
    h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
    d00, d10, d01, d11 = 6 * s**2 - 6 * s, 3 * s**2 - 4 * s + 1, -6 * s**2 + 6 * s, 3 * s**2 - 2 * s
    r = h00 * r0 + h10 * h * v0 + h01 * r1 + h11 * h * v1
    v = (d00 * r0 + d10 * h * v0 + d01 * r1 + d11 * h * v1) / h
    return r, v


def _unit(x: F64) -> F64:
    out: F64 = x / np.linalg.norm(x)
    return out


def run_mission(
    params: Params,
    seed: int,
    duration_s: float | None = None,
    draws: SeedDraws | None = None,
    overrides: dict[str, dict[str, float]] | None = None,
    fault: FaultConfig | None = None,
) -> MissionRun:
    """Propagate all six vehicles end to end with the mission thread.

    `overrides` maps a vehicle to fixed ParamView overrides (dotted path -> value). `fault`
    selects the seeded fault draw (default), no fault, or a forced fault (and lead); the
    injector schedules the faulted parameter(s) on that vehicle's ParamView, and every model
    reads them at the current time.
    """
    start_wall = time.perf_counter()
    draws = draws or draw_seed(params, seed)
    sats = list(params.sat_ids)
    views = {s: ParamView(params, s) for s in sats}
    for sat, ov in (overrides or {}).items():
        views[sat].overrides.update(ov)
    v0 = views[sats[0]]
    clock = Clock.from_view(v0, draws.run_length_s if duration_s is None else duration_s)
    t = clock.tick_times_s()
    tick, dt = clock.tick_s, clock.step_s

    rot = eci_to_ecef(draws.epoch_utc, t)
    pole = np.asarray(rot[0, 2], dtype=np.float64)
    sun_u, sun_d = sun_eci(draws.epoch_utc, t)
    sun_ecef = np.einsum("nij,nj->ni", rot, sun_u * sun_d[:, None])
    sites = stations(v0)
    mask = v0.get("scenario.elevation_mask_deg")
    model = OrbitModel.from_params(params, sats, draws, pole, Forces())
    y0 = initial_states(params, sats, draws.epoch_utc)

    injector = _fault_setup(params, fault or FaultConfig(), draws, clock, rot, sun_u, sun_ecef,
                            sites, mask, pole, overrides or {})
    if injector is not None:
        injector.install(views[injector.fault.vehicle])

    suspend: dict[str, list[tuple[float, float]]] = {}
    plans: dict[str, _TakePlan] = {}
    traj = _integrate(params, views, sats, draws, clock, model, y0, rot, sun_u, sites, mask,
                      suspend)
    for _ in range(_MAX_REPLANS):
        plans = {
            sat: p
            for sat in params.scenario.take_vehicles
            if (p := _plan_take(views[sat], sats.index(sat), t, tick, dt, traj, rot, sun_u,
                                sun_ecef)) is not None
        }
        new_suspend = {
            sat: [p.window] for sat, p in plans.items()
            if sat in traj.burns and _overlaps(traj.burns[sat], p.window)
        }
        if new_suspend == suspend:
            break
        suspend = new_suspend
        traj = _integrate(params, views, sats, draws, clock, model, y0, rot, sun_u, sites, mask,
                          suspend)

    # Attitude, wheels and gates per tick.
    n_t, n_k = len(t), len(sats)
    q = np.zeros((n_t, n_k, 4))
    h_wheel = np.zeros((n_t, n_k, 4))
    omega = np.zeros((n_t, n_k, 3))
    gate_take = np.zeros((n_t, n_k), dtype=bool)
    gate_slew = np.zeros((n_t, n_k), dtype=bool)
    slews: dict[str, list[SlewResult]] = {}
    takes: dict[str, TakeRecord] = {}
    normal = np.cross(traj.r, traj.v)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    for k, sat in enumerate(sats):
        view = views[sat]
        for n in range(n_t):
            if traj.burning[n, k] and np.any(traj.a_cmd[n, k]):
                q[n, k] = pointing.burn(view, traj.a_cmd[n, k], sun_u[n])
            else:
                q[n, k] = pointing.sun_hold(view, sun_u[n], normal[n, k])
        dynamic = np.zeros(n_t, dtype=bool)
        if sat in plans:
            p = plans[sat]
            h_in = _coast(view, np.zeros(h_wheel.shape[-1]), 0.0, p.t_slew_in, tick)
            s_in = slew(view.at(p.t_slew_in), p.q_slew_in_0, p.q_take_0, h0=h_in)
            h_out = _coast(view, s_in.h[-1], p.t_take, p.t_end, tick)
            s_out = slew(view.at(p.t_end), p.q_take_end, p.q_idle_end, h0=h_out)
            slews[sat] = [s_in, s_out]
            t_out_end = p.t_end + s_out.duration_s
            rec = TakeRecord(sat, p.t_take, p.t_end, p.target_ecef, p.lat, p.lon,
                             slew_in_start_s=p.t_slew_in, slew_out_end_s=t_out_end)
            for res, t0 in ((s_in, p.t_slew_in), (s_out, p.t_end)):
                for j, ts in enumerate(res.t_s):
                    hit = np.flatnonzero(np.isclose(t, t0 + ts))
                    if hit.size and not (p.t_take <= t[hit[0]] <= p.t_end):
                        n = int(hit[0])
                        q[n, k], omega[n, k], h_wheel[n, k] = res.q[j], res.omega[j], res.h[j]
                        dynamic[n] = True
            in_slew = ((t >= p.t_slew_in) & (t < p.t_take)) | ((t > p.t_end) & (t <= t_out_end))
            gate_slew[:, k] = in_slew
            anchors = [(p.t_take, s_in.h[-1]), (t_out_end, s_out.h[-1])]
            boresight = pointing.axis_from_label(params.bus(sat).imager.face)
            for n in (int(i) for i in np.flatnonzero((t >= p.t_take) & (t <= p.t_end))):
                gate_take[n, k] = True
                q[n, k] = pointing.take(view, traj.r[n, k], p.target_ecef, rot[n], normal[n, k])
                target_eci = rot[n].T @ p.target_ecef
                los = _unit(target_eci - traj.r[n, k])
                b = quat_rotate(q[n, k], boresight)
                _, up, _, _ = subsatellite_point(p.target_ecef)
                rec.tick_times_s.append(float(t[n]))
                rec.sun_elevation_deg.append(float(sun_elevation_deg(
                    p.target_ecef[None], up, sun_ecef[n][None])[0]))
                rec.off_nadir_deg.append(float(off_nadir_deg(traj.r[n, k][None],
                                                             target_eci[None])[0]))
                rec.boresight_error_rad.append(float(np.arctan2(np.linalg.norm(np.cross(b, los)),
                                                                b @ los)))
            takes[sat] = rec
        else:
            anchors = []
        h_wheel[:, k] = _wheel_ticks(view, t, tick, h_wheel[:, k], dynamic, anchors)
        for n in range(n_t):
            if not dynamic[n]:
                m = min(n + 1, n_t - 1)
                lo = n if m > n else n - 1
                d = quat_mul(quat_conj(q[lo, k]), q[lo + 1, k])
                d = d if d[0] >= 0 else -d
                omega[n, k] = 2.0 * d[1:] / tick

    elev = np.stack(
        [elevation_deg(np.einsum("nij,nkj->nki", rot, traj.r), s) for s in sites], axis=-1
    )
    shadow = shadow_fraction(
        traj.r, sun_u[:, None, :], sun_d[:, None],
        v0.get("scenario.constants.R_E_km"), v0.get("scenario.constants.R_sun_km"),
    )
    gate_pass = elev >= mask
    r_ecef = np.einsum("nij,nkj->nki", rot, traj.r)
    saa = {
        sat: saa_history(draws.seed, views[sat], t, r_ecef[:, k], draws.epoch_utc)
        for k, sat in enumerate(sats)
    }
    eps: dict[str, EpsHistory] = {}
    thermal: dict[str, ThermalHistory] = {}
    therm_noise: dict[str, F64] = {}
    for k, sat in enumerate(sats):
        n_nodes = len(Thermal(views[sat]).names)
        therm_noise[sat] = thermistor_noise(rng(draws.seed, f"noise.thermistor.{sat}"),
                                            views[sat], len(t), n_nodes)
        eps[sat], thermal[sat] = _run_power_thermal(
            views[sat], k, t, tick, q, traj.r, shadow, sun_u, gate_pass, takes.get(sat),
            slews.get(sat, []), saa[sat].edge_off, therm_noise[sat],
        )
    for sat, rec in takes.items():
        n = int(np.flatnonzero(np.isclose(t, rec.t_take_s))[0])
        rec.payload_K = float(thermal[sat].node(PAYLOAD_NODE)[n])
        rec.dark_current_e_s_px = dark_current_e_s_px(views[sat].at(rec.t_take_s), rec.payload_K)
    field = FieldDisturbance.draw(views[sats[0]], draws.seed)
    signals = {
        sat: _vehicle_signals(
            params, pole, rot, draws.epoch_utc, field, draws.seed, views[sat], k, t, traj, q,
            omega, sun_u, gate_pass, gate_take, gate_slew, eps[sat], thermal[sat], therm_noise[sat],
            saa[sat],
        )
        for k, sat in enumerate(sats)
    }
    link_log = crosslink(draws.seed, params, views, sats, t, traj.r, traj.v)
    return MissionRun(
        draws=draws,
        sats=sats,
        station_names=[s.name for s in sites],
        t_s=t,
        r_km=traj.r,
        v_km_s=traj.v,
        q=q,
        omega=omega,
        h_wheel=h_wheel,
        mass_kg=traj.mass,
        propellant_kg=traj.prop,
        a_cmd_m_s2=traj.a_cmd,
        dv_m_s=traj.dv,
        gate_pass=gate_pass,
        gate_take=gate_take,
        gate_slew=gate_slew,
        gate_burn=traj.burning,
        eci_to_ecef=rot,
        sun_unit=sun_u,
        sun_dist_km=sun_d,
        shadow=shadow,
        elevation_deg=elev,
        burns=traj.burns,
        takes=takes,
        slews=slews,
        wall_s=time.perf_counter() - start_wall,
        eps=eps,
        thermal=thermal,
        saa=saa,
        signals=signals,
        crosslink=link_log,
        fault=injector.label(t) if injector is not None else None,
    )


def _overlaps(rec: BurnRecord, window: tuple[float, float]) -> bool:
    if rec.burn_start_s is None:
        return False
    end = rec.burn_end_s if rec.burn_end_s is not None else math.inf
    return rec.burn_start_s < window[1] and end > window[0]


def _integrate(
    params: Params,
    views: dict[str, ParamView],
    sats: list[str],
    draws: SeedDraws,
    clock: Clock,
    model: OrbitModel,
    y0: F64,
    rot: F64,
    sun_u: F64,
    sites: list[Station],
    mask: float,
    suspend: dict[str, list[tuple[float, float]]],
) -> _Trajectory:
    """Integrate the six orbits with the burn state machines (Spec: Model: orbit and attitude,
    Model: actuators)."""
    tick, dt, spt = clock.tick_s, clock.step_s, clock.steps_per_tick
    n_t, n_k = clock.n_ticks, len(sats)
    records = plan_burns(params, draws, dt, views)
    ctrls = {
        s: BurnController(records[s], tick, dt, suspend.get(s)) for s in records
    }
    burn_idx = [(sats.index(s), s) for s in records if s in sats]
    mass = np.array([views[s].get("mass.wet_mass_kg") for s in sats])
    prop = np.array([views[s].get("thruster.propellant_kg") for s in sats])
    out = _Trajectory(
        r=np.zeros((n_t, n_k, 3)), v=np.zeros((n_t, n_k, 3)), mass=np.zeros((n_t, n_k)),
        prop=np.zeros((n_t, n_k)), a_cmd=np.zeros((n_t, n_k, 3)), dv=np.zeros((n_t, n_k)),
        burning=np.zeros((n_t, n_k), dtype=bool), burns=records,
    )
    dv = np.zeros(n_k)
    y = np.array(y0, dtype=np.float64)
    for n in range(n_t):
        out.r[n], out.v[n] = y[:, :3], y[:, 3:]
        out.mass[n], out.prop[n], out.dv[n] = mass, prop, dv
        for k, sat in burn_idx:
            r_ecef = rot[n] @ y[k, :3]
            visible = any(float(elevation_deg(r_ecef, s)) >= mask for s in sites)
            ctrls[sat].at_tick(float(n * tick), visible)
        if n == n_t - 1:
            break
        for j in range(spt):
            t_step = n * tick + j * dt
            a_thr = np.zeros((n_k, 3))
            m_start = mass.copy()  # drag and thrust use the mass at the step start
            for k, sat in burn_idx:
                ctrl = ctrls[sat]
                if not ctrl.active(t_step):
                    continue
                view = views[sat]
                a_cmd = ctrl.rec.accel_m_s2 * _unit(y[k, 3:])
                if j == 0:
                    out.burning[n, k] = True
                    out.a_cmd[n, k] = a_cmd
                ctrl.flew(dt)
                if ctrl.rec.accel_m_s2 <= 0:
                    continue  # a zero-delta-v burn orders nothing
                q = pointing.burn(view, a_cmd, sun_u[n])
                st = thrust_step(view.at(t_step), a_cmd, float(m_start[k]), float(prop[k]), q, dt)
                a_thr[k] = st.a_thr_km_s2
                ctrl.rec.dv_m_s += st.dv_m_s
                ctrl.rec.propellant_used_kg += st.dm_kg
                dv[k] += st.dv_m_s
                mass[k] -= st.dm_kg
                prop[k] -= st.dm_kg
            def f(yy: F64, a: F64 = a_thr, m: F64 = m_start) -> F64:
                return model.deriv(yy, a, m)

            y = rk4_step(f, y, dt)
    return out


def _plan_take(
    view: ParamView,
    k: int,
    t: F64,
    tick: float,
    dt: float,
    traj: _Trajectory,
    rot: F64,
    sun_u: F64,
    sun_ecef: F64,
) -> _TakePlan | None:
    """Take tick, target, slew-in start (so the slew ends at take start) and the suspension
    window. Spec: Scenario > Mission thread; Model: actuators (slews bring the vehicle onto a
    take)."""
    r_ecef = np.einsum("nij,nj->ni", rot, traj.r[:, k])
    n = find_take_tick(view, t, r_ecef, sun_ecef)
    if n is None:
        return None
    point, _, lat, lon = subsatellite_point(r_ecef[n])
    target = point[0]
    t_take = float(t[n])
    t_end = t_take + view.get("scenario.take_duration_s")
    n_end = int(np.searchsorted(t, t_end))
    if n_end >= len(t) or not np.isclose(t[n_end], t_end):
        return None  # take does not fit in the run (or does not end on a tick)

    def normal_at(r: F64, v: F64) -> F64:
        return _unit(np.cross(r, v))

    q_take = pointing.take(
        view, traj.r[n, k], target, rot[n], normal_at(traj.r[n, k], traj.v[n, k])
    )
    rec = traj.burns.get(view.sat_id)

    def attitude_before(ts: float) -> F64:
        m = min(int(ts // tick), len(t) - 2)
        r, v = _hermite(ts, float(t[m]), tick, traj.r[m, k], traj.v[m, k], traj.r[m + 1, k],
                        traj.v[m + 1, k])
        burning = (
            rec is not None and rec.burn_start_s is not None and rec.burn_start_s <= ts
            and (rec.burn_end_s is None or ts < rec.burn_end_s)
        )
        if burning and rec is not None and rec.accel_m_s2 > 0:
            return pointing.burn(view, rec.accel_m_s2 * _unit(v), sun_u[m])
        return pointing.sun_hold(view, sun_u[m], normal_at(r, v))

    duration = tick
    q0 = attitude_before(t_take - duration)
    for _ in range(_SLEW_TIME_ITERATIONS):
        q0 = attitude_before(t_take - duration)
        new = plan_slew(view.at(t_take - duration), q0, q_take).duration_s
        if new == duration:
            break
        duration = new
    t_slew_in = t_take - duration

    q_take_end = pointing.take(view, traj.r[n_end, k], target, rot[n_end],
                               normal_at(traj.r[n_end, k], traj.v[n_end, k]))
    q_idle_end = pointing.sun_hold(view, sun_u[n_end],
                                   normal_at(traj.r[n_end, k], traj.v[n_end, k]))
    out_duration = plan_slew(view.at(t_end), q_take_end, q_idle_end).duration_s
    window = (
        math.floor(t_slew_in / dt) * dt,
        math.ceil((t_end + out_duration) / dt) * dt,
    )
    return _TakePlan(n, target, float(lat[0]), float(lon[0]), t_take, t_end, q0, q_take,
                     t_slew_in, q_take_end, q_idle_end, window)


def _wheel_power_per_tick(
    t: F64, tick: float, take: TakeRecord | None, slews: list[SlewResult], step_s: float
) -> F64:
    """Mean wheel torquing power over each tick interval [t_n, t_n + tick) from the slew
    histories (each slew step's power lasts one slew step)."""
    out = np.zeros(len(t))
    if take is None:
        return out
    for res, t0 in zip(slews, (take.slew_in_start_s, take.t_end_s), strict=False):
        for t_end_rel, p in zip(res.t_s, res.power_W, strict=True):
            mid = t0 + float(t_end_rel) - 0.5 * step_s
            n = int(mid // tick)
            if 0 <= n < len(t):
                out[n] += float(p) * step_s / tick
    return out


def _thermal_start(
    view: ParamView, eps: Eps, th: Thermal, k: int, t: F64, tick: float, q: F64, r: F64,
    shadow: F64, sun_u: F64,
) -> F64:
    """Initial node temperatures (scenario `thermal_init`). `settled` uses the first orbit of the
    run (idle Sun hold; burns and takes come hours later) as the periodic environment before
    epoch; `fixed` starts every node at `thermal_initial_K`."""
    view.at(float(t[0]))  # the settled start uses pre-epoch, fault-free values
    if view.params.scenario.thermal_init == "fixed":
        return th.initial_temps()
    mu = view.get("scenario.constants.mu_km3_s2")
    period = 2 * np.pi * np.sqrt(view.get("scenario.orbit.a_km") ** 3 / mu)
    n_orbit = min(round(period / tick), len(t))
    envs = [thermal_env(q[n, k], r[n, k], sun_u[n], float(shadow[n, k])) for n in range(n_orbit)]
    heats = orbit_heat(eps, th, envs, tick)
    orbits = view.params.scenario.thermal_init_orbits
    return settled_initial_temps(th, envs, heats, tick, orbits).T0


def _run_power_thermal(
    view: ParamView,
    k: int,
    t: F64,
    tick: float,
    q: F64,
    r: F64,
    shadow: F64,
    sun_u: F64,
    gate_pass: B,
    take: TakeRecord | None,
    slews: list[SlewResult],
    edge_off: B,
    therm_noise: F64,
) -> tuple[EpsHistory, ThermalHistory]:
    """Step one vehicle's EPS and thermal network over every tick (Spec: Model: electrical
    power; Model: thermal network). Each tick EPS runs with the node temperatures at the tick
    start, then the network steps with that tick's node heat. The take load applies to ticks
    with t_take <= t < t_end. The edge computer is off per the SAA history, and the cold-charge
    inhibit reads the battery thermistor (true temperature plus its noise)."""
    eps = Eps(view)
    th = Thermal(view)
    state = eps.initial_state()
    T = _thermal_start(view, eps, th, k, t, tick, q, r, shadow, sun_u)
    wheel_w = _wheel_power_per_tick(t, tick, take, slews, view.get("scenario.slew.step_s"))
    steps = []
    temps = np.empty((len(t), len(th.names)))
    ib = th.names.index(BATTERY_NODE)
    for n in range(len(t)):
        view.at(float(t[n]))  # faulted parameters (F1, F2, F7) at this tick
        temps[n] = T
        env = thermal_env(q[n, k], r[n, k], sun_u[n], float(shadow[n, k]))
        in_take = take is not None and take.t_take_s <= t[n] < take.t_end_s
        step, state, T = step_power_and_thermal(
            eps, th, state, T,
            EpsInputs(
                shadow=env.shadow,
                sun_body=env.sun_body,
                node_temps_K={},
                in_pass=bool(gate_pass[n, k].any()),
                take=bool(in_take),
                wheel_power_W=float(wheel_w[n]),
                edge_off=bool(edge_off[n]),
                sensed_battery_K=float(T[ib] + therm_noise[n, ib]),
            ),
            env, tick,
        )
        steps.append(step)
    return history(t, steps), ThermalHistory(t, list(th.names), temps)


def _magnetic_signals(
    view: ParamView, rng_mag: np.random.Generator, t: F64, rot: F64, epoch_utc: datetime,
    field: FieldDisturbance, r_eci: F64, q_true: F64, current_A: F64, nav_r_km: F64, nav_t_s: F64,
    tracker: StarTrackerSignal,
) -> tuple[Sampled, Sampled, Sampled]:
    """The magnetometer, the vehicle's onboard field model and its magnitude, on the magnetometer
    grid (Phase C). Truth is IGRF plus the common disturbance at the true position, in the body
    frame; the model is IGRF at the navigation position with the tracker's attitude."""
    grid = every(t, view.get("signals.magnetometer.rate_s"))
    n = np.flatnonzero(grid)
    r_ecef = np.einsum("nij,nj->ni", rot[n], r_eci[n])
    ideal = igrf_field_ecef_nt(r_ecef, epoch_utc)
    b_eci = np.einsum("nji,nj->ni", rot[n], ideal + field.at(r_ecef, t[n]))
    i_eci = np.einsum("nji,nj->ni", rot[n], ideal)
    dcm = np.stack([quat_to_dcm(q_true[i]).T for i in n])
    b_body = np.full((len(t), 3), np.nan)
    b_body[n] = np.einsum("nij,nj->ni", dcm, b_eci)
    ideal_body = np.einsum("nij,nj->ni", dcm, i_eci)
    mag = magnetometer(rng_mag, view, t, b_body, current_A)
    nav_i = np.searchsorted(nav_t_s, t[n])
    track_i = np.searchsorted(tracker.t_s, t[n])
    model, model_abs = onboard_field(nav_r_km[nav_i], rot[n], epoch_utc, tracker.q_meas[track_i])
    return (mag, Sampled(t[n], model, ideal_body),
            Sampled(t[n], model_abs, np.linalg.norm(ideal, axis=1)))


def _vehicle_signals(
    params: Params,
    pole: F64,
    rot: F64,
    epoch_utc: datetime,
    field: FieldDisturbance,
    seed: int,
    view: ParamView,
    k: int,
    t: F64,
    traj: _Trajectory,
    q: F64,
    omega: F64,
    sun_u: F64,
    gate_pass: B,
    gate_take: B,
    gate_slew: B,
    eps: EpsHistory,
    th: ThermalHistory,
    therm_noise: F64,
    saa: SaaHistory,
) -> VehicleSignals:
    """Every measured signal of one vehicle with its truth (Spec: Signals table), each sensor
    drawing from `noise.<sensor>.<sat>`."""
    sat = view.sat_id

    def stream(name: str) -> np.random.Generator:
        return rng(seed, f"noise.{name}.{sat}")

    r, v = traj.r[:, k], traj.v[:, k]
    pos, vel = gnss(stream("gnss"), view, t, r, v)
    nav = navigate(view, nav_model(params, sat, pole), pos.t_s, pos.meas, vel.meas, t,
                    traj.a_cmd[:, k])
    volts = pack_voltage(stream("voltage"), view, t, eps.v_V)
    amps = pack_current(stream("current"), view, t, eps.i_A)
    soc_meas = soc_estimate(stream("soc"), view, volts.t_s, volts.meas, amps.meas,
                            float(eps.soc[0]))
    burning = traj.burning[:, k]
    tracker = star_tracker(stream("star_tracker"), view, t, q[:, k], r, sun_u)
    mag, mag_model, mag_model_abs = _magnetic_signals(
        view, stream("mag"), t, rot, epoch_utc, field, r, q[:, k], eps.i_A, nav.r_km, pos.t_s,
        tracker)
    return VehicleSignals(
        gnss_r=pos,
        gnss_v=vel,
        nav_r=Sampled(pos.t_s, nav.r_km, pos.truth),
        nav_v=Sampled(vel.t_s, nav.v_km_s, vel.truth),
        mag=mag,
        mag_model=mag_model,
        mag_model_abs=mag_model_abs,
        star_tracker=tracker,
        imu=imu(stream("imu"), view, t, omega[:, k]),
        pack_voltage=volts,
        pack_current=amps,
        string_power=string_power(stream("string_power"), view, t, eps.p_string_W),
        node_temperature=Sampled(t, th.T_K + therm_noise, th.T_K),
        node_names=list(th.names),
        soc_estimate=Sampled(volts.t_s, soc_meas, eps.soc[every(t, view.get(
            "signals.soc_estimate.rate_s"))]),
        tank_gauge=tank_gauge(stream("tank"), view, t, traj.prop[:, k], burning),
        burn_a_cmd_m_s2=traj.a_cmd[:, k],
        burn_dv_m_s=traj.dv[:, k],
        burn_propellant_used_kg=traj.prop[0, k] - traj.prop[:, k],
        heater_commanded={"battery": eps.battery_heater, "payload": eps.payload_heater},
        heater_delivered={"battery": eps.loads_W["battery_heater"] > 0,
                          "payload": eps.loads_W["payload_heater"] > 0},
        gates={
            "pass": gate_pass[:, k].any(axis=1),
            "take": gate_take[:, k],
            "slew": gate_slew[:, k],
            "burn": burning,
            "saa": saa.in_saa,
            "shed": eps.shed,
            "uv_hold": eps.uv_hold,
        },
    )


def _fault_setup(
    params: Params,
    cfg: FaultConfig,
    draws: SeedDraws,
    clock: Clock,
    rot: F64,
    sun_u: F64,
    sun_ecef: F64,
    sites: list[Station],
    mask: float,
    pole: F64,
    overrides: dict[str, dict[str, float]],
) -> Injector | None:
    """Spec: Faults. Pick the fault (seeded draw, or forced), and arm it by the fault lead
    before the faulted vehicle's take start in the fault-free plan of the same seed, so a fault
    cannot move its own onset. Returns None for no fault or when that vehicle has no take in the
    run."""
    if cfg.mode == "none":
        return None
    profile = None
    if cfg.fault_set == "extended":
        fid, lead, profile = draw_fault_extended(params, draws.seed)
    else:
        fid, lead = draw_fault(params, draws.seed)
    if cfg.mode == "forced":
        if cfg.fault_id is None:
            raise ValueError("a forced fault needs fault_id")
        fid = cfg.fault_id
        lead = lead if cfg.lead_s is None else cfg.lead_s
        profile = cfg.profile
    sat = params.fault(fid).vehicle
    view = ParamView(params, sat)
    view.overrides.update(overrides.get(sat, {}))
    model = OrbitModel.from_params(params, [sat], draws, pole, Forces())
    y0 = initial_states(params, [sat], draws.epoch_utc)
    traj = _integrate(params, {sat: view}, [sat], draws, clock, model, y0, rot, sun_u, sites,
                      mask, {})
    t = clock.tick_times_s()
    plan = _plan_take(view, 0, t, clock.tick_s, clock.step_s, traj, rot, sun_u, sun_ecef)
    if plan is None:
        return None
    return Injector(params, fid, plan.t_take - lead, lead, clock.tick_s, profile, draws.seed)


def _coast(view: ParamView, h: F64, t_from: float, t_to: float, tick: float) -> F64:
    """Wheel momentum between slews: no torque, so only bearing friction acts, with the spec's
    discrete update at the slew step, c read at each tick (F3)."""
    step = view.get("scenario.slew.step_s")
    out = np.array(h, dtype=np.float64)
    t_now = t_from
    while t_now < t_to - 1e-9:
        span = min(tick - (t_now % tick) if t_now % tick else tick, t_to - t_now)
        c = view.at(t_now).array("wheels.c")
        out = out * (1.0 - c * step) ** (span / step)
        t_now += span
    return out


def _wheel_ticks(
    view: ParamView, t: F64, tick: float, h: F64, dynamic: B, anchors: list[tuple[float, F64]]
) -> F64:
    """Wheel momentum at every tick: slew samples where a slew is flying, otherwise the latest
    known state (start, or the end of a slew) coasted under bearing friction."""
    out = np.array(h, dtype=np.float64)
    t_ref, h_ref = 0.0, np.zeros(h.shape[1])
    pending = sorted(anchors, key=lambda a: a[0])
    for n in range(len(t)):
        while pending and pending[0][0] <= t[n]:
            t_ref, h_ref = pending[0][0], np.array(pending[0][1])
            pending.pop(0)
        if dynamic[n]:
            continue
        h_ref = _coast(view, h_ref, t_ref, float(t[n]), tick)
        t_ref = float(t[n])
        out[n] = h_ref
    return out

