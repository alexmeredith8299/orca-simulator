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

import numpy as np
from numpy.typing import NDArray

from expand3.env.eclipse import shadow_fraction
from expand3.env.frames import eci_to_ecef
from expand3.env.ground import Station, elevation_deg, stations
from expand3.env.sun import sun_eci
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
from expand3.timebase import Clock, SeedDraws, draw_seed
from expand3.vehicle.attitude import quat_conj, quat_mul, quat_rotate, quat_to_dcm
from expand3.vehicle.eps import Eps, EpsHistory, EpsInputs, history, placeholder_temperatures
from expand3.vehicle.orbit import Forces, OrbitModel, initial_states, rk4_step
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
) -> MissionRun:
    """Propagate all six vehicles end to end with the mission thread.

    `overrides` maps a vehicle to ParamView overrides (dotted path -> value), e.g. fault
    parameters; the fault injector (M7) will fill these with onset profiles.
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
            s_in = slew(view, p.q_slew_in_0, p.q_take_0, h0=h_wheel[0, k])
            s_out = slew(view, p.q_take_end, p.q_idle_end, h0=s_in.h[-1])
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
            h_wheel[t > p.t_take, k] = s_in.h[-1]
            h_wheel[t > t_out_end, k] = s_out.h[-1]
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
    eps = {
        sat: _run_eps(views[sat], k, t, tick, q, shadow, sun_u, gate_pass, takes.get(sat),
                      slews.get(sat, []))
        for k, sat in enumerate(sats)
    }
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
    burn_idx = [(sats.index(s), s) for s in records]
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
                st = thrust_step(view, a_cmd, float(m_start[k]), float(prop[k]), q, dt)
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
        new = plan_slew(view, q0, q_take).duration_s
        if new == duration:
            break
        duration = new
    t_slew_in = t_take - duration

    q_take_end = pointing.take(view, traj.r[n_end, k], target, rot[n_end],
                               normal_at(traj.r[n_end, k], traj.v[n_end, k]))
    q_idle_end = pointing.sun_hold(view, sun_u[n_end],
                                   normal_at(traj.r[n_end, k], traj.v[n_end, k]))
    out_duration = plan_slew(view, q_take_end, q_idle_end).duration_s
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


def _run_eps(
    view: ParamView,
    k: int,
    t: F64,
    tick: float,
    q: F64,
    shadow: F64,
    sun_u: F64,
    gate_pass: B,
    take: TakeRecord | None,
    slews: list[SlewResult],
) -> EpsHistory:
    """Step one vehicle's EPS over every tick (Spec: Model: electrical power). The take load
    applies to ticks with t_take <= t < t_end. Node temperatures are the placeholder until M5,
    and the SAA edge-computer gate is off until M6."""
    eps = Eps(view)
    state = eps.initial_state()
    temps = placeholder_temperatures(view)
    wheel_w = _wheel_power_per_tick(t, tick, take, slews, view.get("scenario.slew.step_s"))
    steps = []
    for n in range(len(t)):
        sun_body = quat_to_dcm(q[n, k]).T @ sun_u[n]
        in_take = take is not None and take.t_take_s <= t[n] < take.t_end_s
        step, state = eps.step(
            state,
            EpsInputs(
                shadow=float(shadow[n, k]),
                sun_body=sun_body,
                node_temps_K=temps,
                in_pass=bool(gate_pass[n, k].any()),
                take=bool(in_take),
                wheel_power_W=float(wheel_w[n]),
            ),
            tick,
        )
        steps.append(step)
    return history(t, steps)

