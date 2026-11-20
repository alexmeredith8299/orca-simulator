"""Thermal network: node equation, topology, explicit Euler per control tick (M5).

Spec: Model: thermal network (F7); Parameters > Thermal nodes, Thermal links, limits and Bus B.

    C_i dT_i/dt = alpha_i A_si S s max(0, n_i . s_hat) + alpha_i A_si F_alb s max(0, n_i . d_hat)
                  + eps_i A_si F_IR v_i + Q_i + sum_j G_ij (T_j - T_i) - eps_i A_i sigma T_i^4
    F_alb = a_alb S (R_E/|r|)^2 max(0, -u_sun . d_hat),   F_IR = q_IR (R_E/|r|)^2

s_hat and d_hat (nadir) are in body axes. A_si is the area facing the source (a wing's sunward
side only); A_i is the full radiating area (both sides of a wing). Internal nodes (battery, obc,
radio_pa, payload, bus_core) get only Q and conduction. One explicit Euler step per control tick.

Topology (CLAUDE.md Known gaps): bus_core to each of the six faces; the side faces form a ring
(each x face to each y face); links named "<a>_to_<b>_W_K" join those two nodes; each wing to the
face with the same suffix (wing_px to bus_px); each internal node to bus_core. alpha and epsilon
are read through ParamView on every call to `network()`, so F7 applies from its onset.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

from expand3.mission.pointing import axis_from_label
from expand3.params import ParamView
from expand3.vehicle.attitude import quat_to_dcm
from expand3.vehicle.eps import Eps, EpsInputs, EpsState, EpsStep

F64 = NDArray[np.float64]

_PAIR_LINK = re.compile(r"^(\w+?)_to_(\w+?)_W_K$")
_SIDE_RING = "face_to_face_around_sides_W_K"
_WING_HINGE = "wing_to_its_face_hinge_W_K"
_WING_PREFIX, _FACE_PREFIX = "wing_", "bus_"
_ORBIT_HEAT_MAX_CYCLES = 20
_ORBIT_HEAT_ENERGY_TOL_J = 1.0  # numerical convergence tolerance, not a model value


@dataclass(frozen=True)
class ThermalEnv:
    """Environment seen by the vehicle at one tick: shadow fraction s, Sun and nadir unit vectors
    in body axes, orbit radius [km], and u_sun . d_hat (frame independent)."""

    shadow: float
    sun_body: F64
    nadir_body: F64
    r_km: float
    sun_dot_nadir: float


def thermal_env(q: F64, r_eci_km: F64, sun_eci: F64, shadow: float) -> ThermalEnv:
    m = quat_to_dcm(q)
    r = float(np.linalg.norm(r_eci_km))
    nadir = -np.asarray(r_eci_km) / r
    return ThermalEnv(
        shadow=shadow,
        sun_body=m.T @ sun_eci,
        nadir_body=m.T @ nadir,
        r_km=r,
        sun_dot_nadir=float(np.asarray(sun_eci) @ nadir),
    )


@dataclass
class Network:
    """A thermal network: per-node arrays and the symmetric conductance matrix G [W/K]."""

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
        """Spec: Model: thermal network. Solar, albedo and Earth-IR absorbed power per node [W]."""
        scale = (self.R_E_km / env.r_km) ** 2
        cos_sun = np.maximum(0.0, self.normal @ env.sun_body)
        cos_nadir = np.maximum(0.0, self.normal @ env.nadir_body)
        f_alb = self.a_alb * self.S * scale * max(0.0, -env.sun_dot_nadir)
        f_ir = self.q_IR * scale
        solar = self.alpha * self.area_src * self.S * env.shadow * cos_sun
        albedo = self.alpha * self.area_src * f_alb * env.shadow * cos_nadir
        ir = self.eps * self.area_src * f_ir * self.view
        return solar, albedo, ir

    def rate(self, T: F64, Q: F64, env: ThermalEnv) -> F64:
        """dT/dt for every node [K/s]."""
        solar, albedo, ir = self.absorbed_w(env)
        conduction = self.G @ T - self.G.sum(axis=1) * T
        emitted = self.eps * self.area_rad * self.sigma * T**4
        out: F64 = (solar + albedo + ir + Q + conduction - emitted) / self.C
        return out

    def step(self, T: F64, Q: F64, env: ThermalEnv, dt: float) -> F64:
        """One explicit Euler step. Spec: one step per control tick."""
        out: F64 = T + dt * self.rate(T, Q, env)
        return out

    def linearized(self, T_ref: float) -> F64:
        """C^-1 (L + diag(4 eps A sigma T_ref^3)), L the conduction Laplacian."""
        lap = np.diag(self.G.sum(axis=1)) - self.G
        rad = np.diag(4 * self.eps * self.area_rad * self.sigma * T_ref**3)
        out: F64 = (lap + rad) / self.C[:, None]
        return out

    def stability_limit_s(self, T_ref: float) -> float:
        """Explicit Euler step limit 2 / lambda_max of the linearized network."""
        lam = np.linalg.eigvals(self.linearized(T_ref)).real.max()
        return float(2.0 / lam)


class Thermal:
    """One vehicle's thermal network built from params (Bus A 15 nodes, Bus B 14)."""

    names: list[str]

    def __init__(self, view: ParamView) -> None:
        self.view = view
        self._cache: tuple[tuple[tuple[str, float], ...], Network] | None = None
        th = view.params.bus(view.sat_id).thermal
        self.names = list(th.nodes)
        n = len(self.names)
        ix = {name: i for i, name in enumerate(self.names)}
        g = view.get
        self.surface = [i for i, name in enumerate(self.names) if th.nodes[name].normal is not None]
        normal = np.zeros((n, 3))
        area_src, area_rad, vf = np.zeros(n), np.zeros(n), np.zeros(n)
        for i in self.surface:
            name, node = self.names[i], th.nodes[self.names[i]]
            assert node.normal is not None
            normal[i] = axis_from_label(node.normal)
            area_rad[i] = g(f"thermal.nodes.{name}.area_m2")
            area_src[i] = (g(f"thermal.nodes.{name}.sunward_m2") if node.sunward_m2 is not None
                           else area_rad[i])
            vf[i] = g(f"thermal.nodes.{name}.view")

        G = np.zeros((n, n))

        def link(a: str, b: str, value: float) -> None:
            G[ix[a], ix[b]] = G[ix[b], ix[a]] = value

        faces = [name for name in self.names
                 if name.startswith(_FACE_PREFIX) and th.nodes[name].normal is not None]
        for name, node in th.nodes.items():
            if node.G_to_each_face_W_K is not None:
                for face in faces:
                    link(name, face, g(f"thermal.nodes.{name}.G_to_each_face_W_K"))
                core = name
        for name, node in th.nodes.items():
            if node.G_to_core_W_K is not None:
                link(name, core, g(f"thermal.nodes.{name}.G_to_core_W_K"))
        sides = [f for f in faces if abs(normal[ix[f]][2]) < 0.5]
        for a in sides:
            for b in sides:
                if a < b and abs(normal[ix[a]] @ normal[ix[b]]) < 0.5:
                    link(a, b, g(f"thermal.links.{_SIDE_RING}"))
        for key in th.links.model_dump():
            m = _PAIR_LINK.match(key)
            if key in (_SIDE_RING, _WING_HINGE) or m is None:
                continue
            link(m.group(1), m.group(2), g(f"thermal.links.{key}"))
        for name in self.names:
            if name.startswith(_WING_PREFIX):
                face = _FACE_PREFIX + name[len(_WING_PREFIX):]
                link(name, face, g(f"thermal.links.{_WING_HINGE}"))

        self._static = Network(
            names=self.names,
            C=np.array([g(f"thermal.nodes.{name}.C_J_K") for name in self.names]),
            alpha=np.zeros(n),
            eps=np.zeros(n),
            area_src=area_src,
            area_rad=area_rad,
            normal=normal,
            view=vf,
            G=G,
            S=g("scenario.constants.S_W_m2"),
            a_alb=g("scenario.constants.a_alb"),
            q_IR=g("scenario.constants.q_IR_W_m2"),
            R_E_km=g("scenario.constants.R_E_km"),
            sigma=g("scenario.constants.sigma_W_m2_K4"),
        )

    def network(self) -> Network:
        """The network with alpha and epsilon read through ParamView now (F7)."""
        view = self.view
        if not any(p.startswith("thermal.") for p in view.schedule):
            # Nothing varies in time, so the network depends only on the thermal overrides.
            key = tuple(sorted(
                (p, v) for p, v in view.overrides.items() if p.startswith("thermal.")
            ))
            if self._cache is not None and self._cache[0] == key:
                return self._cache[1]
            net = self._read_network()
            self._cache = (key, net)
            return net
        return self._read_network()

    def _read_network(self) -> Network:
        n = len(self.names)
        alpha, eps = np.zeros(n), np.zeros(n)
        for i in self.surface:
            name = self.names[i]
            alpha[i] = self.view.get(f"thermal.nodes.{name}.alpha")
            eps[i] = self.view.get(f"thermal.nodes.{name}.epsilon")
        return dataclasses.replace(self._static, alpha=alpha, eps=eps)

    def initial_temps(self) -> F64:
        """Spec silent: every node at scenario `thermal_initial_K`."""
        return np.full(len(self.names), self.view.get("scenario.thermal_initial_K"))

    def heat_vector(self, heat_w: dict[str, float]) -> F64:
        q = np.zeros(len(self.names))
        for node, w in heat_w.items():
            q[self.names.index(node)] += w
        return q

    def temps_map(self, T: F64) -> dict[str, float]:
        return {name: float(T[i]) for i, name in enumerate(self.names)}

    def step(self, T: F64, heat_w: dict[str, float], env: ThermalEnv, dt: float) -> F64:
        return self.network().step(T, self.heat_vector(heat_w), env, dt)


@dataclass
class ThermalHistory:
    """Node temperatures [K] at every tick, columns in `names` order."""

    t_s: F64
    names: list[str]
    T_K: F64
    sensed_battery_K: F64 | None = None  # battery thermistor reading, with its fault

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
    """One coupled control tick: EPS runs with the node temperatures at the tick start, then the
    thermal network steps with this tick's node heat from EPS."""
    temps = thermal.temps_map(T)
    step, state = eps.step(eps_state, dataclasses.replace(inputs, node_temps_K=temps), dt)
    return step, state, thermal.step(T, step.heat_W, env, dt)


def run_offline(
    thermal: Thermal, envs: list[ThermalEnv], heats: list[dict[str, float]], dt: float,
    T0: F64 | None = None,
) -> F64:
    """Drive the network with given environments and node heat; returns T at each tick start."""
    T = thermal.initial_temps() if T0 is None else np.array(T0, dtype=np.float64)
    out = np.empty((len(envs), len(T)))
    for n, (env, heat) in enumerate(zip(envs, heats, strict=True)):
        out[n] = T
        T = thermal.step(T, heat, env, dt)
    return out


@dataclass(frozen=True)
class SettledInit:
    """Settled initial state: the orbit-averaged steady state, its per-node balance residual [W],
    the orbit-mean absorbed environment and node heat [W], and T0 after the periodic spin-up."""

    T0: F64
    T_steady: F64
    residual_W: F64
    absorbed_mean_W: F64
    heat_mean_W: F64


def orbit_heat(
    eps: Eps, thermal: Thermal, envs: list[ThermalEnv], dt: float
) -> list[dict[str, float]]:
    """Nominal node heat over one periodic orbit: EPS with no pass, take, slew or burn gates,
    heaters held off, node temperatures at the fixed initial value. The orbit is flown repeatedly
    from the initial SoC until the pack energy at the end of an orbit repeats (to 1 J), so the
    pack losses are those of the periodic orbit rather than of the initial charge."""
    temps = thermal.temps_map(thermal.initial_temps())
    state = eps.initial_state()
    out: list[dict[str, float]] = []
    for _ in range(_ORBIT_HEAT_MAX_CYCLES):
        e_start = state.E_J
        out = []
        for env in envs:
            state = dataclasses.replace(state, battery_heater=False, payload_heater=False)
            step, state = eps.step(state, EpsInputs(env.shadow, env.sun_body, temps), dt)
            out.append(step.heat_W)
        if abs(state.E_J - e_start) < _ORBIT_HEAT_ENERGY_TOL_J:
            break
    return out


def steady_state(network: Network, absorbed_W: F64, Q_W: F64, T_guess: F64) -> tuple[F64, F64]:
    """Solve absorbed + Q + conduction - radiation = 0 (nonlinear in T^4) for every node; returns
    the temperatures and the per-node residual [W]."""

    def balance(T: F64) -> F64:
        conduction = network.G @ T - network.G.sum(axis=1) * T
        emitted = network.eps * network.area_rad * network.sigma * T**4
        out: F64 = absorbed_W + Q_W + conduction - emitted
        return out

    sol = root(balance, T_guess, method="hybr", tol=1e-13)
    T = np.asarray(sol.x, dtype=np.float64)
    return T, balance(T)


def settled_initial_temps(
    thermal: Thermal,
    envs: list[ThermalEnv],
    heats: list[dict[str, float]],
    dt: float,
    orbits: int,
    shoot: bool = True,
) -> SettledInit:
    """Spec silent (CLAUDE.md Known gaps): solve the steady state of the orbit-averaged
    environment and node heat over one idle orbit, refine it to the periodic orbit of that
    environment (Newton shooting on the one-orbit map), then fly `orbits` orbits from it; the end
    state is T(t = 0). `shoot=False` skips the refinement (plain spin-up), for checks."""
    net = thermal.network()
    absorbed = np.mean([np.sum(net.absorbed_w(env), axis=0) for env in envs], axis=0)
    q_mean = np.mean([thermal.heat_vector(h) for h in heats], axis=0)
    T_ss, residual = steady_state(net, absorbed, q_mean, thermal.initial_temps())
    q_vecs = [thermal.heat_vector(h) for h in heats]

    def one_orbit(T: F64) -> F64:
        for env, q in zip(envs, q_vecs, strict=True):
            T = net.step(T, q, env, dt)
        return T

    # The steady state of the averaged inputs is not the periodic orbit (radiation goes as T^4),
    # and the battery's 5.6 h time constant would carry that offset for many orbits. Shoot for
    # the periodic orbit, T = one_orbit(T), starting from the steady state.
    T = T_ss.copy()
    if shoot:
        sol = root(lambda x: one_orbit(x) - x, T_ss, method="hybr", tol=1e-10)
        T = np.asarray(sol.x, dtype=np.float64)
    for _ in range(orbits):
        T = one_orbit(T)
    return SettledInit(T, T_ss, residual, absorbed, q_mean)

