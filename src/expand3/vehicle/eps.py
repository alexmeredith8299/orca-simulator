"""Electrical power: array strings, loads, pack, terminal voltage, protections, heat (M4).

Spec: Model: electrical power (F1, F2); Parameters > Electrical power, Loads. Stepped once per
control tick, after attitude.

    P_k = S s eta_k(T_k) max(0, n_k . s_hat) A_k p_k o_k eta_ab,   eta_k = eta0 + deta/dT (T - T0)
    D = (sum of loads) / eta_bl,   N = P_gen - D
    P_acc = min(min(N+, P_c,max)(1 - iota), (E_cap - E)/(eta_c dt))
    P_sup = min(N-, eta_d min(E, E_use(T_b))/dt),   E_use = E_cap x derate(T_b)
    E_dot = eta_c P_acc - P_sup/eta_d,   SoC = E/E_cap,   P_pack = P_acc - P_sup
    V = 1/2 (OCV(SoC) + sqrt(OCV(SoC)^2 + 4 P_pack R)),   I = P_pack / V

Conventions (CLAUDE.md Known gaps): string k sits on wing k (wings.names order) and faces the
wing normal. E advances explicitly over the tick, and V and I use the SoC at the tick start.
Tables interpolate linearly and clamp at their ends. Each protection latch (undervoltage hold,
cold-charge inhibit, heaters) is evaluated on this tick's values and applied from the next tick.
Heaters use the true node temperature until M6 provides thermistor readings; the cold-charge
inhibit takes the sensed battery temperature when given. R (F1) and the string health o_k (F2)
are read through ParamView every step; the other values are read once per Eps.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from expand3.mission.pointing import axis_from_label
from expand3.params import ParamView

F64 = NDArray[np.float64]
B = NDArray[np.bool_]

BATTERY_NODE = "battery"
PAYLOAD_NODE = "payload"
CONVERTER_NODE = "bus_nz"  # spec: converter losses heat the bus_nz node
HEATERS = {"battery_heater": BATTERY_NODE, "payload_heater": PAYLOAD_NODE}


@dataclass(frozen=True)
class _LoadRow:
    tier: str
    node: str
    heat_fraction: float
    plus_wheels: bool
    draws: dict[str, float]


@dataclass(frozen=True)
class EpsInputs:
    """One tick's inputs: shadow fraction s, Sun unit vector in body axes, node temperatures [K]
    (from M5, or the placeholder), and the mission gates. wheel_power_W is the mean wheel
    torquing power over the tick."""

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
    """Stored pack energy [J] and the protection latches in force for the coming tick."""

    E_J: float
    uv_hold: bool = False
    inhibit: bool = False
    battery_heater: bool = False
    payload_heater: bool = False


@dataclass(frozen=True)
class EpsStep:
    """Everything the EPS computed for one tick (powers W, energy J, V, A)."""

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
    """Per-tick EPS truth for one vehicle. Latch flags are those in force during each tick."""

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
    """Temporary node temperatures until the M5 thermal network supplies them (scenario
    `placeholder_node_temperature_K`): every wing, the battery and the payload."""
    ph = view.params.scenario.placeholder_node_temperature_K
    out = {w: ph["wings"] for w in view.params.bus(view.sat_id).wings.names}
    out[BATTERY_NODE] = ph["battery"]
    out[PAYLOAD_NODE] = ph["payload"]
    return out


def history(t_s: F64, steps: list[EpsStep]) -> EpsHistory:
    def arr(name: str) -> F64:
        return np.array([getattr(s, name) for s in steps], dtype=np.float64)

    def flags(name: str) -> B:
        return np.array([getattr(s, name) for s in steps], dtype=bool)

    nodes = sorted({n for s in steps for n in s.heat_W})
    return EpsHistory(
        t_s=np.asarray(t_s, dtype=np.float64),
        p_string_W=np.array([s.p_string_W for s in steps]),
        p_gen_W=arr("p_gen_W"),
        loads_W={n: np.array([s.loads_W[n] for s in steps]) for n in steps[0].loads_W},
        demand_W=arr("demand_W"),
        p_pack_W=arr("p_pack_W"),
        curtailed_W=arr("curtailed_W"),
        unmet_W=arr("unmet_W"),
        E_J=arr("E_J"),
        soc=arr("soc"),
        v_V=arr("v_V"),
        i_A=arr("i_A"),
        shed=flags("shed"),
        uv_hold=flags("uv_hold"),
        inhibit=flags("inhibit"),
        battery_heater=flags("battery_heater"),
        payload_heater=flags("payload_heater"),
        disc_clamped=flags("disc_clamped"),
        heat_W={n: np.array([s.heat_W.get(n, 0.0) for s in steps]) for n in nodes},
    )


class Eps:
    """One vehicle's electrical power system."""

    def __init__(self, view: ParamView) -> None:
        self.view = view
        bus = view.params.bus(view.sat_id)
        self.wings = list(bus.wings.names)
        self.normal_b = axis_from_label(bus.wings.normal)
        g = view.get
        self.S = g("scenario.constants.S_W_m2")
        self.eta0, self.T0, self.deta = g("eps.eta0"), g("eps.T0_K"), g("eps.deta_dT_per_K")
        self.area, self.packing = g("eps.string_area_m2"), g("eps.packing")
        self.eta_ab, self.eta_bl = g("eps.eta_ab"), g("eps.eta_bl")
        self.E_cap, self.eta_c, self.eta_d = g("eps.E_cap_J"), g("eps.eta_c"), g("eps.eta_d")
        self.P_c_max = g("eps.P_c_max_W")
        self.ocv_soc, self.ocv_volts = view.array("eps.ocv_curve.soc"), view.array(
            "eps.ocv_curve.volts")
        self.derate_T, self.derate_f = view.array("eps.usable_derate.T_K"), view.array(
            "eps.usable_derate.factor")
        self.uv_shed = g("eps.undervoltage.shed_V")
        self.uv_release = g("eps.undervoltage.release_V")
        self.inhibit_k = g("eps.cold_charge_inhibit.inhibit_K")
        self.release_k = g("eps.cold_charge_inhibit.release_K")
        self.heater_band = {
            name: (g(f"eps.{name}.on_K"), g(f"eps.{name}.off_K")) for name in HEATERS
        }
        self.cutoff_k = g("eps.heater_cutoff_true_K")
        self.shed_tiers = set(bus.undervoltage_sheds_tiers)
        draw_keys = ("draw_W", "take_W", "on_W", "pass_no_take_W", "else_W",
                     "off_in_saa_and_reboot_W")
        self.loads = {
            name: _LoadRow(
                tier=row.tier,
                node=row.node,
                heat_fraction=g(f"loads.{name}.heat_fraction"),
                plus_wheels=row.plus is not None,
                draws={k: g(f"loads.{name}.{k}") for k in draw_keys if getattr(row, k) is not None},
            )
            for name, row in bus.loads.items()
        }

    def initial_state(self) -> EpsState:
        """Spec silent: initial SoC from scenario `eps_initial_soc`, latches off."""
        return EpsState(self.view.get("scenario.eps_initial_soc") * self.E_cap)

    def string_power_w(self, shadow: float, sun_body: F64, temps_k: Mapping[str, float]) -> F64:
        """Spec: Model: electrical power, array. One value per string k (on wing k)."""
        health = self.view.array("eps.string_health")
        cos = max(0.0, float(self.normal_b @ sun_body))
        eta = self.eta0 + self.deta * (np.array([temps_k[w] for w in self.wings]) - self.T0)
        out: F64 = self.S * shadow * eta * cos * self.area * self.packing * health * self.eta_ab
        return out

    def loads_w(self, state: EpsState, inputs: EpsInputs) -> dict[str, float]:
        """Spec: Parameters > Loads. Each draw with its gate, times availability (tier not shed)."""
        heaters_on = {
            "battery_heater": state.battery_heater, "payload_heater": state.payload_heater
        }
        out = {}
        for name, row in self.loads.items():
            d = row.draws
            if "on_W" in d:
                w = d["on_W"] if heaters_on.get(name, False) else 0.0
            elif "pass_no_take_W" in d:
                w = d["pass_no_take_W"] if inputs.in_pass and not inputs.take else d["else_W"]
            elif "off_in_saa_and_reboot_W" in d and inputs.edge_off:
                w = d["off_in_saa_and_reboot_W"]
            elif "take_W" in d and inputs.take:
                w = d["take_W"]
            else:
                w = d["draw_W"]
            if row.plus_wheels:
                w += inputs.wheel_power_W
            if state.uv_hold and row.tier in self.shed_tiers:
                w = 0.0
            out[name] = float(w)
        return out

    def usable_energy_j(self, battery_k: float) -> float:
        """E_use(T_b) = E_cap x derate(T_b), linear in the table, clamped at its ends."""
        return self.E_cap * float(np.interp(battery_k, self.derate_T, self.derate_f))

    def ocv_v(self, soc: float) -> float:
        return float(np.interp(soc, self.ocv_soc, self.ocv_volts))

    def step(self, state: EpsState, inputs: EpsInputs, dt: float) -> tuple[EpsStep, EpsState]:
        """Spec: Model: electrical power. One control tick."""
        temps = inputs.node_temps_K
        t_batt, t_pay = temps[BATTERY_NODE], temps[PAYLOAD_NODE]
        sensed = t_batt if inputs.sensed_battery_K is None else inputs.sensed_battery_K

        p_str = self.string_power_w(inputs.shadow, inputs.sun_body, temps)
        p_gen = float(p_str.sum())
        loads = self.loads_w(state, inputs)
        demand = sum(loads.values()) / self.eta_bl
        net = p_gen - demand
        n_pos, n_neg = max(net, 0.0), max(-net, 0.0)
        iota = 1.0 if state.inhibit else 0.0
        e = state.E_J
        p_acc = max(0.0, min(min(n_pos, self.P_c_max) * (1 - iota),
                             (self.E_cap - e) / (self.eta_c * dt)))
        p_sup = min(n_neg, self.eta_d * min(e, self.usable_energy_j(t_batt)) / dt)
        p_sup = max(0.0, p_sup)
        e_next = min(self.E_cap, max(0.0, e + (self.eta_c * p_acc - p_sup / self.eta_d) * dt))
        soc = e / self.E_cap
        ocv = self.ocv_v(soc)
        p_pack = p_acc - p_sup
        r = self.view.get("eps.R_pack_ohm")
        disc = ocv**2 + 4 * p_pack * r
        clamped = disc < 0
        v = 0.5 * (ocv + float(np.sqrt(max(disc, 0.0))))
        i = p_pack / v

        uv = state.uv_hold
        if not uv and v <= self.uv_shed:
            uv = True
        elif uv and v >= self.uv_release:
            uv = False
        inhibit = state.inhibit
        if not inhibit and sensed <= self.inhibit_k:
            inhibit = True
        elif inhibit and sensed > self.release_k:
            inhibit = False
        heaters = {}
        for name, node_t, on in (("battery_heater", t_batt, state.battery_heater),
                                 ("payload_heater", t_pay, state.payload_heater)):
            on_k, off_k = self.heater_band[name]
            if node_t <= on_k:
                on = True
            elif node_t >= off_k:
                on = False
            heaters[name] = on and node_t <= self.cutoff_k

        curtailed = n_pos - p_acc
        pack_loss = (1 - self.eta_c) * p_acc + (1 / self.eta_d - 1) * p_sup
        array_loss = (p_gen - curtailed) * (1 / self.eta_ab - 1)
        load_loss = demand * (1 - self.eta_bl)
        heat: dict[str, float] = {}
        for name, w in loads.items():
            row = self.loads[name]
            heat[row.node] = heat.get(row.node, 0.0) + w * row.heat_fraction
        heat[BATTERY_NODE] = heat.get(BATTERY_NODE, 0.0) + pack_loss
        heat[CONVERTER_NODE] = heat.get(CONVERTER_NODE, 0.0) + array_loss + load_loss

        step = EpsStep(
            p_string_W=p_str, p_gen_W=p_gen, loads_W=loads, demand_W=demand, p_acc_W=p_acc,
            p_sup_W=p_sup, p_pack_W=p_pack, curtailed_W=curtailed, unmet_W=n_neg - p_sup,
            E_J=e, E_next_J=e_next, soc=soc, ocv_V=ocv, v_V=v, i_A=i, disc_clamped=clamped,
            shed=state.uv_hold, uv_hold=state.uv_hold, inhibit=state.inhibit,
            battery_heater=state.battery_heater, payload_heater=state.payload_heater,
            heat_W=heat, pack_loss_W=pack_loss, array_conv_loss_W=array_loss,
            load_conv_loss_W=load_loss,
        )
        new = EpsState(e_next, uv, inhibit, heaters["battery_heater"], heaters["payload_heater"])
        return step, new
