"""Typed parameters loaded from params/*.yaml, and the ParamView that model code reads through.

The YAML files transcribe the EXPAND.3 Phase I spec tables (Parameters, Signals, Faults).
Model code never holds numeric literals: every number comes from these files via ParamView.
"""

from __future__ import annotations

import copy
import itertools
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

SatId = str
BusId = Literal["A", "B"]
Tier = Literal["survival", "attitude", "comms", "payload"]
Vec3 = tuple[float, float, float]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- YAML loading


class _Loader(yaml.SafeLoader):
    """SafeLoader with the YAML 1.2 float rule, so `3.6e6` (no exponent sign) is a float."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.VERBOSE,
    ),
    list("-+0123456789."),
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.load(f, Loader=_Loader)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge `over` onto `base`. Mappings merge; lists and scalars replace."""
    out = copy.deepcopy(dict(base))
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


# --------------------------------------------------------------------------- scenario.yaml


class Draw(_Model):
    """A per-seed random draw. `year` alone means uniform over that calendar year."""

    draw: Literal["uniform", "loguniform"]
    low: float | None = None
    high: float | None = None
    year: int | None = None
    per: str | None = None

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        if self.year is None and (self.low is None or self.high is None):
            raise ValueError("draw needs either year or both low and high")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError("draw low > high")
        return self


class Orbit(_Model):
    a_km: float
    e: float
    i_deg: float
    ltan_hours: float
    arg_perigee_deg: float


class GroundStation(_Model):
    name: str
    lat_deg: float
    lon_deg: float


class Nominal(_Model):
    nominal: float
    note: str | None = None


class TakeConstraints(_Model):
    max_off_nadir_deg: float
    min_sun_elevation_at_target_deg: float


class Constants(_Model):
    mu_km3_s2: float
    J2: float
    R_E_km: float
    S_W_m2: float
    a_alb: float
    q_IR_W_m2: float
    sigma_W_m2_K4: float
    g0_m_s2: float
    omega_E_rad_s: float
    R_sun_km: float


class DragEnv(_Model):
    h0_km: float
    rho0_kg_m3: Draw
    H_km: Draw


class Slew(_Model):
    step_s: float
    torque_fraction: float = Field(gt=0, le=1)
    momentum_fraction: float = Field(gt=0, le=1)
    pd_natural_freq_rad_s: float
    pd_damping: float
    tol_angle_deg: float
    tol_rate_rad_s: float
    max_settle_s: float


class Scenario(_Model):
    orbit: Orbit
    true_anomaly_at_epoch_deg: dict[SatId, float]
    bus_of: dict[SatId, BusId]
    clusters: list[list[SatId]]
    burn_vehicles: list[SatId]
    take_vehicles: list[SatId]
    epoch: Draw
    run_length_h: Draw
    integration_step_s: float
    control_tick_s: float
    ground_stations: list[GroundStation]
    elevation_mask_deg: float
    burn_command_time_h: Draw
    station_keeping_burn_length_h: Nominal
    station_keeping_burn_length_draw_h: Draw
    station_keeping_dv_m_s: float
    take_duration_s: float
    eps_initial_soc: float = Field(ge=0, le=1)
    thermal_init: Literal["settled", "fixed"] = "settled"
    thermal_init_orbits: int = Field(ge=0)
    thermal_initial_K: float
    placeholder_node_temperature_K: dict[str, float]
    slew: Slew
    take_window_start_h: float
    take_constraints: TakeConstraints
    fault_lead_s: Draw
    constants: Constants
    drag: DragEnv

    @model_validator(mode="after")
    def _sat_ids_consistent(self) -> Self:
        sats = set(self.bus_of)
        if set(self.true_anomaly_at_epoch_deg) != sats:
            raise ValueError("true_anomaly_at_epoch_deg and bus_of name different vehicles")
        named = {s for c in self.clusters for s in c} | set(self.burn_vehicles)
        named |= set(self.take_vehicles)
        if unknown := named - sats:
            raise ValueError(f"unknown vehicles {sorted(unknown)}")
        return self


# ------------------------------------------------------------------- bus_a.yaml / bus_b.yaml


class Mass(_Model):
    wet_mass_kg: float
    inertia_diag_kg_m2: Vec3
    airframe_box_m: Vec3


class Drag(_Model):
    Cd: float
    area_m2: float


class Wings(_Model):
    count: int
    names: list[str]
    sunward_area_m2: float
    hinge: str
    normal: str

    @model_validator(mode="after")
    def _count(self) -> Self:
        if len(self.names) != self.count:
            raise ValueError("wings.count does not match wings.names")
        return self


class Thruster(_Model):
    face: str
    axis_b: Vec3
    F_max_N: float
    Isp_s: float
    propellant_kg: float
    eta_F: float


class Wheels(_Model):
    count: int
    spin_axis_angle_off_z_deg: float
    tau_max_Nm: float
    h_max_Nms: float
    power_while_torquing_W: float
    k: list[float]  # torque constant, dimensionless
    c: list[float]  # bearing friction, 1/s
    azimuth_deg: list[float]

    @model_validator(mode="after")
    def _count(self) -> Self:
        if not len(self.k) == len(self.c) == len(self.azimuth_deg) == self.count:
            raise ValueError("wheels.k, c and azimuth_deg lengths must equal wheels.count")
        return self


class StarTracker(_Model):
    face: str
    tilt_below_level_deg: float
    sun_exclusion_deg: float
    earth_exclusion_deg: float
    noise_cross_arcsec: float
    noise_roll_arcsec: float


class Imager(_Model):
    pixels: int
    fov_deg: float
    face: str


class BusCrosslink(_Model):
    band: str
    range_km: float
    rate_Mbit_s: float


class OcvCurve(_Model):
    soc: list[float]
    volts: list[float]

    @model_validator(mode="after")
    def _table(self) -> Self:
        _check_table(self.soc, self.volts)
        return self


class Derate(_Model):
    T_K: list[float]
    factor: list[float]

    @model_validator(mode="after")
    def _table(self) -> Self:
        _check_table(self.T_K, self.factor)
        return self


def _check_table(x: list[float], y: list[float]) -> None:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("table columns must have equal length >= 2")
    if any(b <= a for a, b in itertools.pairwise(x)):
        raise ValueError("table abscissa must be strictly increasing")


class Undervoltage(_Model):
    shed_V: float
    release_V: float


class ColdChargeInhibit(_Model):
    inhibit_K: float
    release_K: float


class Heater(_Model):
    power_W: float
    on_K: float
    off_K: float


class Eps(_Model):
    strings: int
    string_area_m2: float
    string_health: list[Annotated[float, Field(ge=0, le=1)]]
    packing: float
    eta0: float
    T0_K: float
    deta_dT_per_K: float
    eta_ab: float
    eta_bl: float
    battery: str
    E_cap_J: float
    R_pack_ohm: float
    eta_c: float
    eta_d: float
    P_c_max_W: float
    ocv_curve: OcvCurve
    usable_derate: Derate
    undervoltage: Undervoltage
    cold_charge_inhibit: ColdChargeInhibit
    battery_heater: Heater
    payload_heater: Heater
    heater_cutoff_true_K: float

    @model_validator(mode="after")
    def _strings(self) -> Self:
        if len(self.string_health) != self.strings:
            raise ValueError("eps.string_health length must equal eps.strings")
        return self


class Load(_Model):
    """One row of the Loads table. Which draw fields apply depends on the load's mode logic."""

    tier: Tier
    heat_fraction: float = Field(ge=0, le=1)
    node: str
    draw_W: float | None = None
    take_W: float | None = None
    on_W: float | None = None
    pass_no_take_W: float | None = None
    else_W: float | None = None
    off_in_saa_and_reboot_W: float | None = None
    plus: str | None = None


class ThermalNode(_Model):
    C_J_K: float = Field(gt=0)
    area_m2: float | None = None
    sunward_m2: float | None = None
    alpha: float | None = None
    epsilon: float | None = None
    normal: str | None = None
    view: float | None = None
    G_to_core_W_K: float | None = None
    G_to_each_face_W_K: float | None = None
    note: str | None = None


class ThermalLinks(_Model):
    face_to_face_around_sides_W_K: float
    bus_pz_to_bus_px_W_K: float
    bus_nz_to_bus_nx_W_K: float
    wing_to_its_face_hinge_W_K: float


class Thermal(_Model):
    nodes: dict[str, ThermalNode]
    links: ThermalLinks
    limits_K: dict[str, tuple[float, float]]
    stability_limit_s: float


class Bus(_Model):
    mass: Mass
    drag: Drag
    wings: Wings
    thruster: Thruster
    wheels: Wheels
    star_tracker: StarTracker
    imager: Imager
    crosslink: BusCrosslink
    eps: Eps
    loads: dict[str, Load]
    undervoltage_sheds_tiers: list[Tier]
    thermal: Thermal

    @model_validator(mode="after")
    def _cross_refs(self) -> Self:
        nodes = self.thermal.nodes
        for name, load in self.loads.items():
            if load.node not in nodes:
                raise ValueError(f"load {name} heats unknown node {load.node}")
        wing_nodes = {n for n in nodes if n.startswith(_WING_PREFIX)}
        if wing_nodes != set(self.wings.names):
            raise ValueError("thermal wing nodes do not match wings.names")
        if self.eps.strings != self.wings.count:
            raise ValueError("eps.strings does not match wings.count")
        return self


class BusBThermalOverrides(_Model):
    """The Bus B thermal block, which is expressed as edits to Bus A's node table."""

    face_area_m2: dict[Literal["x", "y", "z"], float]
    face_C_J_K: float
    wing_area_m2: float
    wing_sunward_m2: float
    wing_C_J_K: float
    battery_C_J_K: float
    core_C_J_K: float
    remove_nodes: list[str]
    stability_limit_s: float


_WING_PREFIX = "wing_"
_FACE_NODES: dict[str, tuple[str, str]] = {
    "x": ("bus_px", "bus_nx"),
    "y": ("bus_py", "bus_ny"),
    "z": ("bus_pz", "bus_nz"),
}
_BATTERY_NODE = "battery"
_CORE_NODE = "bus_core"


def _apply_bus_b_thermal(thermal_a: Mapping[str, Any], o: BusBThermalOverrides) -> dict[str, Any]:
    thermal = copy.deepcopy(dict(thermal_a))
    nodes: dict[str, dict[str, Any]] = thermal["nodes"]
    for name in o.remove_nodes:
        if name not in nodes:
            raise ValueError(f"bus_b removes unknown thermal node {name}")
        del nodes[name]
    for axis, area in o.face_area_m2.items():
        for name in _FACE_NODES[axis]:
            nodes[name]["area_m2"] = area
            nodes[name]["C_J_K"] = o.face_C_J_K
    for name, node in nodes.items():
        if name.startswith(_WING_PREFIX):
            node["area_m2"] = o.wing_area_m2
            node["sunward_m2"] = o.wing_sunward_m2
            node["C_J_K"] = o.wing_C_J_K
    nodes[_BATTERY_NODE]["C_J_K"] = o.battery_C_J_K
    nodes[_CORE_NODE]["C_J_K"] = o.core_C_J_K
    thermal["stability_limit_s"] = o.stability_limit_s
    return thermal


# --------------------------------------------------------------------------- faults.yaml


class Fault(_Model):
    """theta(t) = theta0 * (1 + (m - 1) * s(t)), or theta = set_to when the nominal is zero."""

    name: str
    vehicle: SatId
    param: str | None = None
    params: list[str] | None = None
    m: float | None = None
    set_to: float | None = None
    profile: Literal["step", "ramp"]
    t_r_s: float | None = None
    cap: float | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if (self.param is None) == (self.params is None):
            raise ValueError("fault needs exactly one of param / params")
        if (self.m is None) == (self.set_to is None):
            raise ValueError("fault needs exactly one of m / set_to")
        if (self.profile == "ramp") != (self.t_r_s is not None):
            raise ValueError("t_r_s is required for, and only for, a ramp profile")
        return self

    @property
    def paths(self) -> list[str]:
        return [self.param] if self.param is not None else list(self.params or [])


class Onset(_Model):
    t0: str
    arm: str
    duration: str


FaultId = Annotated[str, Field(pattern=r"^F([1-9]|10)$")]


class Faults(_Model):
    faults: dict[FaultId, Fault]
    onset: Onset


# --------------------------------------------------------------------------- links_saa_imager.yaml


class Crosslink(_Model):
    range_km: float
    cadence: str
    rx_noise_pos_m: float
    rx_noise_vel_m_s: float
    drop_probability: float = Field(ge=0, le=1)


class Saa(_Model):
    boundary_nT: float
    field_model: str


class CdhUpsets(_Model):
    lambda_out_per_s: float
    lambda_in_per_s: float
    reset_fraction: float


class DarkCurrent(_Model):
    D0_e_s_px: float
    T_ref_K: float
    delta_T_K: float


class Signal(_Model):
    model_config = ConfigDict(frozen=True, extra="allow")
    rate_s: float


class LinksSaaImager(_Model):
    crosslink: Crosslink
    saa: Saa
    edge_computer_reboot_s: float
    cdh_upsets: CdhUpsets
    imager_dark_current: DarkCurrent
    signals: dict[str, Signal]


# --------------------------------------------------------------------------- Params


class Params(_Model):
    scenario: Scenario
    buses: dict[BusId, Bus]
    faults: Faults
    links: LinksSaaImager

    @model_validator(mode="after")
    def _refs(self) -> Self:
        if missing := set(self.scenario.bus_of.values()) - set(self.buses):
            raise ValueError(f"scenario uses undefined buses {sorted(missing)}")
        for fid, fault in self.faults.faults.items():
            if fault.vehicle not in self.scenario.bus_of:
                raise ValueError(f"{fid} targets unknown vehicle {fault.vehicle}")
        return self

    def bus(self, sat_id: SatId) -> Bus:
        return self.buses[self.scenario.bus_of[sat_id]]

    @property
    def sat_ids(self) -> list[SatId]:
        return list(self.scenario.bus_of)


def load_params(directory: str | Path) -> Params:
    """Load params/*.yaml. Bus B is bus_b.yaml deep-merged onto bus_a.yaml."""
    d = Path(directory)
    bus_a = _read_yaml(d / "bus_a.yaml")
    bus_b_raw = _read_yaml(d / "bus_b.yaml")
    b_thermal = BusBThermalOverrides.model_validate(bus_b_raw.pop("thermal"))
    bus_b = deep_merge(bus_a, bus_b_raw)
    bus_b["thermal"] = _apply_bus_b_thermal(bus_a["thermal"], b_thermal)
    faults_raw = _read_yaml(d / "faults.yaml")
    onset = faults_raw.pop("onset")
    return Params.model_validate(
        {
            "scenario": _read_yaml(d / "scenario.yaml"),
            "buses": {"A": bus_a, "B": bus_b},
            "faults": {"faults": faults_raw, "onset": onset},
            "links": _read_yaml(d / "links_saa_imager.yaml"),
        }
    )


# --------------------------------------------------------------------------- ParamView

_SEGMENT = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)((?:\[\d+\])*)")


class ParamView:
    """Read-only, per-vehicle view over Params that model code reads every number through.

    Paths are the dotted paths used in faults.yaml, e.g. `eps.R_pack_ohm`, `wheels.k[1]`,
    `thermal.nodes.bus_px.alpha`, `cdh_upsets.lambda_in_per_s`. They resolve against the
    vehicle's bus merged with links_saa_imager.yaml; `scenario.` paths resolve against
    scenario.yaml. `overrides` maps a path to a replacement value; the fault injector fills it
    in M7. Overrides win over the loaded value.
    """

    def __init__(self, params: Params, sat_id: SatId) -> None:
        self._params = params
        self._sat_id = sat_id
        self.overrides: dict[str, float] = {}
        tree = deep_merge(params.links.model_dump(), params.bus(sat_id).model_dump())
        tree["scenario"] = params.scenario.model_dump()
        self._tree: dict[str, Any] = tree

    @property
    def sat_id(self) -> SatId:
        return self._sat_id

    @property
    def params(self) -> Params:
        """Typed access to loaded values. Does not see overrides; prefer get()/array()."""
        return self._params

    def _lookup(self, path: str) -> Any:
        node: Any = self._tree
        for part in path.split("."):
            m = _SEGMENT.fullmatch(part)
            if m is None:
                raise KeyError(f"bad parameter path {path!r}")
            node = node[m.group(1)]
            for idx in re.findall(r"\[(\d+)\]", m.group(2)):
                node = node[int(idx)]
        return node

    def has(self, path: str) -> bool:
        if path in self.overrides:
            return True
        try:
            self._lookup(path)
        except (KeyError, IndexError, TypeError):
            return False
        return True

    def get(self, path: str) -> float:
        if path in self.overrides:
            return self.overrides[path]
        value = self._lookup(path)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{path!r} is not a scalar number")
        return float(value)

    def array(self, path: str) -> NDArray[np.float64]:
        """A numeric list as a float array, with element overrides (`path[i]`) applied."""
        out = np.array(self._lookup(path), dtype=np.float64)
        for i in range(out.shape[0]):
            key = f"{path}[{i}]"
            if key in self.overrides:
                out[i] = self.overrides[key]
        out.flags.writeable = False
        return out
