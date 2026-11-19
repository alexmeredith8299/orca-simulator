"""Run recorder: truth and telemetry channels written to Parquet (M8).

Spec: Signals ("The simulator outputs these measurements. Log the truth beside them.").

Layout (CLAUDE.md Known gaps): one directory per seed, long format with `sat_id` and `t_s` on
every row, rows ordered by vehicle then time, units in the column-name suffix and in the schema
metadata (`units`, a JSON map from column to unit).

- `truth.parquet`: every vehicle at every 30 s tick.
- `measurements.parquet`: every Signals-table signal on the same tick grid, `<name>_meas` beside
  `<name>_true`, NaN where a signal is not sampled or not available, with validity columns.
- `wheel_steps.parquet`: wheel momentum, commanded torque, body torque and power per 1 s slew step.
- `burns.parquet`: the burn record at each tick a burn is active.
- `burn_summary.parquet`: one row per burn vehicle (command, uplink, start, end, suspensions).
- `takes.parquet`: one row per take, including dark current.
- `crosslink.parquet`: one row per tick per ordered pair.
- `labels.json`: the fault label, or {"fault": "none"}.

The data files hold no timing or host information, so the same seed and configuration give
byte-identical files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from expand3.mission.thread import MissionRun
from expand3.params import Params, ParamView

DATA_FILES = (
    "truth.parquet", "measurements.parquet", "wheel_steps.parquet", "burns.parquet",
    "burn_summary.parquet", "takes.parquet", "crosslink.parquet", "labels.json",
)

_N_WHEELS = 4
_N_STRINGS = 4
_AXES = ("x", "y", "z")
_QUAT = ("w", "x", "y", "z")
_UNIT_SUFFIXES = (
    ("_km_s", "km/s"), ("_m_s2", "m/s^2"), ("_rad_s", "rad/s"), ("_m_s", "m/s"), ("_Nms", "N m s"),
    ("_Nm", "N m"), ("_km", "km"), ("_kg", "kg"), ("_nT", "nT"), ("_J", "J"), ("_V", "V"),
    ("_A", "A"), ("_W", "W"), ("_K", "K"), ("_s", "s"), ("_deg", "deg"), ("_e_s_px", "e-/s/px"),
)


def _unit(name: str) -> str:
    base = re.sub(r"_(meas|true|cmd|delivered)$", "", name)
    for suffix, unit in _UNIT_SUFFIXES:
        if base.endswith(suffix):
            return unit
    return "1"


def _write(path: Path, columns: dict[str, Any]) -> Path:
    arrays = {}
    for name, values in columns.items():
        arr = np.asarray(values)
        if arr.dtype.kind in "US" or arr.dtype == object:
            arrays[name] = pa.array([str(v) for v in arr], type=pa.string())
        else:
            arrays[name] = pa.array(arr)
    table = pa.table(arrays)
    units = {name: _unit(name) for name in columns}
    table = table.replace_schema_metadata({"units": json.dumps(units, sort_keys=True)})
    pq.write_table(table, path, compression="zstd", write_statistics=True)
    return path


_STRING_COLUMNS = {"sat_id", "bus", "sender", "receiver"}
_INT_COLUMNS = {"slew", "upsets", "resets", "receipts"}


def _stack(parts: list[NDArray[Any]], name: str = "") -> NDArray[Any]:
    """Concatenate parts; an empty column keeps its fixed type so schemas never vary."""
    if parts:
        return np.concatenate(parts)
    if name in _STRING_COLUMNS:
        return np.array([], dtype=str)
    if name in _INT_COLUMNS:
        return np.array([], dtype=np.int64)
    return np.array([], dtype=np.float64)


def _tick_index(t: NDArray[np.float64], sample_t: NDArray[np.float64]) -> NDArray[np.int64]:
    idx = np.searchsorted(t, sample_t)
    assert np.allclose(t[idx], sample_t)
    return idx.astype(np.int64)


def _on_grid(n: int, idx: NDArray[np.int64], values: NDArray[Any], fill: float = np.nan
             ) -> NDArray[np.float64]:
    out = np.full((n, *np.shape(values)[1:]), fill, dtype=np.float64)
    out[idx] = values
    return out


def _all_nodes(params: Params) -> list[str]:
    return list(params.buses["A"].thermal.nodes)


def truth_columns(run: MissionRun, params: Params) -> dict[str, Any]:
    """Per vehicle per tick truth (vehicle-major rows)."""
    n_t = len(run.t_s)
    nodes = _all_nodes(params)
    cols: dict[str, list[NDArray[Any]]] = {}

    def add(name: str, values: NDArray[Any]) -> None:
        cols.setdefault(name, []).append(np.asarray(values))

    for k, sat in enumerate(run.sats):
        eps, th, saa = run.eps[sat], run.thermal[sat], run.saa[sat]
        add("sat_id", np.full(n_t, sat))
        add("bus", np.full(n_t, params.scenario.bus_of[sat]))
        add("t_s", run.t_s)
        for i, a in enumerate(_AXES):
            add(f"r_{a}_km", run.r_km[:, k, i])
        for i, a in enumerate(_AXES):
            add(f"v_{a}_km_s", run.v_km_s[:, k, i])
        for i, a in enumerate(_QUAT):
            add(f"q_{a}", run.q[:, k, i])
        for i, a in enumerate(_AXES):
            add(f"omega_{a}_rad_s", run.omega[:, k, i])
        for i in range(_N_WHEELS):
            add(f"h_wheel_{i + 1}_Nms", run.h_wheel[:, k, i])
        add("mass_kg", run.mass_kg[:, k])
        add("propellant_kg", run.propellant_kg[:, k])
        for i, a in enumerate(_AXES):
            add(f"a_cmd_{a}_m_s2", run.a_cmd_m_s2[:, k, i])
        add("dv_m_s", run.dv_m_s[:, k])
        add("shadow", run.shadow[:, k])
        add("E_J", eps.E_J)
        add("soc", eps.soc)
        add("v_V", eps.v_V)
        add("i_A", eps.i_A)
        add("p_pack_W", eps.p_pack_W)
        add("p_acc_W", eps.p_acc_W)
        add("p_sup_W", eps.p_sup_W)
        add("p_gen_W", eps.p_gen_W)
        for i in range(_N_STRINGS):
            add(f"p_string_{i + 1}_W", eps.p_string_W[:, i] if i < eps.p_string_W.shape[1]
                else np.full(n_t, np.nan))
        add("demand_W", eps.demand_W)
        add("curtailed_W", eps.curtailed_W)
        add("unmet_W", eps.unmet_W)
        for flag in ("shed", "uv_hold", "inhibit", "battery_heater", "payload_heater"):
            add(flag, getattr(eps, flag))
        for node in nodes:
            add(f"T_{node}_K", th.node(node) if node in th.names else np.full(n_t, np.nan))
        add("gate_pass", run.gate_pass[:, k].any(axis=1))
        add("gate_take", run.gate_take[:, k])
        add("gate_slew", run.gate_slew[:, k])
        add("gate_burn", run.gate_burn[:, k])
        add("in_saa", saa.in_saa)
        add("B_nT", saa.B_nT)
        add("edge_off", saa.edge_off)
        add("upsets", saa.upsets)
        add("resets", saa.resets)
        lab = run.fault
        if lab is not None and lab.vehicle == sat:
            add("fault_s", lab.s)
            thetas = [lab.theta[p] for p in lab.paths]
            add("fault_theta_0", thetas[0])
            add("fault_theta_1", thetas[1] if len(thetas) > 1 else np.full(n_t, np.nan))
        else:
            add("fault_s", np.zeros(n_t))
            add("fault_theta_0", np.full(n_t, np.nan))
            add("fault_theta_1", np.full(n_t, np.nan))
    return {name: _stack(parts, name) for name, parts in cols.items()}


def measurement_columns(run: MissionRun, params: Params) -> dict[str, Any]:
    """Every Signals-table signal on the tick grid, measurement beside truth."""
    t = run.t_s
    n_t = len(t)
    nodes = _all_nodes(params)
    cols: dict[str, list[NDArray[Any]]] = {}

    def add(name: str, values: NDArray[Any]) -> None:
        cols.setdefault(name, []).append(np.asarray(values))

    def pair(name: str, idx: NDArray[np.int64], meas: NDArray[Any], true: NDArray[Any]) -> None:
        add(f"{name}_meas", _on_grid(n_t, idx, meas))
        add(f"{name}_true", _on_grid(n_t, idx, true))

    for sat in run.sats:
        sig, saa = run.signals[sat], run.saa[sat]
        add("sat_id", np.full(n_t, sat))
        add("t_s", t)
        gi = _tick_index(t, sig.gnss_r.t_s)
        add("gnss_valid", np.isin(np.arange(n_t), gi[np.isfinite(sig.gnss_r.meas[:, 0])]))
        for i, a in enumerate(_AXES):
            pair(f"gnss_r_{a}_km", gi, sig.gnss_r.meas[:, i], sig.gnss_r.truth[:, i])
        for i, a in enumerate(_AXES):
            pair(f"gnss_v_{a}_km_s", gi, sig.gnss_v.meas[:, i], sig.gnss_v.truth[:, i])
        for i, a in enumerate(_AXES):
            pair(f"nav_r_{a}_km", gi, sig.nav_r.meas[:, i], sig.nav_r.truth[:, i])
        for i, a in enumerate(_AXES):
            pair(f"nav_v_{a}_km_s", gi, sig.nav_v.meas[:, i], sig.nav_v.truth[:, i])
        mi = _tick_index(t, sig.mag.t_s)
        for i, a in enumerate(_AXES):
            pair(f"mag_{a}_nT", mi, sig.mag.meas[:, i], sig.mag.truth[:, i])
        for i, a in enumerate(_AXES):
            pair(f"mag_model_{a}_nT", mi, sig.mag_model.meas[:, i], sig.mag_model.truth[:, i])
        pair("mag_model_abs_nT", mi, sig.mag_model_abs.meas, sig.mag_model_abs.truth)
        st = sig.star_tracker
        si = _tick_index(t, st.t_s)
        avail = np.zeros(n_t, dtype=bool)
        avail[si] = st.available
        add("st_available", avail)
        for i, a in enumerate(_QUAT):
            pair(f"st_q_{a}", si, st.q_meas[:, i], st.q_true[:, i])
        ii = _tick_index(t, sig.imu.t_s)
        for i, a in enumerate(_AXES):
            pair(f"imu_omega_{a}_rad_s", ii, sig.imu.meas[:, i], sig.imu.truth[:, i])
        vi = _tick_index(t, sig.pack_voltage.t_s)
        pair("pack_v_V", vi, sig.pack_voltage.meas, sig.pack_voltage.truth)
        ai = _tick_index(t, sig.pack_current.t_s)
        pair("pack_i_A", ai, sig.pack_current.meas, sig.pack_current.truth)
        pi = _tick_index(t, sig.string_power.t_s)
        for i in range(_N_STRINGS):
            if i < sig.string_power.meas.shape[1]:
                pair(f"string_p_{i + 1}_W", pi, sig.string_power.meas[:, i],
                     sig.string_power.truth[:, i])
            else:
                pair(f"string_p_{i + 1}_W", pi, np.full(len(pi), np.nan),
                     np.full(len(pi), np.nan))
        ti = _tick_index(t, sig.node_temperature.t_s)
        for node in nodes:
            if node in sig.node_names:
                j = sig.node_names.index(node)
                pair(f"T_{node}_K", ti, sig.node_temperature.meas[:, j],
                     sig.node_temperature.truth[:, j])
            else:
                pair(f"T_{node}_K", ti, np.full(len(ti), np.nan), np.full(len(ti), np.nan))
        oi = _tick_index(t, sig.soc_estimate.t_s)
        pair("soc_est", oi, sig.soc_estimate.meas, sig.soc_estimate.truth)
        ki = _tick_index(t, sig.tank_gauge.t_s)
        add("tank_valid", np.isin(np.arange(n_t), ki))
        pair("tank_kg", ki, sig.tank_gauge.meas, sig.tank_gauge.truth)
        for heater in ("battery", "payload"):
            add(f"heater_{heater}_cmd", sig.heater_commanded[heater])
            add(f"heater_{heater}_delivered", sig.heater_delivered[heater])
        for gate, values in sig.gates.items():
            add(f"gate_{gate}", values)
        add("field_nT", saa.B_nT)
        add("upsets", saa.upsets)
        add("resets", saa.resets)
    return {name: _stack(parts, name) for name, parts in cols.items()}


def wheel_step_columns(run: MissionRun) -> dict[str, Any]:
    cols: dict[str, list[NDArray[Any]]] = {}

    def add(name: str, values: NDArray[Any]) -> None:
        cols.setdefault(name, []).append(np.asarray(values))

    for sat in run.sats:
        rec = run.takes.get(sat)
        for idx, res in enumerate(run.slews.get(sat, [])):
            assert rec is not None
            t0 = rec.slew_in_start_s if idx == 0 else rec.t_end_s
            n = len(res.t_s)
            add("sat_id", np.full(n, sat))
            add("slew", np.full(n, idx, dtype=np.int64))
            add("t_s", t0 + res.t_s)
            for i in range(_N_WHEELS):
                add(f"h_{i + 1}_Nms", res.h[:, i])
            for i in range(_N_WHEELS):
                add(f"u_{i + 1}_Nm", res.u[:, i])
            for i, a in enumerate(_AXES):
                add(f"tau_{a}_Nm", res.tau_w[:, i])
            add("power_W", res.power_W)
    names = ["sat_id", "slew", "t_s", *[f"h_{i + 1}_Nms" for i in range(_N_WHEELS)],
             *[f"u_{i + 1}_Nm" for i in range(_N_WHEELS)], *[f"tau_{a}_Nm" for a in _AXES],
             "power_W"]
    return {name: _stack(cols.get(name, []), name) for name in names}


def burn_columns(run: MissionRun) -> dict[str, Any]:
    cols: dict[str, list[NDArray[Any]]] = {}
    for k, sat in enumerate(run.sats):
        on = run.gate_burn[:, k]
        if not on.any():
            continue
        sig = run.signals[sat]
        gauge = _on_grid(len(run.t_s), _tick_index(run.t_s, sig.tank_gauge.t_s),
                         sig.tank_gauge.meas)
        n = int(on.sum())
        parts = {
            "sat_id": np.full(n, sat),
            "t_s": run.t_s[on],
            "a_cmd_x_m_s2": run.a_cmd_m_s2[on, k, 0],
            "a_cmd_y_m_s2": run.a_cmd_m_s2[on, k, 1],
            "a_cmd_z_m_s2": run.a_cmd_m_s2[on, k, 2],
            "dv_true_m_s": run.dv_m_s[on, k],
            "propellant_used_true_kg": run.propellant_kg[0, k] - run.propellant_kg[on, k],
            "tank_gauge_kg_meas": gauge[on],
            "tank_gauge_kg_true": run.propellant_kg[on, k],
        }
        for name, values in parts.items():
            cols.setdefault(name, []).append(values)
    names = ["sat_id", "t_s", "a_cmd_x_m_s2", "a_cmd_y_m_s2", "a_cmd_z_m_s2", "dv_true_m_s",
             "propellant_used_true_kg", "tank_gauge_kg_meas", "tank_gauge_kg_true"]
    return {name: _stack(cols.get(name, []), name) for name in names}


def _num(x: float | None) -> float:
    return float("nan") if x is None else float(x)


def burn_summary_columns(run: MissionRun) -> dict[str, Any]:
    recs = [run.burns[s] for s in sorted(run.burns)]
    return {
        "sat_id": np.array([r.sat for r in recs]),
        "command_time_s": np.array([r.command_time_s for r in recs]),
        "uplink_start_s": np.array([_num(r.uplink_pass[0] if r.uplink_pass else None)
                                    for r in recs]),
        "uplink_end_s": np.array([_num(r.uplink_pass[1] if r.uplink_pass else None)
                                  for r in recs]),
        "burn_start_s": np.array([_num(r.burn_start_s) for r in recs]),
        "burn_end_s": np.array([_num(r.burn_end_s) for r in recs]),
        "length_s": np.array([r.length_s for r in recs]),
        "flown_s": np.array([r.flown_s for r in recs]),
        "suspended_s": np.array([sum(b - a for a, b in r.suspended) for r in recs]),
        "accel_m_s2": np.array([r.accel_m_s2 for r in recs]),
        "dv_true_m_s": np.array([r.dv_m_s for r in recs]),
        "propellant_used_true_kg": np.array([r.propellant_used_kg for r in recs]),
    }


def take_columns(run: MissionRun) -> dict[str, Any]:
    recs = [run.takes[s] for s in sorted(run.takes)]
    return {
        "sat_id": np.array([r.sat for r in recs], dtype=str),
        "t_take_s": np.array([r.t_take_s for r in recs], dtype=float),
        "t_end_s": np.array([r.t_end_s for r in recs], dtype=float),
        "target_lat_deg": np.array([r.target_lat_deg for r in recs], dtype=float),
        "target_lon_deg": np.array([r.target_lon_deg for r in recs], dtype=float),
        "slew_in_start_s": np.array([r.slew_in_start_s for r in recs], dtype=float),
        "slew_out_end_s": np.array([r.slew_out_end_s for r in recs], dtype=float),
        "min_sun_elevation_deg": np.array([min(r.sun_elevation_deg) for r in recs], dtype=float),
        "max_off_nadir_deg": np.array([max(r.off_nadir_deg) for r in recs], dtype=float),
        "payload_K": np.array([r.payload_K for r in recs], dtype=float),
        "dark_current_e_s_px": np.array([r.dark_current_e_s_px for r in recs], dtype=float),
    }


def crosslink_columns(run: MissionRun) -> dict[str, Any]:
    log = run.crosslink
    assert log is not None
    n_t = len(log.t_s)
    cols: dict[str, list[NDArray[Any]]] = {}
    for j, (a, b) in enumerate(log.pairs):
        parts = {
            "t_s": log.t_s,
            "sender": np.full(n_t, a),
            "receiver": np.full(n_t, b),
            "sent": log.sent[:, j],
            "delivered": log.delivered[:, j],
            "r_x_km_meas": log.r_meas_km[:, j, 0],
            "r_y_km_meas": log.r_meas_km[:, j, 1],
            "r_z_km_meas": log.r_meas_km[:, j, 2],
            "v_x_km_s_meas": log.v_meas_km_s[:, j, 0],
            "v_y_km_s_meas": log.v_meas_km_s[:, j, 1],
            "v_z_km_s_meas": log.v_meas_km_s[:, j, 2],
            "rssi_dBm": log.rssi_dBm[:, j],
            "age_s": log.age_s[(b, a)],
            "receipts": log.receipts[b],
        }
        for name, values in parts.items():
            cols.setdefault(name, []).append(np.asarray(values))
    return {name: _stack(parts, name) for name, parts in cols.items()}


def labels(run: MissionRun, params: Params) -> dict[str, Any]:
    lab = run.fault
    if lab is None:
        return {"fault": "none"}
    view = ParamView(params, lab.vehicle)
    return {
        "fault": lab.fault_id,
        "fault_id": lab.fault_id,
        "name": lab.name,
        "vehicle": lab.vehicle,
        "paths": lab.paths,
        "theta0": [view.base(p) for p in lab.paths],
        "magnitude": lab.magnitude,
        "kind": lab.kind,
        "profile": lab.profile,
        "t0_s": lab.t0_s,
        "t_arm_s": lab.t_arm_s,
        "lead_s": lab.lead_s,
        "truth_columns": {"s": "fault_s",
                          **{p: f"fault_theta_{i}" for i, p in enumerate(lab.paths)}},
    }


def write_run(run: MissionRun, params: Params, out_dir: Path) -> list[Path]:
    """Write every data file of one run into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [
        _write(out_dir / "truth.parquet", truth_columns(run, params)),
        _write(out_dir / "measurements.parquet", measurement_columns(run, params)),
        _write(out_dir / "wheel_steps.parquet", wheel_step_columns(run)),
        _write(out_dir / "burns.parquet", burn_columns(run)),
        _write(out_dir / "burn_summary.parquet", burn_summary_columns(run)),
        _write(out_dir / "takes.parquet", take_columns(run)),
        _write(out_dir / "crosslink.parquet", crosslink_columns(run)),
    ]
    path = out_dir / "labels.json"
    path.write_text(json.dumps(labels(run, params), indent=2, sort_keys=True) + "\n")
    files.append(path)
    return files
