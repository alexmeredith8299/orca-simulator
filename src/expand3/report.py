"""Validation report from a batch directory: Markdown plus CSV tables (M8).

Sections: fault draws and leads, per-fault signature size (from the labels and the faulted
vehicle's measured channels, plus a paired delta against a no-fault baseline batch when given),
F5/F6 coverage, invariant checks per run, thermal-limit and undervoltage occurrences, runtime
statistics, and the known gaps from CLAUDE.md. No plotting dependency.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray

from expand3.params import Params, load_params
from expand3.run import MANIFEST

F64 = NDArray[np.float64]
_DEFAULT_PARAMS = Path(__file__).resolve().parents[2] / "params"
_CLAUDE_MD = Path(__file__).resolve().parents[2] / "CLAUDE.md"
_NAN_ALLOWED = {"p_string_4_W", "T_wing_ny_K", "fault_theta_0", "fault_theta_1"}
_E_TOL_J = 1e-3
_Q_TOL = 1e-9
_P_TOL = 1e-9


def _seed_dirs(batch: Path) -> list[Path]:
    return sorted(d for d in batch.glob("seed_*") if (d / MANIFEST).is_file())


def _cols(table: Any, sat: str) -> dict[str, F64]:
    sel = np.asarray(table.column("sat_id")) == sat
    return {name: np.asarray(table.column(name))[sel] for name in table.column_names
            if name not in ("sat_id", "bus")}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _invariants(params: Params, d: Path) -> list[dict[str, Any]]:
    truth = pq.read_table(d / "truth.parquet")
    burns = pq.read_table(d / "burn_summary.parquet")
    used = dict(zip(np.asarray(burns.column("sat_id")).tolist(),
                    np.asarray(burns.column("propellant_used_true_kg")).tolist(), strict=True))
    seed = int(d.name.split("_")[1])
    rows = []
    for sat in params.sat_ids:
        c = _cols(truth, sat)
        eps = params.bus(sat).eps
        dt = float(np.diff(c["t_s"])[0]) if len(c["t_s"]) > 1 else 0.0
        flow = eps.eta_c * c["p_acc_W"][:-1] - c["p_sup_W"][:-1] / eps.eta_d
        de = np.diff(c["E_J"]) - flow * dt
        prop_used = c["propellant_kg"][0] - c["propellant_kg"][-1]
        dry = c["mass_kg"] - c["propellant_kg"]
        qn = np.sqrt(c["q_w"] ** 2 + c["q_x"] ** 2 + c["q_y"] ** 2 + c["q_z"] ** 2)
        nan_cols = sorted(k for k, v in c.items() if v.dtype.kind == "f" and np.isnan(v).any()
                          and k not in _NAN_ALLOWED)
        row = {
            "seed": seed, "sat_id": sat,
            "energy_balance": bool(np.abs(de).max(initial=0.0) < _E_TOL_J),
            "energy_max_err_J": float(np.abs(de).max(initial=0.0)),
            "propellant_balance": bool(abs(prop_used - used.get(sat, 0.0)) < _P_TOL
                                       and np.ptp(dry) < _P_TOL),
            "quaternion_norm": bool(np.abs(qn - 1).max() < _Q_TOL),
            "soc_in_range": bool(np.all((c["soc"] >= 0) & (c["soc"] <= 1))),
            "no_nan": not nan_cols,
            "nan_columns": ";".join(nan_cols),
        }
        row["all_pass"] = all(row[k] for k in ("energy_balance", "propellant_balance",
                                               "quaternion_norm", "soc_in_range", "no_nan"))
        rows.append(row)
    return rows


def _limits(params: Params, d: Path) -> list[dict[str, Any]]:
    truth = pq.read_table(d / "truth.parquet")
    seed = int(d.name.split("_")[1])
    rows = []
    for sat in params.sat_ids:
        c = _cols(truth, sat)
        th = params.bus(sat).thermal
        out = np.zeros(len(c["t_s"]), dtype=bool)
        worst = []
        for node in th.nodes:
            lo, hi = th.limits_K.get(node, th.limits_K["others"])
            bad = (c[f"T_{node}_K"] < lo) | (c[f"T_{node}_K"] > hi)
            if bad.any():
                worst.append(node)
            out |= bad
        rows.append({"seed": seed, "sat_id": sat, "thermal_limit_ticks": int(out.sum()),
                     "nodes_out_of_limits": ";".join(worst),
                     "undervoltage_ticks": int(c["uv_hold"].sum())})
    return rows


def _signature(params: Params, d: Path, lab: dict[str, Any], baseline: Path | None
               ) -> dict[str, Any]:
    """One measured channel per fault, on the faulted vehicle after arm."""
    sat, fid, t_arm = lab["vehicle"], lab["fault_id"], lab["t_arm_s"]
    row: dict[str, Any] = {"seed": int(d.name.split("_")[1]), "fault": fid, "vehicle": sat,
                           "t_arm_h": t_arm / 3600, "lead_s": lab["lead_s"],
                           "label_magnitude": lab["magnitude"], "kind": lab["kind"]}
    truth = _cols(pq.read_table(d / "truth.parquet"), sat)
    after = truth["t_s"] >= t_arm
    row["armed_ticks"] = int(after.sum())
    if not after.any():
        row["measured"] = "not armed within the run"
        return row
    eps = params.bus(sat).eps
    if fid == "F1":
        dis = after & (truth["i_A"] < -0.5)
        ocv = np.interp(truth["soc"], eps.ocv_curve.soc, eps.ocv_curve.volts)
        r_eff = (ocv - truth["v_V"])[dis] / -truth["i_A"][dis]
        row["measured"] = f"R_eff/R0 max {np.max(r_eff) / eps.R_pack_ohm:.3f}" if dis.any() \
            else "no discharge after arm"
    elif fid == "F2":
        row["measured"] = f"string 1 max power after arm {truth['p_string_1_W'][after].max():.3g} W"
    elif fid in ("F3", "F4"):
        wh = pq.read_table(d / "wheel_steps.parquet")
        w = _cols(wh, sat) if len(wh) else {}
        if not w or not (w["t_s"] >= t_arm).any():
            row["measured"] = "no slew after arm"
        else:
            i = 0 if fid == "F3" else 1
            sel = w["t_s"] >= t_arm
            h, u = w[f"h_{i + 1}_Nms"][sel], w[f"u_{i + 1}_Nm"][sel]
            dh = np.diff(h)
            if fid == "F3":
                ok = np.abs(h[:-1]) > 1e-9
                c = -(dh + u[1:])[ok] / h[:-1][ok]
                row["measured"] = f"wheel 1 friction c {np.median(c):.4f} /s"
            else:
                ok = np.abs(u[1:]) > 1e-9
                row["measured"] = f"wheel 2 torque constant {np.median(-dh[ok] / u[1:][ok]):.3f}"
    elif fid in ("F5", "F6"):
        dv = truth["dv_m_s"]
        row["measured"] = (f"burn dv {dv[-1]:.3f} m/s; dv after arm {dv[-1] - dv[after][0]:.3f}"
                           f" m/s")
    elif fid == "F7":
        sides = ["T_bus_nx_K", "T_bus_py_K", "T_bus_ny_K"]
        rel = truth["T_bus_px_K"] - np.mean([truth[s] for s in sides], axis=0)
        row["measured"] = (f"bus_px minus other side faces: {rel[~after].mean():.2f} K before, "
                           f"{rel[after].mean():.2f} K after")
    elif fid == "F8":
        xl = pq.read_table(d / "crosslink.parquet")
        snd, rcv = np.asarray(xl.column("sender")), np.asarray(xl.column("receiver"))
        t = np.asarray(xl.column("t_s"))
        sel = ((snd == sat) | (rcv == sat)) & (t >= t_arm) & np.asarray(xl.column("sent"))
        frac = np.asarray(xl.column("delivered"))[sel].mean()
        row["measured"] = f"delivered fraction after arm {frac:.3f}"
    elif fid == "F9":
        links = params.links.cdh_upsets
        inside = after & truth["in_saa"]
        expect = (links.lambda_out_per_s + links.lambda_in_per_s) * 30.0 * inside.sum()
        row["measured"] = (f"upsets in SAA after arm {truth['upsets'][inside].sum()} vs nominal "
                           f"expectation {expect:.1f}")
    elif fid == "F10":
        takes = _cols(pq.read_table(d / "takes.parquet"), sat)
        dc = params.links.imager_dark_current
        if len(takes["t_take_s"]):
            nominal = dc.D0_e_s_px * 2 ** ((takes["payload_K"][0] - dc.T_ref_K) / dc.delta_T_K)
            ratio = takes["dark_current_e_s_px"][0] / nominal
            row["measured"] = f"dark current / nominal {ratio:.3f}"
    if baseline is not None and (baseline / d.name / "truth.parquet").is_file():
        base = _cols(pq.read_table(baseline / d.name / "truth.parquet"), sat)
        channel = {"F1": "v_V", "F2": "p_gen_W", "F5": "dv_m_s", "F6": "propellant_kg",
                   "F7": "T_bus_px_K"}.get(fid)
        if channel:
            delta = truth[channel][after] - base[channel][after]
            row[f"paired_delta_{channel}"] = float(np.mean(delta))
    return row


def _coverage(params: Params, d: Path, man: dict[str, Any], lab: dict[str, Any]) -> dict[str, Any]:
    fid = man["draws"]["fault_id"]
    row: dict[str, Any] = {"seed": man["seed"], "fault": fid}
    if lab.get("fault") == "none":
        row["burn_at_arm"] = "fault not armed"
        return row
    burns = pq.read_table(d / "burn_summary.parquet")
    b = _cols(burns, lab["vehicle"])
    arm = lab["t_arm_s"]
    start, end = float(b["burn_start_s"][0]), float(b["burn_end_s"][0])
    if np.isnan(start) or start >= arm:
        row["burn_at_arm"] = "not started"
    elif np.isnan(end) or end > arm:
        row["burn_at_arm"] = "running"
    else:
        row["burn_at_arm"] = "finished"
    return row


def _known_gaps() -> str:
    if not _CLAUDE_MD.is_file():
        return "(CLAUDE.md not found)\n"
    text = _CLAUDE_MD.read_text()
    i = text.find("## Known gaps")
    return text[i + len("## Known gaps"):].strip() + "\n" if i >= 0 else "(no Known gaps section)\n"


def _md_table(rows: list[dict[str, Any]], keys: list[str]) -> str:
    if not rows:
        return "(none)\n"
    out = "| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n"
    for r in rows:
        out += "| " + " | ".join(str(r.get(k, "")) for k in keys) + " |\n"
    return out


def build_report(batch_dir: Path, params_dir: Path | None = None,
                 baseline: Path | None = None) -> dict[str, Any]:
    """Write `<batch>/report/report.md` and CSV tables; return the headline numbers."""
    batch_dir = Path(batch_dir)
    params = load_params(params_dir or _DEFAULT_PARAMS)
    out = batch_dir / "report"
    out.mkdir(exist_ok=True)
    dirs = _seed_dirs(batch_dir)
    manifests = [json.loads((d / MANIFEST).read_text()) for d in dirs]
    labels = [json.loads((d / "labels.json").read_text()) for d in dirs]

    drawn = Counter(m["draws"]["fault_id"] for m in manifests)
    applied = Counter(lab.get("fault_id", "none") for lab in labels)
    fault_rows = [{"fault": f"F{i}", "drawn": drawn.get(f"F{i}", 0),
                   "armed": applied.get(f"F{i}", 0)} for i in range(1, 11)]
    fault_rows.append({"fault": "none", "drawn": 0, "armed": applied.get("none", 0)})
    leads = np.array([m["draws"]["fault_lead_s"] for m in manifests]) if manifests else np.zeros(0)

    inv = [r for d in dirs for r in _invariants(params, d)]
    lim = [r for d in dirs for r in _limits(params, d)]
    sig = [_signature(params, d, lab, baseline) for d, lab in zip(dirs, labels, strict=True)
           if lab.get("fault") != "none"]
    cov = [_coverage(params, d, m, lab) for d, m, lab in zip(dirs, manifests, labels, strict=True)
           if m["draws"]["fault_id"] in ("F5", "F6")]
    walls = np.array([m["wall_s"] for m in manifests]) if manifests else np.zeros(0)
    runtime = [{"runs": len(walls), "mean_s": float(walls.mean()) if len(walls) else 0.0,
                "min_s": float(walls.min(initial=0.0)), "max_s": float(walls.max(initial=0.0)),
                "total_s": float(walls.sum())}]
    batch_man = batch_dir / MANIFEST
    if batch_man.is_file():
        bm = json.loads(batch_man.read_text())
        runtime[0]["batch_wall_s"] = bm.get("wall_s")
        runtime[0]["workers"] = bm.get("workers")

    _write_csv(out / "fault_counts.csv", fault_rows)
    _write_csv(out / "invariants.csv", inv)
    _write_csv(out / "limits.csv", lim)
    _write_csv(out / "signatures.csv", sig)
    _write_csv(out / "coverage.csv", cov)
    _write_csv(out / "runtime.csv", runtime)

    runs_pass = [all(r["all_pass"] for r in inv if r["seed"] == m["seed"]) for m in manifests]
    pass_rate = float(np.mean(runs_pass)) if runs_pass else 0.0
    affectable = sum(r.get("burn_at_arm") in ("running", "not started") for r in cov)
    cov_text = (f"{affectable} of {len(cov)} F5/F6 draws have SAT-4's burn running or not yet "
                f"started at arm" if cov else "no F5/F6 draws in this batch")
    failed = [r for r in inv if not r["all_pass"]]
    md = [
        f"# EXPAND.3 validation report\n\nBatch: `{batch_dir}`; {len(dirs)} runs.\n",
        "## Fault draws\n",
        _md_table(fault_rows, ["fault", "drawn", "armed"]),
        (f"\nLeads: {leads.min():.0f} to {leads.max():.0f} s (mean {leads.mean():.0f} s).\n"
         if len(leads) else ""),
        "\n## Fault signatures\n",
        _md_table(sig, ["seed", "fault", "vehicle", "t_arm_h", "label_magnitude", "measured"]),
        "\n## F5/F6 coverage\n", cov_text + ".\n",
        "\n## Invariants\n",
        f"Runs passing every invariant: {sum(runs_pass)} of {len(runs_pass)} "
        f"({pass_rate:.1%}). Checks: energy balance, propellant balance, quaternion norm, SoC in "
        "[0, 1], no NaN in truth.\n",
        "\nFailures:\n" + _md_table(failed, ["seed", "sat_id", "energy_max_err_J", "nan_columns"])
        if failed else "\nNo failures.\n",
        "\n## Thermal limits and undervoltage\n",
        f"Ticks outside thermal limits: {sum(r['thermal_limit_ticks'] for r in lim)}; "
        f"undervoltage-hold ticks: {sum(r['undervoltage_ticks'] for r in lim)}.\n",
        "\n## Runtime\n", _md_table(runtime, list(runtime[0])),
        "\n## Known gaps\n\n", _known_gaps(),
    ]
    (out / "report.md").write_text("".join(md))
    return {"runs": len(dirs), "invariant_pass_rate": pass_rate,
            "fault_counts": dict(drawn), "coverage": cov_text}
