"""Score alarms against the labels: delay, lead, false alarms and isolation.

For a seed with a fault on vehicle V armed at t_arm, an alarm counts as a detection when it is on
V at or after t_arm. Every other alarm (other vehicles, and V before arm) is a false alarm, and
the exposure is the fault-free vehicle-days. Per fault the table gives, over the seeds of a split:

- detection: some alarm on V after arm names the right fault (`detected`), with the delay from
  arm to the first such alarm and the lead, the time from that alarm to the take start of V
  (negative if the alarm comes after the take);
- false alarms per vehicle-day that name this fault;
- isolation before the take: the first right alarm came before the take start of V;
- the observability ceiling: whether the fault's window opened before the take. The window opens
  when the fault first acts on something the vehicle reports: at arm for faults that act
  continuously (F1, F2, F7, F8), at the first SAA tick after arm (F9), the first burn tick after
  arm (F5, F6), the start of the slew in (F3, F4) and the take itself (F10). The ceiling is the
  fraction of seeds with that time before the take start;
- isolation: the confusion matrix of the first alarm on V after arm (any fault id) against the
  true fault, with `missed` for no alarm and a `none` row for the guesses of false alarms.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from expand3.detect.base import Alarm
from expand3.detect.data import SeedData, seed_dirs

MISSED = "missed"
NONE = "none"


def _fault_order(fid: str) -> tuple[int, int, str]:
    """F1..F10 in numeric order, then any other id (an unsupervised subsystem) by name."""
    if fid[:1] == "F" and fid[1:].isdigit():
        return (0, int(fid[1:]), "")
    return (1, 0, fid)


def evidence_time(d: SeedData, vehicle: str, fault: str, arm: float) -> float | None:
    """When the fault's window opens: the first time it acts on something the vehicle reports."""
    if fault in ("F3", "F4"):
        table = d.table("takes", ["sat_id", "slew_in_start_s"])
        sel = np.asarray(table.column("sat_id")) == vehicle
        starts = np.asarray(table.column("slew_in_start_s"))[sel]
        return float(max(arm, starts.min())) if starts.size else None
    if fault == "F10":
        return d.t_take_s(vehicle)
    if fault in ("F5", "F6"):
        table = d.table("burns", ["sat_id", "t_s"])
        t = np.asarray(table.column("t_s"))[np.asarray(table.column("sat_id")) == vehicle]
        t = t[t >= arm]
        return float(t.min()) if t.size else None
    if fault == "F9":
        m = d.measurements(vehicle)
        t = m["t_s"][(m["gate_saa"] > 0) & (m["t_s"] >= arm)]
        return float(t.min()) if t.size else None
    return arm


def score_seeds(batch: Path, alarms: list[Alarm], seeds: list[int],
                detect_only: bool = False) -> dict[str, Any]:
    """Per-fault statistics, the confusion matrix and false-alarm totals for one split.

    With `detect_only` the alarms do not name a fault (unsupervised), so any alarm on the faulted
    vehicle after arm counts as naming the right one.
    """
    dirs = seed_dirs(batch)
    by_seed: dict[int, list[Alarm]] = defaultdict(list)
    for a in alarms:
        by_seed[a.seed].append(a)
    delay: dict[str, list[float]] = defaultdict(list)
    lead: dict[str, list[float]] = defaultdict(list)
    n_seeds: Counter[str] = Counter()
    detected: Counter[str] = Counter()
    first_guess: dict[str, Counter[str]] = defaultdict(Counter)
    before_take: Counter[str] = Counter()
    window_open: Counter[str] = Counter()
    ceiling_lead: dict[str, list[float]] = defaultdict(list)
    fa_by_fault: Counter[str] = Counter()
    exposure = 0.0
    for s in seeds:
        d = SeedData(dirs[s])
        exposure += sum(d.fault_free_days(sat) for sat in d.sat_ids)
        truth, vehicle, arm = d.fault_id, d.vehicle, d.t_arm_s
        mine = sorted(by_seed.get(s, []), key=lambda a: a.t_s)
        post = [a for a in mine if a.sat_id == vehicle and arm is not None and a.t_s >= arm]
        for a in mine:
            if a not in post:
                fa_by_fault[a.fault_id] += 1
                first_guess[NONE][a.fault_id] += 1
        if vehicle is None or arm is None:
            continue
        n_seeds[truth] += 1
        first_guess[truth][(truth if detect_only else post[0].fault_id) if post else MISSED] += 1
        right = [a for a in post if detect_only or a.fault_id == truth]
        take = d.t_take_s(vehicle)
        evidence = evidence_time(d, vehicle, truth, arm)
        if evidence is not None and take is not None and evidence < take:
            window_open[truth] += 1
            ceiling_lead[truth].append(take - evidence)
        if right:
            detected[truth] += 1
            delay[truth].append(right[0].t_s - arm)
            if take is not None:
                lead[truth].append(take - right[0].t_s)
                if right[0].t_s < take:
                    before_take[truth] += 1
    faults = sorted({*n_seeds, *fa_by_fault}, key=_fault_order)
    rows = []
    for f in faults:
        n = n_seeds[f]
        rows.append({
            "fault": f, "seeds": n, "detected": detected[f],
            "detection_rate": detected[f] / n if n else float("nan"),
            "isolated_first": first_guess[f][f],
            "isolated_before_take": before_take[f],
            "isolated_before_take_rate": before_take[f] / n if n else float("nan"),
            "window_open_before_take": window_open[f],
            "ceiling": window_open[f] / n if n else float("nan"),
            "ceiling_lead_median_s": (float(np.median(ceiling_lead[f])) if ceiling_lead[f]
                                      else float("nan")),
            "delay_median_s": float(np.median(delay[f])) if delay[f] else float("nan"),
            "delay_max_s": float(np.max(delay[f])) if delay[f] else float("nan"),
            "lead_median_s": float(np.median(lead[f])) if lead[f] else float("nan"),
            "lead_min_s": float(np.min(lead[f])) if lead[f] else float("nan"),
            "false_alarms": fa_by_fault[f],
            "false_alarms_per_vehicle_day": fa_by_fault[f] / exposure if exposure else 0.0,
        })
    cols = sorted({g for c in first_guess.values() for g in c if g not in (MISSED,)},
                  key=_fault_order)
    return {"seeds": list(seeds), "vehicle_days": exposure, "rows": rows,
            "confusion": {"rows": [*sorted(n_seeds, key=_fault_order), NONE],
                          "cols": [*cols, MISSED],
                          "counts": {r: dict(first_guess[r]) for r in first_guess}},
            "false_alarms": sum(fa_by_fault.values()),
            "false_alarms_per_vehicle_day":
                sum(fa_by_fault.values()) / exposure if exposure else 0.0}


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return "-" if np.isnan(x) else f"{x:.3g}"
    return str(x)


def _table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["(none)"]
    keys = list(rows[0])
    out = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    out += ["| " + " | ".join(_fmt(r[k]) for k in keys) + " |" for r in rows]
    return out


def _confusion_rows(res: dict[str, Any]) -> list[dict[str, Any]]:
    conf = res["confusion"]
    return [{"true": r, **{c: conf["counts"].get(r, {}).get(c, 0) for c in conf["cols"]}}
            for r in conf["rows"]]


def write_score(out: Path, splits: dict[str, dict[str, Any]]) -> None:
    """Markdown and CSV tables for each split."""
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Detection score", ""]
    for name, res in splits.items():
        lines += [f"## {name}: {len(res['seeds'])} seeds, {res['vehicle_days']:.1f} fault-free "
                  f"vehicle-days", "",
                  f"False alarms: {res['false_alarms']} "
                  f"({res['false_alarms_per_vehicle_day']:.3f} per vehicle-day)", "",
                  *_table(res["rows"]), "", "Isolation: first alarm on the faulted vehicle after "
                  "arm, by guessed fault; the `none` row is the guesses of false alarms.", "",
                  *_table(_confusion_rows(res)), ""]
        for fname, rows in (("per_fault", res["rows"]), ("confusion", _confusion_rows(res))):
            if rows:
                with (out / f"{fname}_{name}.csv").open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
    (out / "score.md").write_text("\n".join(lines) + "\n")


def compare(batch: Path, alarms: list[Alarm], seeds: list[int], detection_only: set[str]
            ) -> dict[str, dict[str, Any]]:
    """Score each detector's alarms on their own, for the comparison tables."""
    names = sorted({a.detector for a in alarms})
    return {n: score_seeds(batch, [a for a in alarms if a.detector == n], seeds,
                           n in detection_only) for n in names}


def write_comparison(out: Path, results: dict[str, dict[str, Any]], families: dict[str, str],
                     covers: dict[str, set[str]], split: str) -> None:
    """One table per fault, a row per detector that can address it: detection rate, median delay
    from arm, the fraction isolated before the take and false alarms per vehicle-day (all the
    detector's false alarms), after the fault's observability ceiling."""
    out.mkdir(parents=True, exist_ok=True)
    faults = sorted({r["fault"] for res in results.values() for r in res["rows"] if r["seeds"]},
                    key=_fault_order)
    lines = [f"# Detector comparison, {split} seeds", ""]
    for f in faults:
        any_row = next(r for res in results.values() for r in res["rows"]
                       if r["fault"] == f and r["seeds"])
        lines += [f"## {f}", "",
                  f"{any_row['seeds']} seeds. Observability ceiling: the window opened before the "
                  f"take in {any_row['window_open_before_take']} "
                  f"({_fmt(any_row['ceiling'])}), with a median of "
                  f"{_fmt(any_row['ceiling_lead_median_s'])} s between opening and the take.", ""]
        rows = []
        for n, res in results.items():
            if f not in covers[n]:
                continue
            row = next((r for r in res["rows"] if r["fault"] == f), None)
            if row is None or not row["seeds"]:
                continue
            rows.append({"detector": n, "family": families[n],
                         "detection_rate": row["detection_rate"],
                         "delay_median_s": row["delay_median_s"],
                         "isolated_before_take": row["isolated_before_take_rate"],
                         "false_alarms_per_vehicle_day": res["false_alarms_per_vehicle_day"]})
        lines += [*_table(rows), ""]
        if rows:
            with (out / f"compare_{split}_{f}.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    (out / f"compare_{split}.md").write_text("\n".join(lines) + "\n")
