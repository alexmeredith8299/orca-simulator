"""Command line: `expand3 run`, `expand3 report` (M8), `expand3 detect`, `expand3 score`,
`expand3 compare` and `expand3 posterior`."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

from expand3.detect.classify import run_classifier, save_model
from expand3.detect.config import DetectConfig
from expand3.detect.harness import (
    calibrate_nominal,
    compute_traces,
    detect_batch,
    detector_names,
    read_alarms,
    write_alarms,
)
from expand3.detect.posterior import posterior_batch, write_posterior
from expand3.detect.rawclf import run_raw_classifier
from expand3.detect.registry import CLASSIFIER, CLASSIFIERS, RAW_CLASSIFIER, REGISTRY, describe
from expand3.detect.score import compare, score_seeds, write_comparison, write_score
from expand3.faults import FaultConfig
from expand3.report import build_report
from expand3.run import run_batch

DEFAULT_PARAMS = Path(__file__).resolve().parents[2] / "params"


def parse_seeds(text: str) -> list[int]:
    """`0:100` (half-open range), `7`, or `1,5,9`."""
    if ":" in text:
        lo, hi = text.split(":", 1)
        return list(range(int(lo), int(hi)))
    return [int(x) for x in text.split(",") if x.strip()]


def fault_config(fault: str, lead: float | None) -> FaultConfig:
    if fault == "seeded":
        return FaultConfig(mode="seeded")
    if fault == "none":
        return FaultConfig(mode="none")
    return FaultConfig(mode="forced", fault_id=fault, lead_s=lead)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="expand3", description="EXPAND.3 Phase I simulator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run seeds and write one directory per seed")
    r.add_argument("--seeds", required=True, help="0:100, 7 or 1,5,9")
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--fault", default="seeded",
                   choices=["seeded", "none", *[f"F{i}" for i in range(1, 11)]])
    r.add_argument("--lead", type=float, default=None, help="forced fault lead [s]")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--force", action="store_true", help="rerun completed seeds")
    r.add_argument("--duration-h", type=float, default=None,
                   help="override the drawn run length (tests, CI)")
    r.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    p = sub.add_parser("report", help="write the validation report for a batch directory")
    p.add_argument("batch", type=Path)
    p.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    p.add_argument("--baseline", type=Path, default=None,
                   help="a no-fault batch of the same seeds, for paired signature deltas")
    de = sub.add_parser("detect", help="run fault detectors over a batch and write alarms")
    de.add_argument("batch", type=Path)
    de.add_argument("--detector", required=True,
                    help="a detector name (f2, f8, f9, f10, wheels, lrn_*, peer_*, unsup_*), a "
                         "family (physics, learned, peer, unsupervised), clf, a comma list, or all "
                         "(every detector but clf)")
    de.add_argument("--forced", type=Path, default=None,
                    help="directory of forced-fault batches F1..F10 (for clf training)")
    de.add_argument("--tune", default="0:70", help="seeds that calibrate the detectors")
    de.add_argument("--seeds", default=None, help="seeds to run on (default: every seed)")
    de.add_argument("--out", type=Path, default=None, help="default: BATCH/detect/DETECTOR")
    de.add_argument("--target-far", type=float, default=DetectConfig().target_far_per_vehicle_day,
                    help="alarm episodes per fault-free vehicle-day the thresholds allow")
    de.add_argument("--workers", type=int, default=1)
    de.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    cp = sub.add_parser("compare", help="per-fault comparison of detectors from alarm files")
    cp.add_argument("batch", type=Path)
    cp.add_argument("--alarms", type=Path, nargs="*", default=None,
                    help="default: every BATCH/detect/*/alarms.csv")
    cp.add_argument("--tune", default="0:70")
    cp.add_argument("--test", default="70:100")
    cp.add_argument("--out", type=Path, default=None, help="default: BATCH/detect/compare")
    sc = sub.add_parser("score", help="score alarms against labels.json")
    sc.add_argument("batch", type=Path)
    sc.add_argument("--alarms", type=Path, default=None, help="default: BATCH/detect/*/alarms.csv")
    sc.add_argument("--tune", default="0:70")
    sc.add_argument("--test", default="70:100")
    sc.add_argument("--out", type=Path, default=None, help="default: next to the alarms")
    po = sub.add_parser("posterior", help="fault probabilities from detector scores")
    po.add_argument("batch", type=Path)
    po.add_argument("--detector", required=True,
                    help="detector names, one per fault id unless --select (for example "
                         "mcd_thermal,mcd_eps), or all")
    po.add_argument("--select", action="store_true",
                    help="when several detectors cover a fault, keep the one with the lowest "
                         "tune log loss")
    po.add_argument("--tune", default="0:70", help="seeds that fit the posterior")
    po.add_argument("--test", default="70:100")
    po.add_argument("--nominal", type=Path, default=None,
                    help="nominal.pkl of an earlier `detect` run of the same detectors and tune "
                         "seeds, to skip calibration")
    po.add_argument("--p-alarm", type=float, default=0.9,
                    help="probability at which an alarm is written (for `expand3 score`)")
    po.add_argument("--out", type=Path, default=None, help="default: BATCH/detect/posterior")
    po.add_argument("--workers", type=int, default=1)
    po.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    args = ap.parse_args(argv)
    if args.cmd == "posterior":
        return _posterior(args)
    if args.cmd == "detect":
        return _detect(args)
    if args.cmd == "score":
        return _score(args)
    if args.cmd == "compare":
        return _compare(args)
    if args.cmd == "run":
        batch = run_batch(args.params, parse_seeds(args.seeds), args.out,
                          fault_config(args.fault, args.lead), args.duration_h, args.workers,
                          args.force)
        c = batch["counts"]
        print(f"ok {c['ok']}, skipped {c['skipped']}, failed {c['failed']} "
              f"in {batch['wall_s']:.1f} s -> {args.out}")
        return 1 if c["failed"] else 0
    summary = build_report(args.batch, args.params, args.baseline)
    print(f"report: {summary['runs']} runs, invariant pass rate "
          f"{summary['invariant_pass_rate']:.3f} -> {args.batch / 'report' / 'report.md'}")
    return 0


def _batch_seeds(batch: Path) -> list[int]:
    return sorted(int(d.name.split("_")[1]) for d in batch.glob("seed_*")
                  if (d / "manifest.json").is_file())


_INFO = {CLASSIFIER: "classifier", RAW_CLASSIFIER: "classifier_raw"}


def _posterior(args: argparse.Namespace) -> int:
    # unsupervised detectors name subsystems, not faults, so they carry no posterior
    names = [n for n in detector_names(args.detector)
             if any(re.fullmatch(r"F\d+", f) for f in REGISTRY[n].fault_ids)]
    tune, test = parse_seeds(args.tune), parse_seeds(args.test)
    out = args.out or args.batch / "detect" / "posterior"
    cfg = DetectConfig()
    if args.nominal is not None:
        with args.nominal.open("rb") as f:
            nominal = {n: v for n, v in pickle.load(f).items() if n in names}
    else:
        nominal = calibrate_nominal(args.batch, names, tune, cfg, args.params, args.workers)
    traces = compute_traces(args.batch, nominal, [*tune, *test], cfg, args.params, args.workers)
    post = posterior_batch(args.batch, names, tune, test, traces, cfg, args.params, args.p_alarm,
                           args.select)
    write_posterior(out, post)
    write_alarms(out / "alarms.csv", post["alarms"])
    for fid, c in sorted(post["channels"].items()):
        m = post["metrics"]["test"][fid]
        print(f"{fid} ({c['detector']}): a={c['a']:.3g} c={c['c']:.3g}; test log loss "
              f"{m['log_loss']:.4f} vs prior {m['log_loss_prior']:.4f}, ECE {m['ece']:.4f}")
    print(f"posterior -> {out / 'posterior.md'}")
    return 0


def _detect(args: argparse.Namespace) -> int:
    names = detector_names(args.detector)
    seeds = _batch_seeds(args.batch) if args.seeds is None else parse_seeds(args.seeds)
    out = args.out or args.batch / "detect" / args.detector.replace(",", "+")
    cfg = DetectConfig(target_far_per_vehicle_day=args.target_far)
    tune = parse_seeds(args.tune)
    plain = [n for n in names if n not in CLASSIFIERS]
    alarms = detect_batch(args.batch, plain, tune, seeds, out, cfg, args.params, args.workers)
    for clf in (n for n in names if n in CLASSIFIERS):
        if args.forced is None:
            raise SystemExit(f"{clf} needs --forced DIR with the forced-fault batches F1..F10")
        forced = sorted(p for p in args.forced.glob("F*") if p.is_dir())
        if clf == CLASSIFIER:
            clf_alarms, result = run_classifier(args.batch, forced, list(cfg.clf_inputs), tune,
                                                seeds, cfg, args.params, REGISTRY, args.workers)
        else:
            clf_alarms, result = run_raw_classifier(args.batch, forced, tune, seeds, cfg,
                                                    args.params, args.workers)
        save_model(out, result, clf)
        alarms = sorted([*alarms, *clf_alarms], key=lambda a: (a.seed, a.t_s, a.sat_id))
        write_alarms(out / "alarms.csv", alarms)
        (out / f"{_INFO[clf]}.json").write_text(json.dumps(result["info"], indent=2) + "\n")
    print(f"detect: {len(alarms)} alarms over {len(seeds)} seeds -> {out / 'alarms.csv'}")
    return 0


def _score(args: argparse.Namespace) -> int:
    path = args.alarms
    if path is None:
        found = sorted((args.batch / "detect").glob("*/alarms.csv"))
        if len(found) != 1:
            raise SystemExit(f"pass --alarms: found {len(found)} alarm files under "
                             f"{args.batch / 'detect'}")
        path = found[0]
    done = set(_batch_seeds(args.batch))
    splits = {name: [s for s in parse_seeds(text) if s in done]
              for name, text in (("tune", args.tune), ("test", args.test))}
    alarms = read_alarms(path)
    result = {n: score_seeds(args.batch, alarms, seeds) for n, seeds in splits.items() if seeds}
    out = args.out or path.parent
    write_score(out, result)
    for n, r in result.items():
        print(f"{n}: {r['false_alarms']} false alarms in {r['vehicle_days']:.1f} vehicle-days; "
              + ", ".join(f"{x['fault']} {x['detected']}/{x['seeds']}" for x in r["rows"]
                          if x["seeds"]))
    print(f"score -> {out / 'score.md'}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    paths = args.alarms or sorted((args.batch / "detect").glob("*/alarms.csv"))
    by_detector: dict[str, list[Any]] = {}
    for p in paths:  # a later file replaces an earlier one's alarms for the same detector
        fresh: dict[str, list[Any]] = {}
        for a in read_alarms(p):
            fresh.setdefault(a.detector, []).append(a)
        by_detector.update(fresh)
    alarms = [a for al in by_detector.values() for a in al]
    # detectors that were run but never alarmed still get a row
    ran = {n for p in paths for n in _ran(p)}
    names = sorted({a.detector for a in alarms} | ran)
    info = {n: describe(n) for n in names}
    done = set(_batch_seeds(args.batch))
    out = args.out or args.batch / "detect" / "compare"
    for split, text in (("tune", args.tune), ("test", args.test)):
        seeds = [s for s in parse_seeds(text) if s in done]
        if not seeds:
            continue
        results = compare(args.batch, alarms, seeds, {n for n in names if info[n][1]})
        write_comparison(out, results, {n: info[n][0] for n in names},
                         {n: info[n][2] for n in names}, split)
        print(f"{split}: {len(results)} detectors over {len(seeds)} seeds")
    print(f"compare -> {out}")
    return 0


def _ran(alarms_path: Path) -> list[str]:
    """Detectors that produced the alarm file, from the calibration beside it."""
    names: list[str] = []
    cal = alarms_path.parent / "calibration.json"
    if cal.is_file():
        names += list(json.loads(cal.read_text()).get("thresholds", {}))
    names += [n for n in CLASSIFIERS if (alarms_path.parent / f"{_INFO[n]}.json").is_file()]
    return names
