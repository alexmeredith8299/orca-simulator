"""Command line: `expand3 run` and `expand3 report` (M8)."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    args = ap.parse_args(argv)
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
