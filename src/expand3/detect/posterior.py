"""Fault posterior probabilities from detector score traces.

Each detector's score S_k(t) is a sliding-window generalized likelihood ratio (a log likelihood
ratio at the best-fit shift and window) for fault k against no fault. This module turns it into
P(fault k active on its vehicle | data up to t) by Bayes' rule.

Prior. The spec injects exactly one fault per seed, uniform over F1..F10, on the fault's own
vehicle, and arms it at (take start - lead) with lead ~ U(low, high) (`scenario.fault_lead_s`).
So a fault of vehicle v is active at t with prior probability pi_k F(t), where F is the CDF of
the arm time given the vehicle's take start, and zero where F = 0 (including a vehicle with no
take, where no fault is armed). Faults that no detector covers look like no fault to the data,
so their prior mass stays on the null.

Likelihood. A raw GLR is not a likelihood ratio: the window and shift were maximized over, and
the residuals are correlated and not exactly Gaussian, so e^S is overconfident. The evidence is
tempered, log Lambda_k = a_k S_k - c_k, with a_k >= 0 and c_k fitted by maximum likelihood on the
tune seeds, the logistic regression of "fault k active" on S_k with the log prior odds as a fixed
offset (empirical Bayes). a_k = 1, c_k = 0 would be the plain GLR. Fitting is per channel; the
posterior then normalizes across the channels of a vehicle:

    P_k = pi_k F e^{log Lambda_k} / (1 - F sum_j pi_j) / (1 + sum_j (same for j))

where j runs over the channels of that vehicle, which is exact when the channels are independent
given the hypothesis and an approximation when they are not.

There must be one detector per fault id, or `select=True`, which fits every candidate and keeps
the one with the lowest log loss on the tune seeds (the test seeds play no part in the choice).
Only the faulted vehicle of each fault carries prior
mass, so every other vehicle has P = 0.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

from expand3.detect.base import Alarm, Trace
from expand3.detect.config import DetectConfig
from expand3.detect.data import SeedData, seed_dirs
from expand3.detect.glr import episodes
from expand3.detect.registry import REGISTRY
from expand3.params import load_params

F64 = NDArray[np.float64]
FAULT_ID = "F"


@dataclass(frozen=True)
class Prior:
    """Seed-level prior of one fault: probability pi, its vehicle, and the lead range."""

    fault_id: str
    vehicle: str
    pi: float
    lead_low_s: float
    lead_high_s: float


@dataclass(frozen=True)
class Channel:
    """Tempered evidence log Lambda = a S - c for one fault."""

    a: float
    c: float

    def log_lr(self, s: F64) -> F64:
        return np.asarray(self.a * np.asarray(s, dtype=np.float64) - self.c)

    def prob(self, s: F64, log_prior_odds: F64) -> F64:
        """P(fault | score) for one channel alone: expit(prior log odds + log Lambda)."""
        z = np.asarray(log_prior_odds, dtype=np.float64) + self.log_lr(s)
        return np.asarray(expit(z))


def load_priors(params_dir: Path) -> dict[str, Prior]:
    """Priors from `faults.yaml` (vehicle, one fault per seed uniform over those listed) and
    `scenario.yaml` (the lead draw)."""
    p = load_params(params_dir)
    lead = p.scenario.fault_lead_s
    if lead.low is None or lead.high is None or lead.high <= lead.low:
        raise ValueError("the posterior needs a uniform fault lead with low < high")
    n = len(p.faults.faults)
    return {fid: Prior(fid, f.vehicle, 1.0 / n, float(lead.low), float(lead.high))
            for fid, f in p.faults.faults.items()}


def onset_cdf(t_s: F64, t_take_s: float | None, low_s: float, high_s: float) -> F64:
    """P(arm <= t) with arm = take - lead and lead ~ U(low, high); zero with no take."""
    t = np.asarray(t_s, dtype=np.float64)
    if t_take_s is None:
        return np.zeros(t.shape)
    return np.asarray(np.clip((t - (t_take_s - high_s)) / (high_s - low_s), 0.0, 1.0))


def log_prior_odds(t_s: F64, t_take_s: float | None, prior: Prior, modeled_pi: float) -> F64:
    """log of pi F / (1 - F modeled_pi): the odds of this fault against the null, which holds
    the prior mass of every fault without a channel. -inf where F = 0."""
    f = onset_cdf(t_s, t_take_s, prior.lead_low_s, prior.lead_high_s)
    num = prior.pi * f
    den = 1.0 - f * modeled_pi
    with np.errstate(divide="ignore"):
        return np.asarray(np.log(num) - np.log(den))


def fit_channel(s: F64, log_prior_odds_: F64, y: F64) -> Channel:
    """Maximum likelihood (a >= 0, c) of logit P = offset + a S - c. Rows without prior mass
    are dropped."""
    s, off, y = (np.asarray(v, dtype=np.float64) for v in (s, log_prior_odds_, y))
    keep = np.isfinite(off) & np.isfinite(s)
    s, off, y = s[keep], off[keep], y[keep]
    if len(s) == 0:
        return Channel(a=0.0, c=0.0)

    def nll(w: F64) -> tuple[float, F64]:
        z = off + w[0] * s - w[1]
        loss = float(np.sum(np.logaddexp(0.0, z) - y * z))
        r = expit(z) - y
        return loss, np.array([np.sum(r * s), -np.sum(r)])

    res = minimize(nll, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B",
                   bounds=[(0.0, None), (None, None)])
    return Channel(a=float(res.x[0]), c=float(res.x[1]))


def combine(log_odds: dict[str, F64]) -> tuple[dict[str, F64], F64]:
    """Normalize channel log odds (against the null) into P(fault k) and P(none)."""
    names = list(log_odds)
    stack = np.stack([log_odds[n] for n in names])
    log_z = np.logaddexp.reduce(np.concatenate([np.zeros((1, stack.shape[1])), stack]), axis=0)
    return ({n: np.exp(log_odds[n] - log_z) for n in names}, np.exp(-log_z))


def reliability(p: F64, y: F64, bins: int = 10, p_prior: F64 | None = None) -> dict[str, Any]:
    """Log loss, Brier score, expected calibration error and the reliability table."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    y = np.asarray(y, dtype=np.float64)

    def loss(q: F64) -> float:
        q = np.clip(q, 1e-12, 1.0 - 1e-12)
        return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))

    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        sel = which == b
        if sel.any():
            table.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(sel.sum()),
                          "mean_p": float(p[sel].mean()), "frac_fault": float(y[sel].mean())})
            ece += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    out: dict[str, Any] = {"n": len(p), "n_fault": int(y.sum()), "log_loss": loss(p),
                           "brier": float(np.mean((p - y) ** 2)), "ece": float(ece),
                           "bins": table}
    if p_prior is not None:
        out["log_loss_prior"] = loss(np.asarray(p_prior, dtype=np.float64))
        out["brier_prior"] = float(np.mean((np.asarray(p_prior) - y) ** 2))
    return out


def _candidates(names: list[str], priors: dict[str, Prior], select: bool) -> dict[str, list[str]]:
    """Fault id to the detectors that cover it; two need `select`."""
    out: dict[str, list[str]] = {}
    for n in names:
        for fid in REGISTRY[n].fault_ids:
            if fid in priors:
                out.setdefault(fid, []).append(n)
    if not out:
        raise ValueError("no detector covers a fault with a prior")
    for fid, found in out.items():
        if len(found) > 1 and not select:
            raise ValueError(f"{fid} has two detectors ({', '.join(found)}); pass one per fault "
                             "or select=True")
    return out


def _trace(traces: dict[str, list[Trace]], name: str, fid: str, sat: str) -> Trace:
    for tr in traces[name]:
        if tr.fault_id == fid and tr.sat_id == sat:
            return tr
    raise KeyError(f"{name} produced no {fid} trace for {sat}")


def _label(d: SeedData, prior: Prior, t: F64) -> F64:
    if d.fault_id != prior.fault_id or d.vehicle != prior.vehicle or d.t_arm_s is None:
        return np.zeros(t.shape)
    return (t >= d.t_arm_s).astype(np.float64)


def _zoh(t_to: F64, t_from: F64, x: F64) -> F64:
    """Zero-order hold of x (sampled at t_from) onto t_to; -inf before the first sample."""
    idx = np.searchsorted(t_from, t_to, side="right") - 1
    return np.where(idx >= 0, x[np.clip(idx, 0, None)], -np.inf)


def posterior_batch(batch: Path, names: list[str], tune: list[int], test: list[int],
                    traces: dict[int, Any], cfg: DetectConfig, params_dir: Path,
                    p_alarm: float = 0.9, select: bool = False) -> dict[str, Any]:
    """Fit the channels on the tune seeds and compute the posterior on every seed in tune and
    test. `traces` is the harness trace set (seed -> (traces by detector, info)). With `select`
    a fault covered by several detectors keeps the one with the lowest tune log loss."""
    del cfg
    priors = load_priors(params_dir)
    cands = _candidates(names, priors, select)
    dirs = seed_dirs(batch)
    modeled: dict[str, float] = {}
    for fid in cands:
        modeled[priors[fid].vehicle] = modeled.get(priors[fid].vehicle, 0.0) + priors[fid].pi

    def rows(seed: int, fid: str, det: str) -> tuple[Trace, F64, F64]:
        pr = priors[fid]
        d = SeedData(dirs[seed])
        tr = _trace(traces[seed][0], det, fid, pr.vehicle)
        off = log_prior_odds(tr.t_s, d.t_take_s(pr.vehicle), pr, modeled[pr.vehicle])
        return tr, off, _label(d, pr, tr.t_s)

    chan_of: dict[str, str] = {}
    fitted: dict[str, Channel] = {}
    candidates: dict[str, dict[str, dict[str, float]]] = {}
    for fid, dets in cands.items():
        best = None
        for det in dets:
            cols = [rows(s, fid, det) for s in tune]
            s_all = np.concatenate([np.nan_to_num(c[0].score) for c in cols])
            off_all = np.concatenate([c[1] for c in cols])
            y_all = np.concatenate([c[2] for c in cols])
            ch = fit_channel(s_all, off_all, y_all)
            keep = np.isfinite(off_all)
            loss = reliability(ch.prob(s_all, off_all)[keep], y_all[keep])["log_loss"]
            candidates.setdefault(fid, {})[det] = {"a": ch.a, "c": ch.c, "tune_log_loss": loss}
            if best is None or loss < best[0]:
                best = (loss, det, ch)
        assert best is not None
        chan_of[fid], fitted[fid] = best[1], best[2]

    def rows_of(seed: int, fid: str) -> tuple[Trace, F64, F64]:
        return rows(seed, fid, chan_of[fid])

    posteriors: dict[int, dict[str, dict[str, F64]]] = {}
    pooled: dict[str, dict[str, list[list[F64]]]] = {
        sp: {fid: [[], [], []] for fid in chan_of} for sp in ("tune", "test")}
    alarms: list[Alarm] = []
    for split, seeds in (("tune", tune), ("test", test)):
        for seed in seeds:
            per_sat: dict[str, dict[str, tuple[Trace, F64]]] = {}
            for fid in chan_of:
                tr, off, y = rows_of(seed, fid)
                s = np.nan_to_num(tr.score)
                log_odds = off + fitted[fid].log_lr(s)
                per_sat.setdefault(priors[fid].vehicle, {})[fid] = (tr, log_odds)
                finite = np.isfinite(off)
                pooled[split][fid][0].append(fitted[fid].prob(s, off)[finite])
                pooled[split][fid][1].append(y[finite])
                pooled[split][fid][2].append(1.0 / (1.0 + np.exp(-off[finite])))
            posteriors[seed] = {}
            for sat, chans in per_sat.items():
                grid = next(iter(chans.values()))[0].t_s
                aligned = {fid: _zoh(grid, tr.t_s, lo) for fid, (tr, lo) in chans.items()}
                p, p_none = combine(aligned)
                posteriors[seed][sat] = {"t_s": grid, "none": p_none, **p}
                for fid, pk in p.items():
                    alarms += [Alarm(seed, "posterior", sat, float(grid[i]), fid, float(pk[i]))
                               for i in episodes(pk, p_alarm)]
    metrics = {sp: {fid: reliability(np.concatenate(v[0]), np.concatenate(v[1]),
                                     p_prior=np.concatenate(v[2]))
                    for fid, v in per.items()} for sp, per in pooled.items()}
    alarms.sort(key=lambda a: (a.seed, a.t_s, a.sat_id, a.fault_id))
    return {"channels": {fid: {"a": c.a, "c": c.c, "detector": chan_of[fid],
                               "vehicle": priors[fid].vehicle, "pi": priors[fid].pi}
                         for fid, c in fitted.items()},
            "candidates": candidates, "metrics": metrics, "posteriors": posteriors,
            "alarms": alarms,
            "tune_seeds": sorted(tune), "test_seeds": sorted(test), "p_alarm": p_alarm}


def write_posterior(out: Path, post: dict[str, Any]) -> None:
    """`posterior.json` (channels and metrics), `posterior.csv` (long) and `posterior.md`."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "posterior.json").write_text(json.dumps(
        {k: post[k] for k in ("channels", "candidates", "metrics", "tune_seeds", "test_seeds",
                              "p_alarm")},
        indent=2, sort_keys=True) + "\n")
    test = set(post["test_seeds"])
    with (out / "posterior.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "split", "sat_id", "t_s", "fault_id", "p"])
        for seed, sats in sorted(post["posteriors"].items()):
            for sat, cols in sats.items():
                for fid, p in cols.items():
                    if fid == "t_s":
                        continue
                    for t, v in zip(cols["t_s"], p, strict=True):
                        w.writerow([seed, "test" if seed in test else "tune", sat, f"{t:.0f}",
                                    fid, f"{v:.6g}"])
    lines = ["# Fault posterior", "",
             "Channels (log Lambda = a S - c, fitted on the tune seeds):", "",
             "| fault | detector | vehicle | pi | a | c |", "|---|---|---|---|---|---|"]
    lines += [f"| {fid} | {c['detector']} | {c['vehicle']} | {c['pi']:.3g} | {c['a']:.3g} "
              f"| {c['c']:.3g} |" for fid, c in sorted(post["channels"].items())]
    if any(len(c) > 1 for c in post["candidates"].values()):
        lines += ["", "Candidates (the lowest tune log loss is kept):", "",
                  "| fault | detector | a | c | tune log loss |", "|---|---|---|---|---|"]
        lines += [f"| {fid} | {det} | {m['a']:.3g} | {m['c']:.3g} | {m['tune_log_loss']:.4f} |"
                  for fid, c in sorted(post["candidates"].items()) for det, m in sorted(c.items())]
    for split in ("tune", "test"):
        lines += ["", f"## {split}: calibration of P(fault active) where the prior is nonzero", "",
                  "| fault | n | n fault | log loss | prior-only | Brier | prior-only | ECE |",
                  "|---|---|---|---|---|---|---|---|"]
        for fid, m in sorted(post["metrics"][split].items()):
            lines.append(f"| {fid} | {m['n']} | {m['n_fault']} | {m['log_loss']:.4f} "
                         f"| {m['log_loss_prior']:.4f} | {m['brier']:.4f} "
                         f"| {m['brier_prior']:.4f} | {m['ece']:.4f} |")
    for fid, m in sorted(post["metrics"]["test"].items()):
        lines += ["", f"### {fid} reliability (test)", "",
                  "| P range | n | mean P | fraction active |", "|---|---|---|---|"]
        lines += [f"| {b['lo']:.1f}-{b['hi']:.1f} | {b['n']} | {b['mean_p']:.3f} "
                  f"| {b['frac_fault']:.3f} |" for b in m["bins"]]
    (out / "posterior.md").write_text("\n".join(lines) + "\n")


__all__ = ["Channel", "Prior", "combine", "fit_channel", "load_priors", "log_prior_odds",
           "onset_cdf", "posterior_batch", "reliability", "write_posterior"]
