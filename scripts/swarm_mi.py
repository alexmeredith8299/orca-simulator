"""Swarm value of each fault channel: I(y; x_swarm | x) = H(y | x) - H(y | x, x_swarm).

Exploratory analysis on batch outputs (environment latent: nothing about the environment is
known to the detector beyond the vehicle's own telemetry). Reads batches only, imports nothing
from the simulator but `params` and the detection helpers.

Usage (conda activate orca; batch paths are relative to $ORCA_BATCHES, default ~/orca-batches):
  python scripts/swarm_mi.py selftest
  python scripts/swarm_mi.py null  NONE OUT.pkl            null checks and noise models
  python scripts/swarm_mi.py pair  NONE BATCH...            forced runs vs the no-fault run
  python scripts/swarm_mi.py mi    NONE NULL.pkl OUT.json [FAULT,...]

NONE is a no-fault batch (e.g. $ORCA_BATCHES/gyro/none). `null` takes seeds 0:35 to fit and 35:70
to evaluate:
  - residuals: own-context ridge (thermal, pack voltage, string power); spec models for EPS
    (`vres` = V - OCV(SoC est) - I R, `slr_1` = log of string 1 over the other strings);
    gyro - star-tracker rate, magnetometer - model, GNSS fix - nav;
  - swarm residual e: x minus a ridge on the peers' residuals (lags -2..2, two slow filters),
    the time-domain Schur complement;
  - shape per regime, after scaling by the regime spread, and of window means; kappa = sigma^2 J
    (Fisher information for location, 1 = Gaussian);
  - for the MI stage: regime spreads, within-seed ACF (tapered, PSD) and the per-seed offset.

`mi`: y is binary, "fault active on its vehicle since its arm tick" vs "no fault", prior PI, known
onset and known signature (the paired forced minus no-fault delta). The window runs PRE_S before
arm (signature 0, it pins the baseline) to arm + horizon. Closed forms: Gaussian mean shift
(exact d2 = mu' Sigma^-1 mu by one Cholesky per seed, then a 1-D quadrature), variance faults
(moment-matched LLR), F9 Poisson with a shared Gamma rate multiplier (negative multinomial, finite
sums), F8 binomial, tx_fade partners' RSSI. I is averaged over seeds, so each seed's environment is
treated as given. These are upper bounds for a detector that must also find onset and shape.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import linalg, stats
from scipy.special import expit, gammaln

from expand3.detect.config import DetectConfig
from expand3.detect.data import SeedData
from expand3.detect.features import context, ewma, sunlit
from expand3.detect.models import Ridge
from expand3.detect.peer import match_lag
from expand3.params import load_params

PARAMS = load_params(Path(__file__).resolve().parents[1] / "params")
SIG = PARAMS.links.signals
BATCHES = Path(os.environ.get("ORCA_BATCHES", "~/orca-batches")).expanduser()
FIT = range(0, 35)
EVAL = range(35, 70)
SATS = ["SAT-1", "SAT-2", "SAT-3", "SAT-4", "SAT-5", "SAT-6"]
BUS = {s: str(PARAMS.scenario.bus_of[s]) for s in SATS}
TARGETS = ["SAT-2", "SAT-3", "SAT-4"]
BUSY = ("gate_pass", "gate_take", "gate_slew", "gate_burn")
CFG = DetectConfig()
DT = 30.0
HALF_ORBIT_TICKS = 90
NODES = ["T_battery_K", "T_bus_px_K", "T_bus_nz_K", "T_payload_K", "T_bus_core_K"]
CTX_CH = [*NODES, "pack_v_V", "string_p_1_W"]
SPEC_CH = ["vres", "slr_1"]
DIRECT_CH = [f"{s}_{a}" for s in ("gyro", "mag", "gnss") for a in "xyz"]
ALL_CH = [*CTX_CH, *SPEC_CH, *DIRECT_CH]
REGIMES = ["ecl_quiet", "sun_charge", "sun_rest", "transition", "busy", "shed_uv", "sun_discharge",
           "reref"]
REST_A = 0.5
TRANSITION_TICKS = 2
PEER_LAGS = (-2, -1, 0, 1, 2)
PEER_TAUS_S = (1800.0, 7200.0)
FILL_GAP = 6
LMAX = 3600
PRE_S = 6 * 3600.0
HORIZONS_H = (0.5, 1.0, 2.0, 6.0, 24.0)
WINDOWS = (10, 60, 240)
PI = 0.1
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(100)
GH_W = GH_W / GH_W.sum()


# ---------------------------------------------------------------- channels and regimes
def st_rate(m):
    q = np.stack([m[f"st_q_{c}_meas"] for c in "wxyz"], axis=1)
    a, b = q[:-1], q[1:]
    d = np.stack([
        a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] - a[:, 2] * b[:, 3] + a[:, 3] * b[:, 2],
        a[:, 0] * b[:, 2] + a[:, 1] * b[:, 3] - a[:, 2] * b[:, 0] - a[:, 3] * b[:, 1],
        a[:, 0] * b[:, 3] - a[:, 1] * b[:, 2] + a[:, 2] * b[:, 1] - a[:, 3] * b[:, 0]], axis=1)
    s = np.sum(a * b, axis=1)
    out = np.full((len(q), 3), np.nan)
    out[1:] = 2.0 * np.where(s[:, None] < 0, -d, d) / np.diff(m["t_s"])[:, None]
    return out


def regimes(m, sun):
    """Fine regime per tick (index into REGIMES), from the vehicle's own telemetry. "reref" is
    where the spec re-references the SoC estimate to OCV^-1(V_meas): |I| below the rest current
    and the estimate in a steep part of the curve (Signals: SoC estimate)."""
    n = len(sun)
    busy = np.any([m[g] > 0 for g in BUSY], axis=0)
    i = m["pack_i_A_meas"]
    soc = m["soc_est_meas"]
    se = SIG["soc_estimate"]
    edge = np.zeros(n, bool)
    ch = np.flatnonzero(np.diff(sun) != 0)
    for k in range(-TRANSITION_TICKS, TRANSITION_TICKS + 1):
        edge[np.clip(ch + 1 + k, 0, n - 1)] = True
    r = np.where(sun > 0, np.where(i > REST_A, 1, np.where(i < -REST_A, 6, 2)), 0)
    reref = (np.abs(i) < se.rest_current_A) & ((soc > se.steep_soc_high) | (soc < se.steep_soc_low))
    r = np.where(reref, 7, r)
    r = np.where(edge, 3, r)
    r = np.where(busy, 4, r)
    r = np.where((m["gate_shed"] > 0) | (m["gate_uv_hold"] > 0), 5, r)
    return r.astype(int)


def raw_channels(d: SeedData, sat: str):
    """Measured inputs of every channel for one vehicle (before any fitted model)."""
    m = dict(d.measurements(sat))
    nan = np.full(len(m["t_s"]), np.nan)
    for a in "xyz":  # older batches predate the magnetometer and navigation columns
        for col in (f"mag_{a}_nT_meas", f"mag_model_{a}_nT_meas", f"nav_r_{a}_km_meas"):
            m.setdefault(col, nan)
    bus = PARAMS.bus(sat)
    t, X, _ = context(d, sat, CFG)
    sun = sunlit(m, CFG)
    ch = {c: m[f"{c}_meas"].copy() for c in CTX_CH}
    ch["string_p_1_W"] = np.where(sun > 0, ch["string_p_1_W"], np.nan)
    # Spec, Electrical power: V = OCV(SoC) + I R.
    # Signals: V + N(0, 0.04 V), I + 0.05 A + N(0, 0.02 A).
    ocv = np.interp(m["soc_est_meas"], bus.eps.ocv_curve.soc, bus.eps.ocv_curve.volts)
    ch["vres"] = m["pack_v_V_meas"] - ocv - m["pack_i_A_meas"] * bus.eps.R_pack_ohm
    # Spec: P_k = S s eta(T_k) max(0, n_k.s) A_k p_k o_k eta_ab, measured x (1 + N(0, 1 %)).
    # Every string faces +z, so the ratio to the others cancels S, s, attitude and eta_ab.
    ns = int(bus.eps.strings)
    p = np.stack([m[f"string_p_{k + 1}_W_meas"] for k in range(ns)])
    ok = np.all(p > 1.0, axis=0)
    others = p[1:].mean(axis=0)
    ch["slr_1"] = np.where(ok, np.log(np.where(ok, p[0], 1.0) / np.where(ok, others, 1.0)), np.nan)
    rate = st_rate(m)
    for k, a in enumerate("xyz"):
        g = m[f"imu_omega_{a}_rad_s_meas"]
        r = np.full(len(g), np.nan)
        r[1:] = g[:-1] - rate[1:, k]
        ch[f"gyro_{a}"] = r
        ch[f"mag_{a}"] = m[f"mag_{a}_nT_meas"] - m[f"mag_model_{a}_nT_meas"]
        ch[f"gnss_{a}"] = (m[f"gnss_r_{a}_km_meas"] - m[f"nav_r_{a}_km_meas"]) * 1e3
    extra = np.stack([m["pack_i_A_meas"], m["soc_est_meas"], m["T_battery_K_meas"]], axis=1)
    return dict(t=t, X=X, extra=extra, sun=sun, reg=regimes(m, sun), ch=ch, bus=BUS[sat],
                saa=m["gate_saa"].copy(), upsets=m["upsets"].copy(),
                tank=m["tank_kg_meas"].copy() if "tank_kg_meas" in m else None)


def design(p, c):
    return np.concatenate([p["X"], p["extra"]], axis=1) if c == "pack_v_V" else p["X"]


def residual(p, c, models):
    """Own-context residual of channel c (ridge for CTX_CH, fleet mean removed otherwise)."""
    x = p["ch"][c]
    if c in CTX_CH:
        return x - models["ctx"][(c, p["bus"])].predict(np.nan_to_num(design(p, c)))
    return x - models["mean"][(c, p["bus"])]


def fill(x, max_gap=None):
    """Linear interpolation over NaN gaps (gaps longer than max_gap stay NaN); edges held."""
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return x.copy()
    idx = np.arange(len(x))
    y = np.interp(idx, idx[ok], x[ok])
    if max_gap is not None:
        bad = ~ok
        lab = np.cumsum(np.concatenate([[0], np.diff(bad.astype(int)) == 1]))
        lens = np.bincount(lab[bad], minlength=lab.max() + 1) if bad.any() else np.zeros(1, int)
        long = np.zeros(len(x), bool)
        long[bad] = lens[lab[bad]] > max_gap
        y[long] = np.nan
    return y


def shifted(x, tau):
    y = np.full(len(x), np.nan)
    if tau >= 0:
        y[tau:] = x[:len(x) - tau]
    else:
        y[:tau] = x[-tau:]
    return y


def peer_features(res_peers, lags):
    cols = []
    for x, tau in zip(res_peers, lags, strict=True):
        f = fill(x, FILL_GAP)
        full = fill(x)
        for k in PEER_LAGS:
            cols.append(shifted(f, tau + k))
        for tau_s in PEER_TAUS_S:
            cols.append(shifted(ewma(np.nan_to_num(full), tau_s, DT), tau))
    return np.stack(cols, axis=1)


# ---------------------------------------------------------------- statistics
def kappa(z, nmax=40000, grid=2001):
    """sigma^2 J: Fisher information for the location of a standardized density (1 = Gaussian,
    larger = more information than the Gaussian of the same variance), from a Gaussian kernel
    density estimate (the smoothing biases it toward 1)."""
    rng = np.random.default_rng(0)
    if len(z) > nmax:
        z = rng.choice(z, nmax, replace=False)
    lo, hi = np.quantile(z, [0.0005, 0.9995])
    g = np.linspace(lo - 1, hi + 1, grid)
    f = stats.gaussian_kde(z, bw_method="silverman")(g)
    fp = np.gradient(f, g)
    ok = f > 1e-6 * f.max()
    return float(np.trapezoid(fp[ok] ** 2 / f[ok], g[ok]))


def shape(x, with_kappa=True):
    x = x[np.isfinite(x)]
    if len(x) < 200:
        return None
    z = (x - x.mean()) / x.std()
    out = dict(n=len(x), skew=float(stats.skew(z)), exkurt=float(stats.kurtosis(z)),
               p_gt4=float(np.mean(np.abs(z) > 4)))
    if with_kappa:
        out["kappa"] = kappa(z)
    return out


def tau_int(z, maxlag=400):
    z = z[np.isfinite(z)]
    z = z - z.mean()
    if len(z) < 2 * maxlag:
        return np.nan
    f = np.fft.rfft(z, 2 * len(z))
    ac = np.fft.irfft(f * np.conj(f))[:maxlag] / np.sum(z * z)
    neg = np.flatnonzero(ac < 0.05)
    cut = neg[0] if neg.size else maxlag
    return float(1.0 + 2.0 * ac[1:cut].sum())


def window_means(x, w):
    """Means of non-overlapping windows of w ticks with at least half the samples present,
    after removing the seed's own mean (the window statistic of a detector with a baseline)."""
    x = x - np.nanmean(x)
    n = len(x) // w * w
    b = x[:n].reshape(-1, w)
    ok = np.isfinite(b).sum(axis=1) >= max(1, w // 2)
    return np.nanmean(b[ok], axis=1) if ok.any() else np.zeros(0)


def masked_acf(zs, lmax):
    """ACF of standardized, within-seed demeaned series (NaN = missing), pooled over seeds,
    tapered and projected onto a positive semidefinite sequence."""
    num = np.zeros(lmax + 1)
    den = np.zeros(lmax + 1)
    for z in zs:
        m = np.isfinite(z).astype(float)
        v = np.nan_to_num(z)
        nfft = 1 << int(np.ceil(np.log2(2 * len(z))))
        fv, fm = np.fft.rfft(v, nfft), np.fft.rfft(m, nfft)
        num += np.fft.irfft(fv * np.conj(fv))[:lmax + 1]
        den += np.fft.irfft(fm * np.conj(fm))[:lmax + 1]
    acf = num / np.maximum(den, 1.0)
    acf /= acf[0]
    k = np.arange(lmax + 1)
    taper = np.where(k < lmax / 2, 1.0, 0.5 * (1 + np.cos(np.pi * (k - lmax / 2) / (lmax / 2))))
    a = acf * taper
    ext = np.concatenate([a, a[-2:0:-1]])
    s = np.fft.rfft(ext).real
    s = np.maximum(s, 1e-6 * s.max())
    a = np.fft.irfft(s, len(ext))[:lmax + 1]
    return a / a[0]


def offset_var(xs):
    """Variance of per-seed means beyond what the within-seed noise explains."""
    means, inner = [], []
    for x in xs:
        v = x[np.isfinite(x)]
        if len(v) > 200:
            means.append(v.mean())
            inner.append(v.var() * tau_int(v) / len(v))
    return float(max(0.0, np.var(means) - np.nanmean(inner))) if len(means) > 3 else 0.0


# ---------------------------------------------------------------- stage: null
def load_seed(args):
    batch, seed = args
    d = SeedData(batch / f"seed_{seed:05d}")
    per = {sat: raw_channels(d, sat) for sat in SATS}
    n = CFG.peer_lag_ticks
    lags = {(a, b): match_lag(per[a]["sun"][:n], per[b]["sun"][:n], HALF_ORBIT_TICKS)
            for a in SATS for b in SATS if a != b}
    return seed, per, lags


def stage_null(none: Path, out: Path):
    seeds = [*FIT, *EVAL]
    with ProcessPoolExecutor(12) as ex:
        data = {s: (per, lags) for s, per, lags in ex.map(load_seed, [(none, s) for s in seeds])}
    models = {"ctx": {}, "mean": {}, "peer": {}}
    for c in ALL_CH:
        for bus in "AB":
            xs, ys = [], []
            for s in FIT:
                for sat in SATS:
                    if BUS[sat] != bus:
                        continue
                    p = data[s][0][sat]
                    y = p["ch"][c]
                    ok = np.isfinite(y)
                    xs.append(design(p, c)[ok][::3])
                    ys.append(y[ok][::3])
            if c in CTX_CH:
                models["ctx"][(c, bus)] = Ridge(1.0).fit(np.concatenate(xs), np.concatenate(ys))
            else:
                models["mean"][(c, bus)] = float(np.mean(np.concatenate(ys)))
    res = {s: {sat: {c: residual(data[s][0][sat], c, models) for c in ALL_CH} for sat in SATS}
           for s in seeds}
    report, null = {}, {}
    for c in ALL_CH:
        for tgt in TARGETS:
            peers = [p for p in SATS if p != tgt]
            F, Y = [], []
            for s in FIT:
                per, lags = data[s]
                f = peer_features([res[s][p][c] for p in peers], [lags[(tgt, p)] for p in peers])
                x = res[s][tgt][c]
                ok = np.isfinite(x) & np.all(np.isfinite(f), axis=1)
                F.append(f[ok])
                Y.append(x[ok])
            pm = Ridge(1.0).fit(np.concatenate(F), np.concatenate(Y))
            models["peer"][(c, tgt)] = pm
            xs, es, regs = [], [], []
            for s in EVAL:
                per, lags = data[s]
                f = peer_features([res[s][p][c] for p in peers], [lags[(tgt, p)] for p in peers])
                x = res[s][tgt][c]
                pred = pm.predict(np.nan_to_num(f))
                e = np.where(np.all(np.isfinite(f), axis=1), x - pred, np.nan)
                e = np.where(np.isfinite(x), e, np.nan)
                xs.append(x)
                es.append(e)
                regs.append(per[tgt]["reg"])
            r = {}
            for name, ser in (("x", xs), ("e", es)):
                cat, rc = np.concatenate(ser), np.concatenate(regs)
                ok = np.isfinite(cat)
                r[f"{name}_all"] = shape(cat)
                r[f"{name}_reg"] = {REGIMES[k]: shape(cat[rc == k]) for k in range(len(REGIMES))}
                r[f"{name}_tau"] = float(np.nanmedian([tau_int(v) for v in ser]))
                r[f"{name}_win"] = {}
                for w in WINDOWS:
                    wm = np.concatenate([window_means(v, w) for v in ser])
                    r[f"{name}_win"][w] = shape(wm, with_kappa=False) if len(wm) > 200 else None
                # Regime spreads are within-seed: the per-seed offset enters Sigma once, as `off`.
                dm = np.concatenate([v - np.nanmean(v) for v in ser])
                sd = np.full(len(REGIMES), np.nan)
                for k in range(len(REGIMES)):
                    v = dm[ok & (rc == k)]
                    if len(v) > 200:
                        sd[k] = v.std()
                sd = np.where(np.isfinite(sd), sd, np.nanmax(sd))
                zs = [(v - np.nanmean(v)) / sd[rg] for v, rg in zip(ser, regs, strict=True)]
                r[f"{name}_z_all"] = shape(np.concatenate(zs))
                r[f"{name}_z_win"] = {}
                for w in WINDOWS:
                    wm = np.concatenate([window_means(v, w) for v in zs])
                    r[f"{name}_z_win"][w] = shape(wm, with_kappa=False) if len(wm) > 200 else None
                null[(c, tgt, name)] = dict(sd=sd, acf=masked_acf(zs, LMAX), off=offset_var(ser))
                r[f"{name}_var"] = float(np.nanvar(cat))
                r[f"{name}_off_frac"] = null[(c, tgt, name)]["off"] / r[f"{name}_var"]
            r["schur_var_ratio"] = r["e_var"] / r["x_var"]
            report[f"{c}|{tgt}"] = r
            print(c, tgt,
                  f"kurt x {r['x_all']['exkurt']:.2f} e {r['e_all']['exkurt']:.2f} | "
                  f"z-kurt x {r['x_z_all']['exkurt']:.2f} e {r['e_z_all']['exkurt']:.2f} | "
                  f"z-kappa x {r['x_z_all']['kappa']:.2f} e {r['e_z_all']['kappa']:.2f} | "
                  f"schur {r['schur_var_ratio']:.3f}", flush=True)
    cnt = {}
    for sat in SATS:
        disp = []
        for s in EVAL:
            p = data[s][0][sat]
            u = p["upsets"][p["saa"] > 0]
            if len(u) > 10 and u.mean() > 0:
                disp.append(u.var() / u.mean())
        cnt[sat] = float(np.mean(disp))
    report["_upset_dispersion"] = cnt
    with open(out, "wb") as fh:
        pickle.dump(dict(models=models, null=null, report=report), fh)
    Path(str(out).replace(".pkl", "_report.json")).write_text(json.dumps(report, default=float))


# ---------------------------------------------------------------- stage: pairing check
def pair_one(args):
    batch, none, seed = args
    fd, nd = SeedData(batch / f"seed_{seed:05d}"), SeedData(none / f"seed_{seed:05d}")
    ft = pq.read_table(fd.directory / "measurements.parquet").to_pandas()
    nt = pq.read_table(nd.directory / "measurements.parquet").to_pandas()
    cols = [c for c in ft.columns if c in nt.columns and c != "sat_id" and not c.endswith("_true")]
    v, t_arm = fd.vehicle, fd.t_arm_s
    out = {"seed": seed, "vehicle": v, "other_diff": 0, "prearm_diff": 0, "cols": len(cols),
           "missing_cols": len([c for c in nt.columns if c not in ft.columns])}
    for sat in SATS:
        a = ft[ft.sat_id == sat][cols].to_numpy(float)
        b = nt[nt.sat_id == sat][cols].to_numpy(float)
        if a.shape != b.shape:
            out["shape_mismatch"] = True
            return out
        diff = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        if sat != v:
            out["other_diff"] += int(diff.sum())
        elif t_arm is not None:
            pre = ft[ft.sat_id == sat]["t_s"].to_numpy() < t_arm
            out["prearm_diff"] += int(diff[pre].sum())
    return out


def stage_pair(none: Path, batches: list[str]):
    summary = {}
    for b in batches:
        bp = BATCHES / b
        seeds = sorted(int(p.name.split("_")[1]) for p in bp.glob("seed_*")
                       if (p / "manifest.json").is_file() and (none / p.name).is_dir())[:10]
        with ProcessPoolExecutor(10) as ex:
            rows = list(ex.map(pair_one, [(bp, none, s) for s in seeds]))
        summary[b] = dict(n=len(rows), other_diff=sum(r["other_diff"] for r in rows),
                          prearm_diff=sum(r["prearm_diff"] for r in rows),
                          shape_mismatch=sum(bool(r.get("shape_mismatch")) for r in rows),
                          missing_cols=max(r["missing_cols"] for r in rows) if rows else None)
        print(b, summary[b], flush=True)
    return summary


# ---------------------------------------------------------------- MI closed forms
def h2(p):
    p = np.clip(p, 1e-300, 1 - 1e-16)
    q = np.clip(1 - p, 1e-300, 1)
    return -(p * np.log2(p) + q * np.log2(q))


def h_llr(pi, m1, v1, m0, v0):
    """H(y | x) [bits] when the log likelihood ratio is N(m1, v1) under y=1 and N(m0, v0) under
    y=0. Exact for a Gaussian mean shift (m1 = -m0 = d2/2, v1 = v0 = d2); a moment match for a
    variance change."""
    lo = np.log(pi / (1 - pi))
    l1 = m1 + np.sqrt(max(v1, 0.0)) * GH_X
    l0 = m0 + np.sqrt(max(v0, 0.0)) * GH_X
    h1 = np.sum(GH_W * h2(expit(lo + l1)))
    h0 = np.sum(GH_W * h2(expit(lo + l0)))
    return float(pi * h1 + (1 - pi) * h0)


def h_gauss(pi, d2):
    return h_llr(pi, d2 / 2, d2, -d2 / 2, d2) if np.isfinite(d2) else 0.0


def cov_window(nl, rg, idx):
    """Covariance of the residual at tick indices idx: regime scales x within-seed ACF, plus the
    per-seed offset (a constant the window shares)."""
    s = nl["sd"][rg[idx]]
    lag = np.minimum(np.abs(idx[:, None] - idx[None, :]), LMAX)
    return s[:, None] * nl["acf"][lag] * s[None, :] + nl["off"]


def d2_curve(mu, cov, hp):
    """d2 = mu' cov^-1 mu over growing prefixes of the window: the prefix Cholesky factor is the
    leading block of the full one, so d2(prefix p) = sum of the first p squares of L^-1 mu."""
    jit = 1e-9 * np.mean(np.diag(cov))
    L = linalg.cholesky(cov + jit * np.eye(len(cov)), lower=True)
    w = linalg.solve_triangular(L, mu, lower=True)
    cs = np.cumsum(w ** 2)
    return np.array([cs[min(p, len(cs)) - 1] if p > 0 else 0.0 for p in hp])


def var_llr_moments(S0, D):
    """Mean and variance of the LLR of N(0, S0 + D) against N(0, S0), under each hypothesis."""
    S1 = S0 + D
    c0, c1 = linalg.cho_factor(S0), linalg.cho_factor(S1)
    eye = np.eye(len(S0))
    A = linalg.cho_solve(c0, eye) - linalg.cho_solve(c1, eye)
    ld = 2 * (np.sum(np.log(np.diag(c1[0]))) - np.sum(np.log(np.diag(c0[0]))))
    AS0, AS1 = A @ S0, A @ S1
    return (0.5 * np.trace(AS1) - 0.5 * ld, 0.5 * np.sum(AS1 * AS1.T),
            0.5 * np.trace(AS0) - 0.5 * ld, 0.5 * np.sum(AS0 * AS0.T))


def poisson_gamma_h(pi, f, lam, e_post, e_base, k):
    """H(y | N_post, N_base) [bits]. Spec F9: upsets are Poisson. N_post ~ Poisson(f m lam e_post)
    (f = 1 under no fault) on the vehicle's SAA exposure since arm; N_base ~ Poisson(m lam e_base)
    on its pre-arm exposure and, with the swarm, the peers' exposure. The rate multiplier m is
    shared, m ~ Gamma(k, rate k) (coefficient of variation 1/sqrt(k)); k = inf is a known rate.
    The joint law is negative multinomial, so H is a finite double sum."""
    mu1, mu0, mb = f * lam * e_post, lam * e_post, lam * e_base
    if not np.isfinite(k):
        amax = int(stats.poisson.ppf(1 - 1e-12, max(mu1, 1e-9))) + 20
        a = np.arange(amax + 1)
        j1, j0 = stats.poisson.pmf(a, mu1), stats.poisson.pmf(a, mu0)
    else:
        amax = int(stats.nbinom.ppf(1 - 1e-12, k, k / (k + max(mu1, 1e-9)))) + 20
        bmax = int(stats.nbinom.ppf(1 - 1e-12, k, k / (k + mb))) + 20 if mb > 0 else 0
        A, B = np.meshgrid(np.arange(amax + 1), np.arange(bmax + 1), indexing="ij")

        def lj(mu_a):
            return (gammaln(k + A + B) - gammaln(k) - gammaln(A + 1) - gammaln(B + 1)
                    + k * np.log(k) + A * np.log(max(mu_a, 1e-300)) + B * np.log(max(mb, 1e-300))
                    - (k + A + B) * np.log(k + mu_a + mb))
        j1, j0 = np.exp(lj(mu1)), np.exp(lj(mu0))
    pj = pi * j1 + (1 - pi) * j0
    post = pi * j1 / np.maximum(pj, 1e-300)
    return float(np.sum(pj * h2(post)))


def binom_h(pi, p, n):
    """H(y | drops on n expected messages) [bits]: spec F8, each message drops independently
    with probability p under the fault, never nominally."""
    q = (1 - p) ** n
    pn = pi * q + (1 - pi)
    return float(pn * h2(pi * q / pn))


# ---------------------------------------------------------------- stage: mi
FAULTS = [  # name, batch, channels, kind
    ("F1", "m9_forced/F1", ["vres"], "gauss"),
    ("F1 (ridge V)", "m9_forced/F1", ["pack_v_V"], "gauss"),
    ("F2", "m9_forced/F2", ["slr_1"], "gauss"),
    ("F7 (bus_px)", "m9_forced/F7", ["T_bus_px_K"], "gauss"),
    ("F7 (core)", "m9_forced/F7", ["T_bus_core_K"], "gauss"),
    ("F7 (battery)", "m9_forced/F7", ["T_battery_K"], "gauss"),
    ("F5", "m9_forced/F5", ["tank"], "white"),
    ("F6", "m9_forced/F6", ["tank"], "white"),
    ("F8", "m9_forced/F8", [], "binom"),
    ("F9", "m9_forced/F9", [], "poisson"),
    ("gyro_bias step", "gyro/gyro_bias_step", ["gyro_x", "gyro_y", "gyro_z"], "gauss"),
    ("gyro_bias ramp", "gyro/gyro_bias_ramp", ["gyro_x", "gyro_y", "gyro_z"], "gauss"),
    ("gyro_scale step", "gyro/gyro_scale_step", ["gyro_x", "gyro_y", "gyro_z"], "gauss"),
    ("gyro_noise ramp", "gyro/gyro_noise_ramp", ["gyro_x", "gyro_y", "gyro_z"], "var"),
    ("mag_bias step", "multisat/mag_bias_step", ["mag_x", "mag_y", "mag_z"], "gauss"),
    ("mag_bias ramp", "multisat/mag_bias_ramp", ["mag_x", "mag_y", "mag_z"], "gauss"),
    ("mag_coupling step", "multisat/mag_coupling_step", ["mag_x", "mag_y", "mag_z"], "gauss"),
    ("mag_noise ramp", "multisat/mag_noise_ramp", ["mag_x", "mag_y", "mag_z"], "var"),
    ("gnss_jump step", "multisat/gnss_jump_step", ["gnss_x", "gnss_y", "gnss_z"], "gauss"),
    ("gnss_noise ramp", "multisat/gnss_noise_ramp", ["gnss_x", "gnss_y", "gnss_z"], "var"),
    ("tx_fade step", "multisat/tx_fade_step", [], "rssi"),
]
SEEDS_PER_FAULT = 30
VAR_MAX = 1200  # variance faults: dense algebra on at most the last VAR_MAX samples
OWN_WHITE = {"gyro": SIG["imu"].sigma_rad_s, "mag": float(SIG["magnetometer"].sigma_nT),
             "gnss": SIG["gnss"].pos_sigma_m}


def fault_theta(fd: SeedData, v: str):
    t = pq.read_table(fd.directory / "truth.parquet", columns=["sat_id", "fault_theta_0"])
    sel = np.asarray(t.column("sat_id")) == v
    return np.asarray(t.column("fault_theta_0"))[sel].astype(float)


def mi_seed(args):
    _name, batch, chans, kind, seed, none, models, null, kap = args
    fd, nd = SeedData(BATCHES / batch / f"seed_{seed:05d}"), SeedData(none / f"seed_{seed:05d}")
    v, t_arm = fd.vehicle, fd.t_arm_s
    if v is None or t_arm is None:
        return None
    t = nd.measurements(v)["t_s"]
    i_arm = int(np.searchsorted(t, t_arm))
    i0 = max(0, int(np.searchsorted(t, t_arm - PRE_S)))
    hab = [min(i_arm + int(h * 3600 / DT), len(t)) for h in HORIZONS_H]
    valid_h = [i_arm + int(h * 3600 / DT) <= len(t) for h in HORIZONS_H]
    out = {"seed": seed, "vehicle": v, "h": {}}
    hy = float(h2(PI))
    if kind in ("gauss", "var", "white"):
        pf, pn = raw_channels(fd, v), raw_channels(nd, v)
        nh = len(HORIZONS_H)
        d2x, d2e = np.zeros(nh), np.zeros(nh)
        mom = {nm: np.zeros((nh, 4)) for nm in ("x", "e")}
        for c in chans:
            if kind == "white":  # Signals: tank gauge = propellant + N(0, 0.1 kg), burns only
                if pf["tank"] is None:
                    return None
                mu = pf["tank"] - pn["tank"]
                sig = SIG["tank_gauge"].sigma_kg
                for j, p in enumerate(hab):
                    seg = mu[i_arm:p]
                    d2x[j] += float(np.nansum(seg ** 2) / sig ** 2)
                d2e[:] = d2x
                continue
            rf, rn = residual(pf, c, models), residual(pn, c, models)
            idx = np.arange(i0, hab[-1])
            idx = idx[np.isfinite(rf[idx]) & np.isfinite(rn[idx])]
            if len(idx) < 5:
                continue
            hp = [int(np.searchsorted(idx, p)) for p in hab]
            rg = pn["reg"]
            if kind == "gauss":
                mu = (rf - rn)[idx]
                d2x += d2_curve(mu, cov_window(null[(c, v, "x")], rg, idx), hp)
                d2e += d2_curve(mu, cov_window(null[(c, v, "e")], rg, idx), hp)
            else:  # variance fault: extra own white noise (m(t)^2 - 1) sigma_w^2
                mt = np.nan_to_num(fault_theta(fd, v), nan=1.0)
                extra = (mt ** 2 - 1.0) * OWN_WHITE[c.split("_")[0]] ** 2
                for j, p in enumerate(hp):
                    sub = idx[:p][-VAR_MAX:]
                    if len(sub) < 2 or extra[sub].max() <= 0:
                        continue
                    D = np.diag(extra[sub])
                    for nm in ("x", "e"):
                        S0 = cov_window(null[(c, v, nm)], rg, sub)
                        S0 = S0 + 1e-9 * np.mean(np.diag(S0)) * np.eye(len(S0))
                        mom[nm][j] += np.array(var_llr_moments(S0, D))
        for j, hh in enumerate(HORIZONS_H):
            if not valid_h[j]:
                continue
            if kind == "var":
                if mom["x"][j][1] <= 0:
                    hx = he = hy
                else:
                    hx, he = h_llr(PI, *mom["x"][j]), h_llr(PI, *mom["e"][j])
                out["h"][hh] = dict(Ix=hy - hx, Ixs=hy - he, dI=hx - he,
                                    d2x=float(mom["x"][j][0] - mom["x"][j][2]),
                                    d2e=float(mom["e"][j][0] - mom["e"][j][2]))
            else:
                hx, he = h_gauss(PI, d2x[j]), h_gauss(PI, d2e[j])
                out["h"][hh] = dict(d2x=float(d2x[j]), d2e=float(d2e[j]), Ix=hy - hx,
                                    Ixs=hy - he, dI=hx - he)
                if kind == "gauss":  # small-signal non-Gaussian correction d2 -> kappa d2
                    kx = np.mean([kap[(c, v, "x")] for c in chans])
                    ke = np.mean([kap[(c, v, "e")] for c in chans])
                    hxk, hek = h_gauss(PI, kx * d2x[j]), h_gauss(PI, ke * d2e[j])
                    out["h"][hh].update(Ix_k=hy - hxk, Ixs_k=hy - hek, dI_k=hxk - hek)
    elif kind == "binom":
        p = float(fd.labels["magnitude"])
        xl = pq.read_table(nd.directory / "crosslink.parquet",
                           columns=["t_s", "sender", "receiver", "sent"]).to_pandas()
        xl = xl[xl["sent"]]
        for j, hh in enumerate(HORIZONS_H):
            if not valid_h[j]:
                continue
            w = xl[(xl.t_s >= t_arm) & (xl.t_s < t[hab[j] - 1] + DT)]
            n_r, n_s = int((w.receiver == v).sum()), int((w.sender == v).sum())
            hx, he = binom_h(PI, p, n_r), binom_h(PI, p, n_r + n_s)
            out["h"][hh] = dict(n_own=n_r, n_swarm=n_s, Ix=hy - hx, Ixs=hy - he, dI=hx - he)
    elif kind == "poisson":
        lam = PARAMS.links.cdh_upsets.lambda_in_per_s
        f = float(fd.labels["magnitude"])
        saa = {s: nd.measurements(s)["gate_saa"] > 0 for s in SATS}
        out["cv"] = {}
        for cv in (0.0, 0.3, 1.0):
            k = np.inf if cv == 0 else 1 / cv ** 2
            rows = {}
            for j, hh in enumerate(HORIZONS_H):
                if not valid_h[j]:
                    continue
                p = hab[j]
                e_post = saa[v][i_arm:p].sum() * DT
                e_pre = saa[v][i0:i_arm].sum() * DT
                e_sw = sum(saa[s][i0:p].sum() for s in SATS if s != v) * DT
                hx = poisson_gamma_h(PI, f, lam, e_post, e_pre, k)
                he = poisson_gamma_h(PI, f, lam, e_post, e_pre + e_sw, k)
                rows[hh] = dict(e_post_s=float(e_post), Ix=hy - hx, Ixs=hy - he, dI=hx - he)
            out["cv"][cv] = rows
        out["h"] = out["cv"][0.3]
    elif kind == "rssi":
        # Supplement: tx_fade shows only in the RSSI its partners read, RSSI + N(0, 0.5 dB).
        sig = PARAMS.links.link_budget.rssi_sigma_dB
        cols = ["t_s", "sender", "receiver", "rssi_dBm"]
        cf = pq.read_table(fd.directory / "crosslink.parquet", columns=cols).to_pandas()
        cn = pq.read_table(nd.directory / "crosslink.parquet", columns=cols).to_pandas()
        mu = cf["rssi_dBm"].to_numpy() - cn["rssi_dBm"].to_numpy()
        sel = (cn["sender"].to_numpy() == v) & np.isfinite(mu)
        ts = cn["t_s"].to_numpy()
        for j, hh in enumerate(HORIZONS_H):
            if not valid_h[j]:
                continue
            w = sel & (ts >= t_arm) & (ts < t[hab[j] - 1] + DT)
            d2s = float(np.sum(mu[w] ** 2) / sig ** 2)
            he = h_gauss(PI, d2s)
            out["h"][hh] = dict(d2x=0.0, d2e=d2s, Ix=0.0, Ixs=hy - he, dI=hy - he)
    return out


def stage_mi(none: Path, nullpkl: Path, out: Path):
    with open(nullpkl, "rb") as fh:
        nl = pickle.load(fh)
    models, null = nl["models"], nl["null"]
    rep = nl["report"]
    kap = {(c, v, nm): max(1.0, rep[f"{c}|{v}"][f"{nm}_z_all"]["kappa"])
           for c in ALL_CH for v in TARGETS for nm in ("x", "e")}
    result = {"_H_y_bits": float(h2(PI)), "_pi": PI}
    only = set(sys.argv[5].split(",")) if len(sys.argv) > 5 else None
    for name, batch, chans, kind in FAULTS:
        if only and name not in only:
            continue
        bp = BATCHES / batch
        seeds = sorted(int(p.name.split("_")[1]) for p in bp.glob("seed_*")
                       if (p / "manifest.json").is_file() and (none / p.name).is_dir())
        seeds = seeds[:SEEDS_PER_FAULT]
        args = [(name, batch, chans, kind, s, none, models, null, kap) for s in seeds]
        with ProcessPoolExecutor(8) as ex:
            rows = [r for r in ex.map(mi_seed, args) if r is not None and r["h"]]
        summ = {}
        for hh in HORIZONS_H:
            vals = [r["h"][hh] for r in rows if hh in r["h"]]
            if not vals:
                continue
            # sizes (d2, counts, exposure) as medians, information as means over seeds
            summ[hh] = {k: float(np.median([v[k] for v in vals]))
                        if k.startswith(("d2", "n_", "e_"))
                        else float(np.mean([v[k] for v in vals])) for k in vals[0]}
            summ[hh]["n_seeds"] = len(vals)
        if kind == "poisson":
            summ["cv"] = {
                cv: {hh: {k: float(np.mean([r["cv"][cv][hh][k] for r in rows
                                            if hh in r["cv"][cv]]))
                          for k in ("Ix", "Ixs", "dI")} for hh in HORIZONS_H}
                for cv in (0.0, 0.3, 1.0)}
        result[name] = summ
        shown = {str(k): v for k, v in summ.items() if k != "cv"}
        print(name, json.dumps(shown, default=float)[:1200], flush=True)
        out.write_text(json.dumps(result, default=float))


def selftest() -> None:
    """The closed forms against their limits and against Monte Carlo."""
    rng = np.random.default_rng(1)
    pi = 0.1
    assert abs(h_gauss(pi, 0.0) - h2(pi)) < 1e-9
    assert h_gauss(pi, 400.0) < 1e-9
    d2, n = 2.0, 400000
    y = rng.random(n) < pi
    x = rng.normal(np.where(y, np.sqrt(d2), 0.0), 1.0)
    post = expit(np.log(pi / (1 - pi)) + np.sqrt(d2) * x - d2 / 2)
    assert abs(np.mean(h2(post)) - h_gauss(pi, d2)) < 3e-3
    m1, v1, m0, v0 = var_llr_moments(np.eye(50), np.eye(50))
    x0, x1 = rng.normal(size=(20000, 50)), rng.normal(size=(20000, 50)) * np.sqrt(2.0)

    def llr(z):
        return 0.25 * np.sum(z**2, axis=1) - 25.0 * np.log(2.0)
    assert np.allclose([m1, v1, m0, v0], [llr(x1).mean(), llr(x1).var(), llr(x0).mean(),
                                          llr(x0).var()], rtol=0.03)
    lam = 1 / 1800
    known = [poisson_gamma_h(pi, 50, lam, 3000, e, np.inf) for e in (2000, 20000)]
    assert known[0] == known[1]
    own, swarm = (poisson_gamma_h(pi, 3, lam, 3000, e, 11.1) for e in (2000, 20000))
    assert poisson_gamma_h(pi, 3, lam, 3000, 0, np.inf) < swarm < own
    assert abs(poisson_gamma_h(pi, 1, lam, 3000, 2000, 4.0) - h2(pi)) < 1e-9
    assert abs(binom_h(pi, 0.9, 0) - h2(pi)) < 1e-12
    idx = np.arange(200)
    cov = 0.9 ** np.abs(idx[:, None] - idx[None, :])
    mu = rng.normal(size=200)
    got = d2_curve(mu, cov, [50, 200])
    want = [mu[:50] @ np.linalg.solve(cov[:50, :50], mu[:50]), mu @ np.linalg.solve(cov, mu)]
    assert np.allclose(got, want, rtol=1e-6)
    print("selftest ok")


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "selftest":
        selftest()
    elif stage == "null":
        stage_null(Path(sys.argv[2]), Path(sys.argv[3]))
    elif stage == "pair":
        print(json.dumps(stage_pair(Path(sys.argv[2]), sys.argv[3:])))
    elif stage == "mi":
        stage_mi(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
