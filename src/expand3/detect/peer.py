"""Peer differencing: SAT-2 and SAT-3 against SAT-1, matched by orbit phase.

The reference sees the same environment a few seconds earlier or later, so a difference cancels
what the context features cannot model. The lag comes from the agreement of the sunlit patterns
over the first hours of the run (the smallest lag among the best). Ticks where either vehicle is
slewing, burning, taking or in a pass are masked, and for the power signals both must be sunlit.
Node-temperature differences are regressed on the difference of the two vehicles' context
features, since they differ a little in how they warm up. SAA upsets are compared by exposure
instead of by time: the k-th tick either vehicle spends in the SAA, with a binomial test on which
vehicle took the upsets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from expand3.detect.base import Detector, Trace
from expand3.detect.config import DetectConfig
from expand3.detect.data import SeedData
from expand3.detect.features import context, node_columns, sunlit
from expand3.detect.glr import binomial_glr, gaussian_glr, slope_glr
from expand3.detect.models import Ridge

F64 = NDArray[np.float64]
BoolArr = NDArray[np.bool_]
N_STRINGS = 4
BUSY = ("gate_pass", "gate_burn", "gate_take", "gate_slew")


@dataclass
class Matched:
    """Target ticks `i` and the reference ticks `j` at the same orbit phase."""

    target: dict[str, F64]
    ref: dict[str, F64]
    i: NDArray[np.intp]
    j: NDArray[np.intp]
    ok: BoolArr  # both quiet (no pass, burn, take or slew) and same sunlit state
    sun: BoolArr  # both sunlit


def match_lag(a: F64, b: F64, max_lag: int) -> int:
    """Lag tau such that a[i] resembles b[i - tau]; the smallest |tau| among the best."""
    best = (-1.0, 0, 0)
    for tau in range(-max_lag, max_lag + 1):
        x = a[max(tau, 0):]
        y = b[max(-tau, 0):]
        n = min(len(x), len(y))
        if n < 10:
            continue
        cand = (float(np.mean(x[:n] == y[:n])), -abs(tau), tau)
        best = max(best, cand)
    return best[2]


def match(d: SeedData, ref: str, tgt: str, cfg: DetectConfig) -> Matched:
    mt, mr = d.measurements(tgt), d.measurements(ref)
    st, sr = sunlit(mt, cfg), sunlit(mr, cfg)
    n = min(len(st), cfg.peer_lag_ticks)
    tau = match_lag(st[:n], sr[:n], cfg.peer_max_lag_ticks)
    i = np.arange(len(st))
    j = i - tau
    valid = (j >= 0) & (j < len(sr))
    jj = np.clip(j, 0, len(sr) - 1)
    busy_t = np.any([mt[g] > 0 for g in BUSY], axis=0)
    busy_r = np.any([mr[g] > 0 for g in BUSY], axis=0)[jj]
    same = st == sr[jj]
    return Matched(mt, mr, i, jj, valid & ~busy_t & ~busy_r & same,
                   valid & (st > 0) & (sr[jj] > 0))


class _Peer(Detector):
    family = "peer"

    def _pairs(self, d: SeedData) -> list[tuple[str, str]]:
        ref = self.cfg.peer_reference
        return [(ref, t) for t in self.cfg.peer_targets if t in d.sat_ids and ref in d.sat_ids]

    def _clean(self, d: SeedData, tgt: str, mt: Matched) -> BoolArr:
        """Matched ticks where neither vehicle carries the fault."""
        t = mt.target["t_s"]
        return d.fault_free(tgt, t) & d.fault_free(self.cfg.peer_reference, mt.ref["t_s"][mt.j])


class PeerThermal(_Peer):
    """F7: the worst node's temperature difference to the reference, after a ridge fit of that
    difference on the difference of the two vehicles' context features (their sunlit histories
    and gates differ a little even in the same orbit)."""

    name = "peer_thermal"
    fault_ids = ("F7",)

    def _diff_context(self, d: SeedData, ref: str, tgt: str, mt: Matched) -> F64:
        _, xt, _ = context(d, tgt, self.cfg)
        _, xr, _ = context(d, ref, self.cfg)
        return xt - xr[mt.j]

    def summarize(self, d: SeedData) -> dict[str, Any]:
        out: dict[str, Any] = {}
        stride = self.cfg.learn_stride
        for ref, tgt in self._pairs(d):
            mt = match(d, ref, tgt, self.cfg)
            use = mt.ok & self._clean(d, tgt, mt)
            dx = self._diff_context(d, ref, tgt, mt)
            ys = {}
            for c in node_columns(mt.target):
                if c in mt.ref and not np.all(np.isnan(mt.ref[c])):
                    diff = mt.target[c] - mt.ref[c][mt.j]
                    ys[c] = diff
            use &= np.all([np.isfinite(y) for y in ys.values()], axis=0)
            out[tgt] = {"X": dx[use][::stride], "Y": {c: y[use][::stride] for c, y in ys.items()}}
        return out

    def fit(self, summaries: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for tgt in {k for s in summaries for k in s}:
            parts = [s[tgt] for s in summaries if tgt in s]
            x = np.concatenate([p["X"] for p in parts])
            for c in parts[0]["Y"]:
                y = np.concatenate([p["Y"][c] for p in parts])
                model = Ridge(self.cfg.learn_alpha).fit(x, y)
                sigma = max(float((y - model.predict(x)).std()), self.cfg.learn_sigma_floor)
                out[f"{tgt}|{c}"] = (model, sigma)
        return out

    def traces(self, d: SeedData, nominal: dict[str, Any]) -> list[Trace]:
        out = []
        for ref, tgt in self._pairs(d):
            mt = match(d, ref, tgt, self.cfg)
            dx = self._diff_context(d, ref, tgt, mt)
            cols = [k.split("|")[1] for k in nominal if k.startswith(f"{tgt}|")]
            t = mt.target["t_s"][mt.ok]
            scores = []
            for c in cols:
                model, sigma = nominal[f"{tgt}|{c}"]
                diff = mt.target[c] - mt.ref[c][mt.j]
                r = ((diff - model.predict(dx)) / sigma)[mt.ok]
                scores.append(gaussian_glr(np.nan_to_num(r), 0.0, 1.0, self.cfg.learn_window))
            if not scores:
                continue
            stack = np.stack(scores)
            names = np.array([c.removeprefix("T_").removesuffix("_K_meas") for c in cols],
                             dtype="U32")
            out.append(Trace(tgt, "F7", t, stack.max(axis=0), names[stack.argmax(axis=0)]))
        return out


def _volt_features(mt: Matched) -> F64:
    cols = ("pack_i_A_meas", "soc_est_meas", "T_battery_K_meas")
    return np.stack([mt.target[c] - mt.ref[c][mt.j] for c in cols], axis=1)


class PeerEps(_Peer):
    """F2: each string's power over the reference's, both sunlit. F1: the pack voltage difference
    after regressing out the current, SoC and battery-temperature differences, with the slope
    GLR against the target's current."""

    name = "peer_eps"
    fault_ids = ("F1", "F2")

    def summarize(self, d: SeedData) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for ref, tgt in self._pairs(d):
            mt = match(d, ref, tgt, self.cfg)
            clean = self._clean(d, tgt, mt)
            for k in range(N_STRINGS):
                c = f"string_p_{k + 1}_W_meas"
                if np.all(np.isnan(mt.target[c])) or np.all(np.isnan(mt.ref[c])):
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = mt.target[c] / mt.ref[c][mt.j]
                use = mt.ok & mt.sun & clean & np.isfinite(ratio)
                out[f"{tgt}|s{k}"] = [float(use.sum()), float(ratio[use].sum()),
                                      float((ratio[use] ** 2).sum())]
            use = mt.ok & clean
            out[f"{tgt}|v"] = {"X": _volt_features(mt)[use][::self.cfg.learn_stride],
                               "y": (mt.target["pack_v_V_meas"]
                                     - mt.ref["pack_v_V_meas"][mt.j])[use][::self.cfg.learn_stride]}
        return out

    def fit(self, summaries: list[dict[str, Any]]) -> dict[str, Any]:
        nominal: dict[str, Any] = {}
        keys = {k for s in summaries for k in s}
        for k in keys:
            parts = [s[k] for s in summaries if k in s]
            if k.endswith("|v"):
                x = np.concatenate([p["X"] for p in parts])
                y = np.concatenate([p["y"] for p in parts])
                model = Ridge(self.cfg.learn_alpha).fit(x, y)
                sigma = max(float((y - model.predict(x)).std()), self.cfg.learn_sigma_floor)
                nominal[k] = (model, sigma)
            else:
                n, sx, sxx = (float(sum(p[i] for p in parts)) for i in range(3))
                mean = sx / n
                nominal[k] = (mean, max(float(np.sqrt(max(sxx / n - mean**2, 0.0))),
                                        self.cfg.f2_sigma_floor))
        return nominal

    def traces(self, d: SeedData, nominal: dict[str, Any]) -> list[Trace]:
        out = []
        for ref, tgt in self._pairs(d):
            mt = match(d, ref, tgt, self.cfg)
            t_all = mt.target["t_s"]
            for k in range(N_STRINGS):
                key = f"{tgt}|s{k}"
                if key not in nominal:
                    continue
                c = f"string_p_{k + 1}_W_meas"
                mean, sigma = nominal[key]
                use = mt.ok & mt.sun
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = (mt.target[c] / mt.ref[c][mt.j])[use]
                score = gaussian_glr(np.nan_to_num((ratio - mean) / sigma), 0.0, 1.0,
                                     self.cfg.f2_window, side="down")
                out.append(Trace(tgt, "F2", t_all[use], score,
                                 np.full(int(use.sum()), f"string_{k + 1}", dtype="U16")))
            model, sigma = nominal[f"{tgt}|v"]
            use = mt.ok
            resid = (mt.target["pack_v_V_meas"] - mt.ref["pack_v_V_meas"][mt.j]
                     - model.predict(_volt_features(mt)))[use]
            out.append(Trace(tgt, "F1", t_all[use],
                             slope_glr(np.nan_to_num(resid), mt.target["pack_i_A_meas"][use],
                                       sigma, self.cfg.learn_window)))
        return out


class PeerUpsets(_Peer):
    """F9: of the upsets the two vehicles take in matched SAA exposure, the target's share is
    tested against its fault-free share."""

    name = "peer_saa"
    fault_ids = ("F9",)

    def _counts(self, d: SeedData, ref: str, tgt: str) -> tuple[F64, F64, F64]:
        mt, mr = d.measurements(tgt), d.measurements(ref)
        in_t, in_r = mt["gate_saa"] > 0, mr["gate_saa"] > 0
        c_t, c_r = mt["upsets"][in_t], mr["upsets"][in_r]
        k = min(len(c_t), len(c_r))
        return mt["t_s"][in_t][:k], c_t[:k], c_r[:k]

    def summarize(self, d: SeedData) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for ref, tgt in self._pairs(d):
            t, c_t, c_r = self._counts(d, ref, tgt)
            ok = d.fault_free(tgt, t)
            key = f"{tgt}"
            out[key] = [float(c_t[ok].sum()) + out.get(key, [0.0, 0.0])[0],
                        float(c_t[ok].sum() + c_r[ok].sum()) + out.get(key, [0.0, 0.0])[1]]
        return out

    def fit(self, summaries: list[dict[str, Any]]) -> dict[str, Any]:
        out = {}
        for k in {k for s in summaries for k in s}:
            mine, total = (float(sum(s[k][i] for s in summaries if k in s)) for i in range(2))
            out[k] = float(np.clip(mine / total if total else 0.5, 0.2, 0.8))
        return out

    def traces(self, d: SeedData, nominal: dict[str, Any]) -> list[Trace]:
        out = []
        for ref, tgt in self._pairs(d):
            t, c_t, c_r = self._counts(d, ref, tgt)
            trials = c_t + c_r
            use = trials > 0
            score = binomial_glr(c_t[use], trials[use], nominal[tgt], self.cfg.peer_saa_window)
            out.append(Trace(tgt, "F9", t[use], score))
        return out


PEER: dict[str, type[Detector]] = {
    "peer_thermal": PeerThermal,
    "peer_eps": PeerEps,
    "peer_saa": PeerUpsets,
}
