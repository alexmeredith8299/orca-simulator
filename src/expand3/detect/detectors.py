"""Detectors for the easier faults: F2, F8, F9, F10 and F3/F4.

Each reads measurement-side files only (measurements, crosslink, takes, wheel_steps) and turns
them into GLR score traces. Nominal values (means, rates, scales) are fitted from fault-free
tune data rather than read from the simulator, except the dark-current model of F10, which the
spec gives as an equation (Model: imager) and which is read from the params directory.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from expand3.detect.base import Detector, Trace
from expand3.detect.data import SeedData
from expand3.detect.glr import binomial_glr, gaussian_glr, poisson_glr
from expand3.params import load_params

F64 = NDArray[np.float64]
N_STRINGS = 4
N_WHEELS = 4


def _moments(x: F64) -> list[float]:
    return [float(x.size), float(x.sum()), float((x**2).sum())]


def _pool(summaries: list[dict[str, Any]], floor: float) -> dict[str, tuple[float, float]]:
    """Pool [n, sum, sumsq] triples by key into (mean, sigma), with sigma at least `floor`."""
    acc: dict[str, list[float]] = {}
    for s in summaries:
        for key, m in s.items():
            tot = acc.setdefault(key, [0.0, 0.0, 0.0])
            for i in range(3):
                tot[i] += m[i]
    out = {}
    for key, (n, sx, sxx) in acc.items():
        if n < 2:
            continue
        mean = sx / n
        out[key] = (mean, max(float(np.sqrt(max(sxx / n - mean**2, 0.0))), floor))
    return out


class StringPowerDetector(Detector):
    """F2: each string's power over the mean of the other strings, while sunlit.

    A string is sunlit-gated on the mean of all present strings exceeding a fraction of its
    own high quantile, so the gate needs no knowledge of the orbit. The GLR looks for the ratio
    falling below its fault-free mean; the string that fires is the alarm detail.
    """

    name = "f2"
    fault_ids = ("F2",)

    def _ratios(self, d: SeedData, sat: str) -> tuple[F64, dict[int, F64]]:
        cols = [f"string_p_{i + 1}_W_meas" for i in range(N_STRINGS)]
        c = d.sat_columns("measurements", sat, ["t_s", *cols])
        power = np.stack([np.asarray(c[x], dtype=np.float64) for x in cols])
        present = np.flatnonzero(~np.all(np.isnan(power), axis=1))
        t = np.asarray(c["t_s"], dtype=np.float64)
        if present.size < 3:
            return t[:0], {}
        p = power[present]
        mean_all = p.mean(axis=0)
        cfg = self.cfg
        sun = mean_all > cfg.f2_sunlit_fraction * np.nanquantile(mean_all, cfg.f2_sunlit_quantile)
        others = (p.sum(axis=0)[None, :] - p) / (present.size - 1)
        sun &= np.all(others > 0, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):  # night: 0 / 0, not used
            return t[sun], {int(k): (p[j] / others[j])[sun] for j, k in enumerate(present)}

    def summarize(self, d: SeedData) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for sat in d.sat_ids:
            t, ratios = self._ratios(d, sat)
            ok = d.fault_free(sat, t)
            for k, r in ratios.items():
                out[f"{sat}|{k}"] = _moments(r[ok])
        return out

    def fit(self, summaries: list[dict[str, Any]]) -> dict[str, Any]:
        return {k: list(v) for k, v in _pool(summaries, self.cfg.f2_sigma_floor).items()}

    def traces(self, d: SeedData, nominal: dict[str, Any]) -> list[Trace]:
        out = []
        for sat in d.sat_ids:
            t, ratios = self._ratios(d, sat)
            for k, r in ratios.items():
                if f"{sat}|{k}" not in nominal:
                    continue
                mu, sigma = nominal[f"{sat}|{k}"]
                score = gaussian_glr(r, mu, sigma, self.cfg.f2_window, side="down")
                out.append(Trace(sat, "F2", t, score,
                                 np.full(t.shape, f"string_{k + 1}", dtype="U16")))
        return out


class CrosslinkDropDetector(Detector):
    """F8: binomial GLR on the delivered flags of every link, then a vehicle score.

    A drop under F8 hits every link the faulted vehicle sends or receives on, so the score of a
    vehicle is the sum of its incident link statistics, and only the vehicle with the largest
    score can alarm. That keeps the neighbours of the faulted vehicle quiet.
    """

    name = "f8"
    fault_ids = ("F8",)

    def _messages(self, d: SeedData) -> dict[str, Any]:
        table = d.table("crosslink", ["t_s", "sender", "receiver", "sent", "delivered"])
        return {c: np.asarray(table.column(c)) for c in table.column_names}

    def summarize(self, d: SeedData) -> dict[str, Any]:
        m = self._messages(d)
        sent = m["sent"].astype(bool)
        # a message is fault-free when neither end is faulted at that time
        free = np.ones(sent.shape, dtype=bool)
        for sat in d.sat_ids:
            touches = (m["sender"] == sat) | (m["receiver"] == sat)
            free &= ~touches | d.fault_free(sat, m["t_s"])
        use = sent & free
        return {"drops": float(np.sum(use & ~m["delivered"].astype(bool))),
                "sent": float(np.sum(use))}

    def fit(self, summaries: list[dict[str, Any]]) -> dict[str, Any]:
        sent = sum(s["sent"] for s in summaries)
        drops = sum(s["drops"] for s in summaries)
        return {"p0": max(drops / sent if sent else 0.0, self.cfg.f8_p0_floor)}

    def traces(self, d: SeedData, nominal: dict[str, Any]) -> list[Trace]:
        m = self._messages(d)
        grid = np.unique(m["t_s"]).astype(np.float64)
        sats = d.sat_ids
        total = {s: np.zeros(grid.size) for s in sats}
        for a in sats:
            for b in sats:
                sel = (m["sender"] == a) & (m["receiver"] == b) & m["sent"].astype(bool)
                if a == b or not sel.any():
                    continue
                t = m["t_s"][sel].astype(np.float64)
                order = np.argsort(t, kind="stable")
                drop = (~m["delivered"][sel].astype(bool)).astype(np.float64)[order]
                score = binomial_glr(drop, np.ones_like(drop), nominal["p0"], self.cfg.f8_window)
                at = np.searchsorted(t[order], grid, side="right") - 1
                held = np.where(at >= 0, score[np.maximum(at, 0)], 0.0)
                total[a] += held
                total[b] += held
        best = np.max(np.stack([total[s] for s in sats]), axis=0)
        return [Trace(s, "F8", grid, np.where(total[s] >= best, total[s], 0.0)) for s in sats]


class UpsetRateDetector(Detector):
    """F9: Poisson GLR on edge-computer upsets per tick while inside the SAA."""

    name = "f9"
    fault_ids = ("F9",)

    def _exposure(self, d: SeedData, sat: str) -> tuple[F64, F64, float]:
        c = d.sat_columns("measurements", sat, ["t_s", "gate_saa", "upsets"])
        t = np.asarray(c["t_s"], dtype=np.float64)
        inside = np.asarray(c["gate_saa"]).astype(bool)
        dt = float(np.median(np.diff(t)))
        return t[inside], np.asarray(c["upsets"], dtype=np.float64)[inside], dt

    def summarize(self, d: SeedData) -> dict[str, Any]:
        upsets = seconds = 0.0
        for sat in d.sat_ids:
            t, n, dt = self._exposure(d, sat)
            ok = d.fault_free(sat, t)
            upsets += float(n[ok].sum())
            seconds += float(ok.sum()) * dt
        return {"upsets": upsets, "exposure_s": seconds}

    def fit(self, summaries: list[dict[str, Any]]) -> dict[str, Any]:
        exposure = sum(s["exposure_s"] for s in summaries)
        rate = sum(s["upsets"] for s in summaries) / exposure if exposure else 0.0
        return {"lambda_in_per_s": max(rate, self.cfg.f9_lambda_floor_per_s)}

    def traces(self, d: SeedData, nominal: dict[str, Any]) -> list[Trace]:
        out = []
        for sat in d.sat_ids:
            t, n, dt = self._exposure(d, sat)
            expected = np.full(t.shape, nominal["lambda_in_per_s"] * dt)
            out.append(Trace(sat, "F9", t, poisson_glr(n, expected, self.cfg.f9_window)))
        return out


class DarkCurrentDetector(Detector):
    """F10: measured dark current over D0 2^((Tp - Tref)/dT) at each take (Model: imager).

    The statistic is a Gaussian GLR on log2 of that ratio, one sample per take, stamped at take
    start (where the simulator evaluates D).
    """

    name = "f10"
    fault_ids = ("F10",)

    def _log2_ratio(self, d: SeedData, sat: str) -> tuple[F64, F64]:
        assert self.params_dir is not None
        dark = load_params(self.params_dir).links.imager_dark_current
        c = d.sat_columns("takes", sat, ["t_take_s", "payload_K", "dark_current_e_s_px"])
        model = dark.D0_e_s_px * 2.0 ** ((c["payload_K"] - dark.T_ref_K) / dark.delta_T_K)
        return (np.asarray(c["t_take_s"], dtype=np.float64),
                np.log2(np.asarray(c["dark_current_e_s_px"], dtype=np.float64) / model))

    def summarize(self, d: SeedData) -> dict[str, Any]:
        xs = [x[d.fault_free(s, t)] for s in d.sat_ids for t, x in [self._log2_ratio(d, s)]]
        return {"all": _moments(np.concatenate(xs))}

    def fit(self, summaries: list[dict[str, Any]]) -> dict[str, Any]:
        mu, sigma = _pool(summaries, self.cfg.f10_sigma_floor)["all"]
        return {"mu": mu, "sigma": sigma}

    def traces(self, d: SeedData, nominal: dict[str, Any]) -> list[Trace]:
        out = []
        for sat in d.sat_ids:
            t, x = self._log2_ratio(d, sat)
            if t.size:
                score = gaussian_glr(x, nominal["mu"], nominal["sigma"], 1)
                out.append(Trace(sat, "F10", t, score))
        return out


def wheel_fits(t: F64, h: F64, u: F64, min_rows: int,
               min_conditioning: float) -> tuple[F64, F64]:
    """Causal least-squares fit of k and c for one wheel over one slew (Model: actuators).

    The spec's update h(t + dt) = h(t) - (k u + c h(t)) dt gives, for each step,
    (h_prev - h) / dt = k u + c h_prev. Row i of the result is the fit from the steps up to i.
    Rows before `min_rows` steps, or where u and h_prev are nearly collinear, are NaN.
    """
    n = t.size
    k_hat = np.full(n, np.nan)
    c_hat = np.full(n, np.nan)
    if n < 2:
        return k_hat, c_hat
    dt = np.diff(t)
    x1, x2, y = u[1:], h[:-1], (h[:-1] - h[1:]) / dt
    suu, suh, shh = np.cumsum(x1 * x1), np.cumsum(x1 * x2), np.cumsum(x2 * x2)
    suy, shy = np.cumsum(x1 * y), np.cumsum(x2 * y)
    det = suu * shh - suh**2
    with np.errstate(divide="ignore", invalid="ignore"):
        good = det > min_conditioning * suu * shh
        k = (shh * suy - suh * shy) / det
        c = (suu * shy - suh * suy) / det
    good &= np.arange(1, n) >= min_rows
    k_hat[1:] = np.where(good, k, np.nan)
    c_hat[1:] = np.where(good, c, np.nan)
    return k_hat, c_hat


class WheelFitDetector(Detector):
    """F3 and F4: per-wheel fit of the torque constant k and friction c within each slew.

    The score is the largest deviation over wheels from the fault-free fit, in units of the
    fault-free spread (floored). F4 scores k and F3 scores c; the wheel is the alarm detail.
    """

    name = "wheels"
    fault_ids = ("F3", "F4")

    def _slews(self, d: SeedData, sat: str) -> list[tuple[F64, F64, F64]]:
        names = ["slew", "t_s", *[f"h_{j + 1}_Nms" for j in range(N_WHEELS)],
                 *[f"u_{j + 1}_Nm" for j in range(N_WHEELS)]]
        c = d.sat_columns("wheel_steps", sat, names)
        out = []
        for s in np.unique(c["slew"]):
            sel = c["slew"] == s
            t = np.asarray(c["t_s"], dtype=np.float64)[sel]
            order = np.argsort(t, kind="stable")
            h = np.stack([c[f"h_{j + 1}_Nms"][sel] for j in range(N_WHEELS)], axis=1)[order]
            u = np.stack([c[f"u_{j + 1}_Nm"][sel] for j in range(N_WHEELS)], axis=1)[order]
            out.append((t[order], h.astype(np.float64), u.astype(np.float64)))
        return out

    def _fit_all(self, t: F64, h: F64, u: F64) -> tuple[F64, F64]:
        cfg = self.cfg
        fits = [wheel_fits(t, h[:, j], u[:, j], cfg.wheel_min_rows, cfg.wheel_min_conditioning)
                for j in range(N_WHEELS)]
        return np.stack([f[0] for f in fits], axis=1), np.stack([f[1] for f in fits], axis=1)

    def summarize(self, d: SeedData) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for sat in d.sat_ids:
            for t, h, u in self._slews(d, sat):
                if not d.fault_free(sat, t[-1:])[0]:
                    continue
                k, c = self._fit_all(t, h, u)
                for j in range(N_WHEELS):
                    for name, x in (("k", k), ("c", c)):
                        v = x[:, j][np.isfinite(x[:, j])][-1:]
                        if v.size:
                            m = out.setdefault(f"{name}|{j}", [0.0, 0.0, 0.0])
                            m[:] = [a + b for a, b in zip(m, _moments(v), strict=True)]
        return out

    def fit(self, summaries: list[dict[str, Any]]) -> dict[str, Any]:
        return {k: list(v) for k, v in _pool(summaries, self.cfg.wheel_sigma_floor).items()}

    def traces(self, d: SeedData, nominal: dict[str, Any]) -> list[Trace]:
        out = []
        for sat in d.sat_ids:
            ts: list[F64] = []
            score: dict[str, list[F64]] = {"k": [], "c": []}
            which: dict[str, list[F64]] = {"k": [], "c": []}
            for t, h, u in self._slews(d, sat):
                ts.append(t)
                fits = dict(zip(("k", "c"), self._fit_all(t, h, u), strict=True))
                for name, x in fits.items():
                    z = np.stack([np.abs(x[:, j] - nominal[f"{name}|{j}"][0])
                                  / nominal[f"{name}|{j}"][1] for j in range(N_WHEELS)], axis=1)
                    z = np.where(np.isfinite(z), z, 0.0)
                    score[name].append(z.max(axis=1))
                    which[name].append(z.argmax(axis=1).astype(np.float64))
            if not ts:
                continue
            t_all = np.concatenate(ts)
            for name, fid in (("c", "F3"), ("k", "F4")):
                detail = np.array([f"wheel_{int(j) + 1}" for j in np.concatenate(which[name])],
                                  dtype="U16")
                out.append(Trace(sat, fid, t_all, np.concatenate(score[name]), detail))
        return out


DETECTORS: dict[str, type[Detector]] = {
    "f2": StringPowerDetector,
    "f8": CrosslinkDropDetector,
    "f9": UpsetRateDetector,
    "f10": DarkCurrentDetector,
    "wheels": WheelFitDetector,
}
