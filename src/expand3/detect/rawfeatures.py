"""Window features of the measured signals for the raw-signal classifier.

The inputs are the ones the physics residuals read: the four string powers, the pack voltage and
current, the thermistors, the upset counts inside the SAA, the crosslink delivery flags, the
per-slew wheel telemetry and the dark current of the latest take. They are summarised over
trailing windows ending at each decision time, with the exogenous context of `features.context`
beside them. The features carry no nominal values and no thresholds: every number is a plain
window statistic, so what separates a fault from nominal is left to the classifier to learn.
Nothing here reads truth.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from expand3.detect.config import DetectConfig
from expand3.detect.data import SeedData
from expand3.detect.detectors import N_WHEELS, wheel_fits
from expand3.detect.features import N_STRINGS, context, sunlit

F64 = NDArray[np.float64]


class Windows:
    """Sums of a per-tick series over the trailing windows ending at each grid time."""

    def __init__(self, t: F64, grid: F64, width_s: float, min_samples: int) -> None:
        self.hi = np.searchsorted(t, grid, side="right")
        self.lo = np.searchsorted(t, grid - width_s, side="right")
        self.min_samples = min_samples

    def sum(self, x: F64) -> F64:
        c = np.concatenate([[0.0], np.cumsum(np.where(np.isfinite(x), x, 0.0))])
        return np.asarray(c[self.hi] - c[self.lo], dtype=np.float64)

    def count(self, ok: NDArray[np.bool_]) -> F64:
        return self.sum(ok.astype(np.float64))

    def mean(self, x: F64, mask: NDArray[np.bool_] | None = None) -> F64:
        ok = np.isfinite(x) if mask is None else np.isfinite(x) & mask
        n = self.count(ok)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = self.sum(np.where(ok, x, 0.0)) / n
        return np.where(n >= self.min_samples, out, np.nan)

    def slope(self, y: F64, x: F64, mask: NDArray[np.bool_]) -> F64:
        """Least-squares slope of y on x over the masked ticks of each window."""
        ok = np.isfinite(x) & np.isfinite(y) & mask
        xs, ys = np.where(ok, x, 0.0), np.where(ok, y, 0.0)
        n, sx, sy = self.count(ok), self.sum(xs), self.sum(ys)
        sxx, sxy = self.sum(xs * xs), self.sum(xs * ys)
        with np.errstate(divide="ignore", invalid="ignore"):
            var = sxx - sx * sx / n
            out = (sxy - sx * sy / n) / var
        return np.where((n >= self.min_samples) & (var > 0.0), out, np.nan)


def _at(grid: F64, t: F64, x: F64) -> F64:
    """Value of a per-tick series at the last tick at or before each grid time."""
    return x[np.maximum(np.searchsorted(t, grid, side="right") - 1, 0)]


def _label(name: str, w: float) -> str:
    return f"{name}@{w:g}"


def _strings(m: dict[str, F64], sun: F64, ws: list[Windows], cfg: DetectConfig,
             wnames: list[float], cols: dict[str, F64]) -> None:
    p = np.stack([m[f"string_p_{i + 1}_W_meas"] for i in range(N_STRINGS)])
    p = p[~np.all(np.isnan(p), axis=1)]
    total = p.sum(axis=0)
    lit = sun > 0.0
    ratios = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(p.shape[0]):
            others = (total - p[i]) / (p.shape[0] - 1)
            ratios.append(np.where(others > cfg.raw_ratio_floor, p[i] / others, np.nan))
    reference = np.nanquantile(total / p.shape[0], cfg.f2_sunlit_quantile)
    for w, win in zip(wnames, ws, strict=True):
        r = np.stack([win.mean(x, lit) for x in ratios])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # windows with no sunlit sample
            cols[_label("string_ratio_min", w)] = np.nanmin(r, axis=0)
            cols[_label("string_ratio_max", w)] = np.nanmax(r, axis=0)
        cols[_label("string_power_rel", w)] = win.mean(total / p.shape[0], lit) / reference


def _pack(m: dict[str, F64], sun: F64, ws: list[Windows], wnames: list[float],
          cols: dict[str, F64]) -> None:
    v, i = m["pack_v_V_meas"], m["pack_i_A_meas"]
    lit = sun > 0.0
    for w, win in zip(wnames, ws, strict=True):
        cols[_label("pack_v_lit", w)] = win.mean(v, lit)
        cols[_label("pack_v_dark", w)] = win.mean(v, ~lit)
        cols[_label("pack_i_lit", w)] = win.mean(i, lit)
        cols[_label("pack_i_dark", w)] = win.mean(i, ~lit)
        cols[_label("pack_v_on_i", w)] = win.slope(v, i, np.ones(v.shape, dtype=bool))


def _thermistors(m: dict[str, F64], ws: list[Windows], wnames: list[float],
                 cols: dict[str, F64]) -> None:
    for c in sorted(k for k in m if k.startswith("T_") and k.endswith("_K_meas")):
        for w, win in zip(wnames, ws, strict=True):
            cols[_label(c, w)] = win.mean(m[c])


def _counts(m: dict[str, F64], t: F64, grid: F64, cfg: DetectConfig,
            cols: dict[str, F64]) -> None:
    """Upsets, SAA time and the upset rate inside the SAA, over longer windows."""
    saa = m["gate_saa"] > 0.0
    for w in cfg.raw_count_windows_s:
        win = Windows(t, grid, w, cfg.raw_min_samples)
        ticks, upsets = win.count(saa), win.sum(np.where(saa, m["upsets"], 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = upsets / ticks
        cols[_label("saa_ticks", w)] = ticks
        cols[_label("upsets", w)] = upsets
        cols[_label("upset_per_saa_tick", w)] = np.where(ticks >= cfg.raw_min_samples, rate, np.nan)


def _crosslink(d: SeedData, sat: str, grid: F64, cfg: DetectConfig,
               cols: dict[str, F64]) -> None:
    """Delivered fraction of the messages a vehicle sends and of those sent to it."""
    table = d.table("crosslink", ["t_s", "sender", "receiver", "sent", "delivered"])
    t = np.asarray(table.column("t_s"), dtype=np.float64)
    sent = np.asarray(table.column("sent"), dtype=np.float64)
    got = np.asarray(table.column("delivered"), dtype=np.float64)
    for side in ("sender", "receiver"):
        sel = np.asarray(table.column(side)) == sat
        order = np.argsort(t[sel], kind="stable")
        ts, ss, gs = t[sel][order], sent[sel][order], got[sel][order]
        for w in cfg.raw_windows_s:
            win = Windows(ts, grid, w, cfg.raw_min_samples)
            n = win.sum(ss)
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = win.sum(gs * ss) / n
            cols[_label(f"link_delivered_{side}", w)] = np.where(n >= cfg.raw_min_samples,
                                                                 frac, np.nan)


def _wheels(d: SeedData, sat: str, grid: F64, cfg: DetectConfig, cols: dict[str, F64]) -> None:
    """Least-squares k and c of each wheel over the latest finished slew, held until the next."""
    names = ["slew", "t_s", *[f"h_{j + 1}_Nms" for j in range(N_WHEELS)],
             *[f"u_{j + 1}_Nm" for j in range(N_WHEELS)]]
    c = d.sat_columns("wheel_steps", sat, names)
    ends, fits = [], []
    for s in np.unique(c["slew"]):
        sel = c["slew"] == s
        t = np.asarray(c["t_s"], dtype=np.float64)[sel]
        order = np.argsort(t, kind="stable")
        row = []
        for j in range(N_WHEELS):
            h = np.asarray(c[f"h_{j + 1}_Nms"], dtype=np.float64)[sel][order]
            u = np.asarray(c[f"u_{j + 1}_Nm"], dtype=np.float64)[sel][order]
            k, cc = wheel_fits(t[order], h, u, cfg.wheel_min_rows, cfg.wheel_min_conditioning)
            row += [k[-1], cc[-1]]
        ends.append(t[order][-1])
        fits.append(row)
    order = np.argsort(ends)
    ends_a = np.asarray(ends, dtype=np.float64)[order]
    fit_a = np.asarray(fits, dtype=np.float64).reshape(len(ends), 2 * N_WHEELS)[order]
    idx = np.searchsorted(ends_a, grid, side="right") - 1
    for j in range(N_WHEELS):
        for q, name in enumerate(("k", "c")):
            col = np.full(grid.shape, np.nan)
            have = idx >= 0
            col[have] = fit_a[idx[have], 2 * j + q]
            cols[f"wheel_{name}_{j + 1}"] = col


def _dark(d: SeedData, sat: str, grid: F64, cols: dict[str, F64]) -> None:
    """Dark current and payload temperature of the latest take, held after it."""
    c = d.sat_columns("takes", sat, ["t_take_s", "payload_K", "dark_current_e_s_px"])
    order = np.argsort(c["t_take_s"])
    t = np.asarray(c["t_take_s"], dtype=np.float64)[order]
    dark = np.asarray(c["dark_current_e_s_px"], dtype=np.float64)[order]
    temp = np.asarray(c["payload_K"], dtype=np.float64)[order]
    idx = np.searchsorted(t, grid, side="right") - 1
    have = idx >= 0
    for name, x in (("dark_log2", np.log2(np.maximum(dark, np.finfo(np.float64).tiny))),
                    ("dark_payload_K", temp)):
        col = np.full(grid.shape, np.nan)
        col[have] = x[idx[have]] if x.size else np.nan
        cols[name] = col


def raw_features(d: SeedData, sat: str, grid: F64, cfg: DetectConfig
                 ) -> tuple[F64, list[str]]:
    """(len(grid), n_features) window features of one vehicle at the decision times `grid`."""
    m = d.measurements(sat)
    t = m["t_s"]
    sun = sunlit(m, cfg)
    wnames = list(cfg.raw_windows_s)
    ws = [Windows(t, grid, w, cfg.raw_min_samples) for w in wnames]
    cols: dict[str, F64] = {}
    _strings(m, sun, ws, cfg, wnames, cols)
    _pack(m, sun, ws, wnames, cols)
    _thermistors(m, ws, wnames, cols)
    _counts(m, t, grid, cfg, cols)
    _crosslink(d, sat, grid, cfg, cols)
    _wheels(d, sat, grid, cfg, cols)
    _dark(d, sat, grid, cols)
    tc, xc, cnames = context(d, sat, cfg)
    for j, n in enumerate(cnames):
        cols[f"ctx_{n}"] = _at(grid, tc, xc[:, j])
    names = list(cols)
    return np.stack([cols[n] for n in names], axis=1), names
