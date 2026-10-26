"""Generalized likelihood ratio change detectors over a sliding window, and threshold tuning.

Each detector tests a known fault-free parameter against an unknown shifted one on the most
recent n samples, maximized over n = 1..W (a sliding-window GLR). The statistic at sample i is
the log likelihood ratio at the maximum likelihood shifted parameter, so it is zero when the data
do not favor a shift. Samples that do not apply (night, outside the SAA, no message sent) are
removed by the caller before windowing, so the window is in samples of exposure.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]
Side = Literal["both", "up", "down"]


def _window_max(sums: Sequence[F64], window: int, llr: Callable[..., F64]) -> F64:
    """Max over n = 1..window of llr(sum of the last n of each input)."""
    n_samples = len(sums[0])
    cums = [np.concatenate([[0.0], np.cumsum(s)]) for s in sums]
    best = np.zeros(n_samples)
    idx = np.arange(1, n_samples + 1)
    for n in range(1, min(window, n_samples) + 1):
        sel = idx >= n
        win = [c[idx[sel]] - c[idx[sel] - n] for c in cums]
        best[sel] = np.maximum(best[sel], llr(*win, n=np.full(sel.sum(), float(n))))
    return best


def gaussian_glr(x: F64, mu0: float, sigma: float, window: int, side: Side = "both") -> F64:
    """Mean-shift GLR for x ~ N(mu, sigma^2) against mu0: (sum of x - mu0)^2 / (2 sigma^2 n)."""

    def llr(total: F64, *, n: F64) -> F64:
        out = total**2 / (2.0 * sigma**2 * n)
        if side == "up":
            out = np.where(total > 0, out, 0.0)
        elif side == "down":
            out = np.where(total < 0, out, 0.0)
        return np.asarray(out, dtype=np.float64)

    return _window_max([np.asarray(x, dtype=np.float64) - mu0], window, llr)


def slope_glr(r: F64, x: F64, sigma: float, window: int) -> F64:
    """GLR for a residual that grows with a known regressor, r = beta x + noise, against beta = 0:
    (sum of x r)^2 / (2 sigma^2 sum of x^2) over the window. A resistance change shows up this
    way, because the voltage residual is proportional to the current."""

    def llr(xr: F64, xx: F64, *, n: F64) -> F64:
        del n
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(xx > 0, xr**2 / (2.0 * sigma**2 * xx), 0.0)
        return np.asarray(out, dtype=np.float64)

    r64, x64 = np.asarray(r, dtype=np.float64), np.asarray(x, dtype=np.float64)
    return _window_max([x64 * r64, x64 * x64], window, llr)


def _xlogx_ratio(a: F64, b: F64) -> F64:
    """a ln(a / b), zero where a is zero."""
    safe = np.where(a > 0, a, 1.0)
    return np.where(a > 0, a * np.log(safe / b), 0.0)


def poisson_glr(counts: F64, expected: F64, window: int, side: Side = "up") -> F64:
    """Rate-shift GLR for Poisson counts with known fault-free expectation per sample.

    With N counts against M expected in the window: N ln(N / M) - N + M.
    """

    def llr(n_obs: F64, m_exp: F64, *, n: F64) -> F64:
        del n
        out = _xlogx_ratio(n_obs, m_exp) - n_obs + m_exp
        if side == "up":
            out = np.where(n_obs > m_exp, out, 0.0)
        elif side == "down":
            out = np.where(n_obs < m_exp, out, 0.0)
        return np.asarray(out, dtype=np.float64)

    return _window_max([np.asarray(counts, dtype=np.float64),
                        np.asarray(expected, dtype=np.float64)], window, llr)


def binomial_glr(k: F64, n_trials: F64, p0: float, window: int) -> F64:
    """One-sided (success probability rises above p0) GLR on K successes in N trials:
    K ln(K / (N p0)) + (N - K) ln((N - K) / (N (1 - p0)))."""

    def llr(k_sum: F64, n_sum: F64, *, n: F64) -> F64:
        del n
        out = _xlogx_ratio(k_sum, n_sum * p0) + _xlogx_ratio(n_sum - k_sum, n_sum * (1.0 - p0))
        return np.asarray(np.where(k_sum > n_sum * p0, out, 0.0), dtype=np.float64)

    return _window_max([np.asarray(k, dtype=np.float64),
                        np.asarray(n_trials, dtype=np.float64)], window, llr)


def episodes(score: F64, h: float) -> NDArray[np.intp]:
    """Indices where the score first exceeds h after being at or below it (or at the start)."""
    above = np.asarray(score) > h
    prev = np.concatenate([[False], above[:-1]])
    return np.flatnonzero(above & ~prev)


def tune_threshold(null: Sequence[F64], exposure_days: float, target_far: float,
                   floor: float) -> float:
    """Smallest threshold h >= floor for which fault-free scores give at most target_far
    alarm episodes per vehicle-day, by bisection on the (non-increasing) episode count."""
    allowed = target_far * exposure_days

    def count(h: float) -> int:
        return sum(len(episodes(s, h)) for s in null)

    top = max([float(np.max(s)) for s in null if len(s)] + [floor])
    if count(floor) <= allowed:
        return floor
    lo, hi = floor, top
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if count(mid) <= allowed:
            hi = mid
        else:
            lo = mid
    return hi
