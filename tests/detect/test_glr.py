"""Generic GLR change detectors and threshold tuning."""

import numpy as np
import pytest

from expand3.detect.glr import (
    binomial_glr,
    episodes,
    gaussian_glr,
    poisson_glr,
    tune_threshold,
    variance_glr,
)


def test_gaussian_matches_closed_form() -> None:
    x = np.concatenate([np.zeros(30), 2.0 * np.ones(5)])
    s = gaussian_glr(x, mu0=0.0, sigma=0.5, window=10)
    assert np.all(s[:30] == 0)
    # n shifted samples of size d: n d^2 / (2 sigma^2)
    assert s[34] == pytest.approx(5 * 4.0 / (2 * 0.25))


def test_gaussian_window_forgets_old_samples() -> None:
    x = np.concatenate([3.0 * np.ones(3), np.zeros(20)])
    s = gaussian_glr(x, 0.0, 1.0, window=5)
    assert s[2] > 0
    assert s[-1] == 0


def test_gaussian_sides() -> None:
    x = -np.ones(6)
    assert gaussian_glr(x, 0.0, 1.0, 6, side="up")[-1] == 0
    assert gaussian_glr(x, 0.0, 1.0, 6, side="down")[-1] == pytest.approx(3.0)
    assert gaussian_glr(x, 0.0, 1.0, 6, side="both")[-1] == pytest.approx(3.0)


def test_poisson_matches_closed_form() -> None:
    counts = np.array([0, 0, 4, 5], dtype=float)
    expected = np.full(4, 0.5)
    s = poisson_glr(counts, expected, window=4)
    # the best window is the last two samples: 9 counts against 1 expected
    n, m = 9.0, 1.0
    assert s[-1] == pytest.approx(n * np.log(n / m) - n + m)
    assert poisson_glr(np.zeros(4), expected, 4)[-1] == 0


def test_poisson_below_expectation_is_zero_one_sided() -> None:
    assert poisson_glr(np.zeros(5), np.ones(5), 5, side="up")[-1] == 0
    assert poisson_glr(np.zeros(5), np.ones(5), 5, side="down")[-1] == pytest.approx(5.0)


def test_binomial_matches_closed_form() -> None:
    n = np.full(10, 2.0)
    k = np.array([0] * 5 + [2] * 5, dtype=float)
    s = binomial_glr(k, n, p0=0.1, window=10)
    kk, nn = 10.0, 10.0  # the last five samples: every trial succeeded
    kk, nn = 10.0, 10.0
    want = kk * np.log(kk / (nn * 0.1))
    assert s[-1] == pytest.approx(want)
    assert binomial_glr(np.zeros(10), n, 0.1, 10)[-1] == 0


def test_binomial_all_failures_is_finite() -> None:
    s = binomial_glr(np.ones(4), np.ones(4), 0.01, 4)
    assert np.isfinite(s[-1])
    assert s[-1] == pytest.approx(4 * np.log(1 / 0.01))


def test_episodes_mark_upward_crossings() -> None:
    s = np.array([0, 5, 6, 0, 0, 7, 0, 8, 8], dtype=float)
    assert episodes(s, 4.0).tolist() == [1, 5, 7]
    assert episodes(s, 10.0).tolist() == []
    assert episodes(np.array([9.0, 9.0]), 4.0).tolist() == [0]


def test_tune_threshold_meets_target_and_floor() -> None:
    rng = np.random.default_rng(0)
    null = [np.maximum(rng.exponential(1.0, 5000) - 1.5, 0.0) for _ in range(4)]
    days = 10.0
    h = tune_threshold(null, days, target_far=0.5, floor=0.0)
    n = sum(len(episodes(s, h)) for s in null)
    assert n / days <= 0.5
    assert h > 0
    assert tune_threshold(null, days, 0.5, floor=1e3) == 1e3
    assert tune_threshold([np.zeros(10)], 1.0, 0.1, floor=2.0) == 2.0


def test_variance_glr_is_zero_at_unit_variance_and_matches_the_closed_form() -> None:
    z = np.concatenate([np.ones(10), 4.0 * np.ones(5)])  # squared residuals: variance 4 for 5
    s = variance_glr(z, window=10, dof=3)
    assert np.all(s[:10] == 0)
    # n dof / 2 (r - 1 - ln r) with r = 4
    assert s[14] == pytest.approx(0.5 * 3 * 5 * (4.0 - 1.0 - np.log(4.0)))
    assert np.all(variance_glr(0.25 * np.ones(8), 5) == 0)  # a smaller variance is no alarm
