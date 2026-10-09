"""M1: named seed substreams, per-seed scenario draws, and the run clock."""

from datetime import UTC, datetime

import numpy as np
import pytest

from expand3.params import Params, ParamView
from expand3.timebase import Clock, draw_seed, rng
from expand3.units import S_PER_HOUR


def test_same_seed_same_draws(params: Params) -> None:
    assert draw_seed(params, 5) == draw_seed(params, 5)
    assert draw_seed(params, 5) != draw_seed(params, 6)


def test_substreams_deterministic_and_order_independent() -> None:
    a1 = rng(1, "noise.gnss.SAT-2").standard_normal(4)
    rng(1, "draw.fault").standard_normal(100)  # consuming another stream changes nothing
    a2 = rng(1, "noise.gnss.SAT-2").standard_normal(4)
    np.testing.assert_array_equal(a1, a2)


def test_substreams_independent() -> None:
    x = rng(3, "noise.gnss.SAT-2").standard_normal(20000)
    y = rng(3, "noise.gnss.SAT-3").standard_normal(20000)
    z = rng(4, "noise.gnss.SAT-2").standard_normal(20000)
    assert abs(np.corrcoef(x, y)[0, 1]) < 0.03
    assert abs(np.corrcoef(x, z)[0, 1]) < 0.03
    assert not np.array_equal(x[:5], y[:5])


def test_draws_within_spec_ranges(params: Params) -> None:
    s = params.scenario
    year = s.epoch.year
    assert year is not None
    start = datetime(year, 1, 1, tzinfo=UTC)
    end = datetime(year + 1, 1, 1, tzinfo=UTC)
    all_draws = [draw_seed(params, k) for k in range(300)]
    for d in all_draws:
        assert start <= d.epoch_utc < end
        assert d.epoch_utc.tzinfo is not None
        assert 48 * S_PER_HOUR <= d.run_length_s <= 72 * S_PER_HOUR
        assert 1e-13 <= d.rho0_kg_m3 <= 1e-12
        assert 60 <= d.H_km <= 70
    # log-uniform: log10(rho0) is roughly uniform on [-13, -12], so its mean is near -12.5
    assert np.mean([np.log10(d.rho0_kg_m3) for d in all_draws]) == pytest.approx(-12.5, abs=0.05)
    months = {d.epoch_utc.month for d in all_draws}
    assert len(months) == 12


def test_clock(params: Params) -> None:
    view = ParamView(params, "SAT-1")
    c = Clock.from_view(view, run_length_s=48 * S_PER_HOUR)
    assert c.step_s == 10
    assert c.tick_s == 30
    assert c.steps_per_tick == 3
    t = c.tick_times_s()
    assert t[0] == 0
    assert t[-1] == 48 * S_PER_HOUR
    assert c.n_ticks == len(t) == 5761
    np.testing.assert_allclose(np.diff(t), 30)
