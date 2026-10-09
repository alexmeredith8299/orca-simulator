"""M3: burn planning and the command / uplink / burn state machine."""

import pytest

from expand3.mission.burns import BurnController, BurnRecord, accel_magnitude_m_s2, plan_burns
from expand3.params import Params, ParamView
from expand3.timebase import draw_seed
from expand3.units import S_PER_HOUR

TICK, STEP = 30.0, 10.0


def test_accel_sized_for_dv(params: Params) -> None:
    a = accel_magnitude_m_s2(ParamView(params, "SAT-2"), 6 * S_PER_HOUR)
    assert a * 6 * S_PER_HOUR == pytest.approx(1.0)


def test_plan(params: Params) -> None:
    for seed in range(20):
        d = draw_seed(params, seed)
        plans = plan_burns(params, d, STEP)
        assert set(plans) == {"SAT-2", "SAT-3", "SAT-4"}
        assert 5 * S_PER_HOUR - STEP <= d.burn_length_s <= 7 * S_PER_HOUR
        for p in plans.values():
            assert 12 * S_PER_HOUR <= p.command_time_s <= 36 * S_PER_HOUR
            assert p.length_s % STEP == 0
            assert p.accel_m_s2 * p.length_s == pytest.approx(1.0)
    assert draw_seed(params, 3) == draw_seed(params, 3)
    assert draw_seed(params, 3).burn_command_s != draw_seed(params, 4).burn_command_s


def drive(ctrl: BurnController, visible_ticks: set[int], n_ticks: int) -> list[float]:
    active = []
    for n in range(n_ticks):
        ctrl.at_tick(n * TICK, n in visible_ticks)
        for j in range(3):
            t = n * TICK + j * STEP
            if ctrl.active(t):
                active.append(t)
                ctrl.flew(STEP)
    return active


def test_uplink_then_burn() -> None:
    rec = BurnRecord("SAT-2", command_time_s=100 * TICK, length_s=600.0, accel_m_s2=1e-3)
    ctrl = BurnController(rec, TICK, STEP)
    active = drive(ctrl, set(range(50, 60)) | set(range(110, 121)), 400)
    assert rec.uplink_pass == (110 * TICK, 120 * TICK)
    assert rec.burn_start_s == 121 * TICK  # the first tick after loss of signal
    assert active[0] == rec.burn_start_s
    assert len(active) * STEP == 600.0
    assert rec.flown_s == 600.0
    assert rec.burn_end_s == rec.burn_start_s + 600.0


def test_command_during_a_pass_uses_that_pass() -> None:
    rec = BurnRecord("SAT-3", command_time_s=105 * TICK, length_s=60.0, accel_m_s2=1e-3)
    drive(BurnController(rec, TICK, STEP), set(range(100, 111)), 200)
    assert rec.uplink_pass == (100 * TICK, 110 * TICK)
    assert rec.burn_start_s == 111 * TICK


def test_suspension_extends_the_burn() -> None:
    rec = BurnRecord("SAT-4", command_time_s=0.0, length_s=900.0, accel_m_s2=1e-3)
    window = (20 * TICK, 25 * TICK)  # inside the burn, which starts at tick 10
    ctrl = BurnController(rec, TICK, STEP, suspend=[window])
    active = drive(ctrl, set(range(5, 10)), 400)
    assert rec.burn_start_s == 10 * TICK
    assert not any(window[0] <= t < window[1] for t in active)
    assert len(active) * STEP == 900.0
    assert rec.suspended == [window]
    assert rec.burn_end_s == rec.burn_start_s + 900.0 + (window[1] - window[0])


def test_no_pass_no_burn() -> None:
    rec = BurnRecord("SAT-2", command_time_s=10 * TICK, length_s=60.0, accel_m_s2=1e-3)
    assert drive(BurnController(rec, TICK, STEP), set(range(0, 5)), 100) == []
    assert rec.burn_start_s is None
