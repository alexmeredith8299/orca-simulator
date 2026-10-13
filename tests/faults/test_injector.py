"""M7: onset profiles, perturbation forms, ParamView schedules and the per-seed fault draw.
Spec: Faults."""

import numpy as np
import pytest
from scipy.stats import chisquare

from expand3.faults import Injector, arm_time, draw_fault, onset, perturbed
from expand3.params import Params, ParamView
from expand3.timebase import rng

TICK = 30.0


def test_arm_on_first_tick_at_or_after_t0() -> None:
    assert arm_time(3000.0, TICK) == 3000.0
    assert arm_time(3000.1, TICK) == 3030.0
    assert arm_time(3029.9, TICK) == 3030.0


def test_step_and_ramp_profiles(params: Params) -> None:
    f = params.faults.faults
    t_arm = 90_000.0
    ticks = np.arange(0, 140_000, TICK)
    step = np.array([onset(f["F2"], t_arm, t) for t in ticks])
    np.testing.assert_array_equal(step, (ticks >= t_arm).astype(float))
    ramp = np.array([onset(f["F1"], t_arm, t) for t in ticks])
    expected = np.clip((ticks - t_arm) / 21600.0, 0.0, 1.0)
    np.testing.assert_array_equal(ramp, expected)
    assert onset(f["F1"], t_arm, t_arm + 21600.0) == 1.0
    assert onset(f["F1"], t_arm, t_arm + 10800.0) == 0.5


def test_perturbation_forms(params: Params) -> None:
    f = params.faults.faults
    assert perturbed(f["F1"], 0.08, 1.0) == pytest.approx(0.24)  # x3 at full ramp
    assert perturbed(f["F1"], 0.08, 0.5) == pytest.approx(0.16)
    assert perturbed(f["F2"], 1.0, 1.0) == 0.0  # o_1 = 1 - s
    assert perturbed(f["F2"], 1.0, 0.0) == 1.0
    assert perturbed(f["F3"], 0.0, 1.0) == 0.02  # set directly
    assert perturbed(f["F8"], 0.0, 1.0) == 0.9
    assert perturbed(f["F7"], 0.08, 1.0) == pytest.approx(0.24)
    assert perturbed(f["F7"], 0.5, 1.0) == 1.0  # capped at 1
    assert perturbed(f["F5"], 220.0, 1.0) == pytest.approx(154.0)
    assert perturbed(f["F9"], 1 / 1800, 1.0) == pytest.approx(50 / 1800)


def test_schedule_through_paramview(params: Params) -> None:
    inj = Injector(params, "F7", t0_s=50_000.0, lead_s=8000.0, tick_s=TICK)
    view = ParamView(params, "SAT-2")
    inj.install(view)
    with pytest.raises(RuntimeError):
        view.get("thermal.nodes.bus_px.alpha")  # a scheduled path needs the time set
    assert view.get("eps.R_pack_ohm") == 0.08  # unscheduled paths need no time
    assert view.at(49_990.0).get("thermal.nodes.bus_px.alpha") == 0.08
    assert view.at(50_010.0).get("thermal.nodes.bus_px.alpha") == 0.08  # arms at 50 010 -> tick
    assert view.at(50_010.0).get("thermal.nodes.bus_px.epsilon") == pytest.approx(0.03)
    assert inj.t_arm_s == 50_010.0
    assert view.at(50_040.0).get("thermal.nodes.bus_px.alpha") == pytest.approx(0.24)
    assert view.at(50_040.0).get("thermal.nodes.bus_px.epsilon") == pytest.approx(0.09)
    f2 = Injector(params, "F2", t0_s=1000.0, lead_s=6000.0, tick_s=TICK)
    v2 = ParamView(params, "SAT-2")
    f2.install(v2)
    np.testing.assert_array_equal(v2.at(0.0).array("eps.string_health"), [1, 1, 1, 1])
    np.testing.assert_array_equal(v2.at(1020.0).array("eps.string_health"), [0, 1, 1, 1])
    with pytest.raises(ValueError, match="SAT-2"):
        f2.install(ParamView(params, "SAT-1"))  # the table's vehicle only


def test_label(params: Params) -> None:
    inj = Injector(params, "F1", t0_s=100_000.0, lead_s=9000.0, tick_s=TICK)
    t = np.arange(0, 130_000, TICK)
    lab = inj.label(t)
    assert (lab.fault_id, lab.vehicle, lab.profile) == ("F1", "SAT-2", "ramp")
    assert lab.paths == ["eps.R_pack_ohm"]
    assert lab.magnitude == 3
    assert lab.kind == "multiply"
    assert lab.t_arm_s == 100_020.0
    assert lab.lead_s == 9000.0
    np.testing.assert_allclose(lab.theta["eps.R_pack_ohm"], 0.08 * (1 + 2 * lab.s))
    assert lab.s[-1] == pytest.approx((t[-1] - 100_020.0) / 21600)


def test_draw_uniform_and_lead(params: Params) -> None:
    draws = [draw_fault(params, seed) for seed in range(3000)]
    ids = [d[0] for d in draws]
    counts = [ids.count(f"F{i}") for i in range(1, 11)]
    assert chisquare(counts).pvalue > 0.001
    leads = np.array([d[1] for d in draws])
    assert leads.min() >= 5700
    assert leads.max() <= 11400
    assert draw_fault(params, 42) == draw_fault(params, 42)
    rng(42, "noise.gnss.SAT-2").normal(size=1000)  # sensor streams do not move the draw
    assert draw_fault(params, 42) == draws[42]
