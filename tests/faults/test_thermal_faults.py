"""Extended faults of Step 2, thermal group (docs/extended-faults-plan.md), in paired mission runs:
forced fault vs no fault, same seed."""

import numpy as np
import pytest

from expand3.faults import FaultConfig
from expand3.mission.thread import MissionRun, run_mission
from expand3.params import Params
from expand3.units import S_PER_HOUR

pytestmark = pytest.mark.slow
SEED = 20270
LEAD_S = 10_000.0
DURATION_S = 36 * S_PER_HOUR
FAULTS = ["temp_sensor_bias", "temp_sensor_freeze", "heater_stuck_on", "heater_stuck_off",
          "radiator_degradation"]
BATTERY, RADIATOR = "battery", "bus_nz"


@pytest.fixture(scope="module")
def nominal(params: Params) -> MissionRun:
    return run_mission(params, SEED, duration_s=DURATION_S, fault=FaultConfig(mode="none"))


@pytest.fixture(scope="module")
def faulted(params: Params) -> dict[str, MissionRun]:
    return {fid: run_mission(params, SEED, duration_s=DURATION_S,
                             fault=FaultConfig(mode="forced", fault_id=fid, lead_s=LEAD_S))
            for fid in FAULTS}


def _node(run: MissionRun, sat: str, name: str, truth: bool) -> np.ndarray:
    s = run.signals[sat].node_temperature
    j = run.signals[sat].node_names.index(name)
    out: np.ndarray = (s.truth if truth else s.meas)[:, j]
    return out


def _after(run: MissionRun) -> np.ndarray:
    assert run.fault is not None
    return np.asarray(run.signals[run.fault.vehicle].node_temperature.t_s >= run.fault.t_arm_s)


def test_a_thermistor_bias_moves_the_reading_and_not_the_battery(
        params: Params, nominal: MissionRun, faulted: dict[str, MissionRun]) -> None:
    run = faulted["temp_sensor_bias"]
    sat = params.faults.extended["temp_sensor_bias"].vehicle
    after = _after(run)
    d = _node(run, sat, BATTERY, False) - _node(nominal, sat, BATTERY, False)
    np.testing.assert_allclose(d[after], 5.0, atol=1e-9)
    np.testing.assert_array_equal(d[~after], 0.0)
    np.testing.assert_array_equal(_node(run, sat, BATTERY, True),
                                  _node(nominal, sat, BATTERY, True))  # nominal never inhibits


def test_a_frozen_thermistor_holds_its_last_reading(
        params: Params, faulted: dict[str, MissionRun]) -> None:
    run = faulted["temp_sensor_freeze"]
    sat = params.faults.extended["temp_sensor_freeze"].vehicle
    after = _after(run)
    held = _node(run, sat, BATTERY, False)[after]
    assert np.ptp(held) == 0.0
    assert np.ptp(_node(run, sat, BATTERY, True)[after]) > 0.5  # the battery itself moves


def test_a_heater_stuck_on_warms_the_battery_up_to_the_cutoff(
        params: Params, nominal: MissionRun, faulted: dict[str, MissionRun]) -> None:
    run = faulted["heater_stuck_on"]
    sat = params.faults.extended["heater_stuck_on"].vehicle
    after = _after(run)
    warm = _node(run, sat, BATTERY, True)[after] - _node(nominal, sat, BATTERY, True)[after]
    assert warm.max() > 5.0
    cutoff = params.bus("SAT-3").eps.heater_cutoff_true_K
    assert _node(run, sat, BATTERY, True).max() <= cutoff + 1.0


def test_a_heater_stuck_off_changes_nothing_in_a_nominal_run(
        params: Params, nominal: MissionRun, faulted: dict[str, MissionRun]) -> None:
    run = faulted["heater_stuck_off"]
    sat = params.faults.extended["heater_stuck_off"].vehicle
    assert not run.signals[sat].heater_delivered["battery"].any()
    np.testing.assert_array_equal(_node(run, sat, BATTERY, True),
                                  _node(nominal, sat, BATTERY, True))


def test_a_degraded_radiator_runs_warmer(
        params: Params, nominal: MissionRun, faulted: dict[str, MissionRun]) -> None:
    run = faulted["radiator_degradation"]
    sat = params.faults.extended["radiator_degradation"].vehicle
    after = _after(run)
    d = _node(run, sat, RADIATOR, True) - _node(nominal, sat, RADIATOR, True)
    assert d[after][-20:].mean() > 0.5
    np.testing.assert_array_equal(d[~after], 0.0)
