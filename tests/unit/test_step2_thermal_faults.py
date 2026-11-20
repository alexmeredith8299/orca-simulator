"""Extended faults of Step 2, thermal group (docs/extended-faults-plan.md): the fault paths exist,
are off, and the thermistor and heater faults count as truth faults, not sensor-only ones."""

from expand3.mission.thread import _sensor_only
from expand3.params import Params, ParamView


def test_the_thermal_fault_paths_exist_and_are_off(params: Params) -> None:
    view = ParamView(params, "SAT-2")
    for path in ("sensor_faults.thermistor.battery_bias_K",
                 "sensor_faults.thermistor.battery_freeze",
                 "sensor_faults.heater.battery_stuck_on",
                 "sensor_faults.heater.battery_stuck_off"):
        assert view.get(path) == 0.0
    for fid, sat, shapes in [("temp_sensor_bias", "SAT-2", ["step", "ramp"]),
                             ("temp_sensor_freeze", "SAT-3", ["step"]),
                             ("heater_stuck_on", "SAT-3", ["step"]),
                             ("heater_stuck_off", "SAT-2", ["step"]),
                             ("radiator_degradation", "SAT-2", ["ramp"])]:
        f = params.faults.extended[fid]
        assert f.vehicle == sat
        assert (f.shapes or [f.profile]) == shapes


def test_the_thermal_faults_reuse_no_stored_truth(params: Params) -> None:
    for fid in ("temp_sensor_bias", "temp_sensor_freeze", "heater_stuck_on", "heater_stuck_off",
                "radiator_degradation"):
        assert not _sensor_only(params, fid)
    assert _sensor_only(params, "st_dropout")
