"""A stored fault-free truth gives the same run as computing it again (sensor-only faults and
the no-fault run share it)."""

from pathlib import Path

import numpy as np
import pytest

from expand3.faults import FaultConfig
from expand3.mission.thread import compute_truth, reuses_baseline, run_mission
from expand3.params import load_params
from expand3.run import load_or_compute_truth
from expand3.timebase import draw_seed
from expand3.units import S_PER_HOUR

PARAMS_DIR = Path(__file__).resolve().parents[2] / "params"


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b, equal_nan=True))


def test_no_fault_run_from_truth_matches() -> None:
    params = load_params(PARAMS_DIR)
    dur = 2 * S_PER_HOUR
    truth = compute_truth(params, draw_seed(params, 1), dur, None, None)
    fresh = run_mission(params, 1, duration_s=dur, fault=FaultConfig(mode="none"))
    reused = run_mission(params, 1, duration_s=dur, fault=FaultConfig(mode="none"),
                         baseline=truth)
    assert _same(fresh.r_km, reused.r_km)
    assert _same(fresh.signals["SAT-2"].gnss_r.meas, reused.signals["SAT-2"].gnss_r.meas)
    assert _same(fresh.eps["SAT-3"].soc, reused.eps["SAT-3"].soc)


def test_which_faults_reuse_the_truth() -> None:
    params = load_params(PARAMS_DIR)
    ext = FaultConfig(mode="forced", fault_id="gnss_jump", fault_set="extended")
    assert reuses_baseline(params, 1, FaultConfig(mode="none"))
    assert reuses_baseline(params, 1, ext)
    assert not reuses_baseline(params, 1, FaultConfig(mode="forced", fault_id="F1"))
    assert not reuses_baseline(params, 1, FaultConfig(mode="forced", fault_id="F5"))


def test_truth_file_is_reused(tmp_path: Path) -> None:
    first = load_or_compute_truth(PARAMS_DIR, 1, 1.0, tmp_path)
    files = list(tmp_path.glob("truth_00001_*.pkl"))
    assert len(files) == 1
    second = load_or_compute_truth(PARAMS_DIR, 1, 1.0, tmp_path)
    assert _same(first.traj.r, second.traj.r)
    assert len(list(tmp_path.glob("truth_00001_*.pkl"))) == 1


@pytest.mark.slow
def test_sensor_fault_from_truth_matches() -> None:
    params = load_params(PARAMS_DIR)
    dur = 36 * S_PER_HOUR
    cfg = FaultConfig(mode="forced", fault_id="gnss_jump", fault_set="extended")
    truth = compute_truth(params, draw_seed(params, 3), dur, None, None)
    fresh = run_mission(params, 3, duration_s=dur, fault=cfg)
    reused = run_mission(params, 3, duration_s=dur, fault=cfg, baseline=truth)
    assert fresh.fault is not None
    assert reused.fault is not None
    assert _same(fresh.fault.s, reused.fault.s)
    for sat in fresh.sats:
        assert _same(fresh.signals[sat].gnss_r.meas, reused.signals[sat].gnss_r.meas)
        assert _same(fresh.signals[sat].imu.meas, reused.signals[sat].imu.meas)
