"""Detectors on synthetic batches with known faults, and the scoring of their alarms."""

from pathlib import Path

import numpy as np
import pytest
from detect_synth import ARM_S, FAULT_VEHICLE, TAKE_S, make_batch

from expand3.detect.base import Alarm
from expand3.detect.config import DetectConfig
from expand3.detect.detectors import wheel_fits
from expand3.detect.harness import detect_batch, read_alarms, run_detectors
from expand3.detect.score import score_seeds, write_score

PARAMS_DIR = Path(__file__).resolve().parents[2] / "params"
TRAIN = ["none", "none", "none", "none", "F2", "F8", "F9", "F3", "F4", "F10", "none"]
# Two upsets in one tick scores about 9.7, which the default floor of 8 lets through
CFG = DetectConfig(h_floor=12.0)
TEST = ["F2", "F3", "F4", "F8", "F9", "F10", "none"]


@pytest.fixture(scope="module")
def batch(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[int], list[int]]:
    out = tmp_path_factory.mktemp("synth")
    seeds = make_batch(out, [*TRAIN, *TEST])
    return out, seeds[: len(TRAIN)], seeds[len(TRAIN):]


@pytest.fixture(scope="module")
def detected(batch: tuple[Path, list[int], list[int]]) -> list[Alarm]:
    out, tune, test = batch
    alarms, _ = run_detectors(out, ["f2", "f8", "f9", "f10", "wheels"], tune, tune + test,
                              CFG, PARAMS_DIR)
    return alarms


def test_wheel_fit_recovers_k_and_c() -> None:
    rng = np.random.default_rng(1)
    n = 30
    u = rng.uniform(-0.1, 0.1, n)
    h = np.zeros(n)
    for i in range(1, n):
        h[i] = h[i - 1] - (0.5 * u[i] + 0.02 * h[i - 1])
    k, c = wheel_fits(np.arange(n, dtype=float), h, u, min_rows=4, min_conditioning=1e-9)
    assert np.isnan(k[:4]).all()
    assert k[-1] == pytest.approx(0.5)
    assert c[-1] == pytest.approx(0.02)


def test_wheel_fit_skips_collinear_rows() -> None:
    n = 10
    k, _ = wheel_fits(np.arange(n, dtype=float), np.zeros(n), np.zeros(n), 4, 1e-9)
    assert np.isnan(k).all()


def test_each_fault_is_detected_and_isolated(batch: tuple[Path, list[int], list[int]],
                                             detected: list[Alarm]) -> None:
    out, _, test = batch
    for seed, fault in zip(test, TEST, strict=True):
        mine = [a for a in detected if a.seed == seed and a.t_s >= ARM_S
                and a.sat_id == FAULT_VEHICLE.get(fault)]
        if fault == "none":
            continue
        assert mine, f"{fault} missed"
        assert mine[0].fault_id == fault
    res = score_seeds(out, detected, test)
    rows = {r["fault"]: r for r in res["rows"]}
    for fault in set(TEST) - {"none"}:
        assert rows[fault]["detected"] == 1
        assert rows[fault]["isolated_first"] == 1


def test_no_alarms_on_fault_free_data(batch: tuple[Path, list[int], list[int]],
                                      detected: list[Alarm]) -> None:
    _, _, test = batch
    quiet = {s for s, f in zip(test, TEST, strict=True) if f == "none"}
    assert not [a for a in detected if a.seed in quiet]
    other = [a for a in detected if a.sat_id != FAULT_VEHICLE.get(TEST[a.seed - len(TRAIN)])
             and a.seed >= len(TRAIN)]
    assert not other


def test_f2_names_the_string_and_f8_the_vehicle(batch: tuple[Path, list[int], list[int]],
                                                detected: list[Alarm]) -> None:
    _, _, test = batch
    f2 = next(a for a in detected if a.seed == test[0] and a.fault_id == "F2")
    assert (f2.sat_id, f2.detail) == ("SAT-2", "string_1")
    f8 = [a for a in detected if a.seed == test[3] and a.fault_id == "F8"]
    assert {a.sat_id for a in f8} == {"SAT-2"}
    assert f8[0].t_s == pytest.approx(ARM_S, abs=60)


def test_wheel_alarm_fires_inside_the_slew_before_the_take(
        batch: tuple[Path, list[int], list[int]], detected: list[Alarm]) -> None:
    out, _, test = batch
    f4 = next(a for a in detected if a.seed == test[2] and a.fault_id == "F4")
    assert f4.detail == "wheel_2"
    assert f4.t_s < TAKE_S + 60  # SAT-3's take
    res = score_seeds(out, detected, [test[2]])
    row = next(r for r in res["rows"] if r["fault"] == "F4")
    assert row["lead_median_s"] > 0


def test_f10_alarm_is_at_take_start(detected: list[Alarm], batch: tuple[Path, list[int], list[int]]
                                    ) -> None:
    _, _, test = batch
    a = next(a for a in detected if a.seed == test[5] and a.fault_id == "F10")
    assert a.t_s == pytest.approx(TAKE_S + 30.0)  # SAT-2 is the second take


def test_threshold_follows_the_false_alarm_target(batch: tuple[Path, list[int], list[int]]) -> None:
    out, tune, _ = batch
    _, loose = run_detectors(out, ["f2"], tune, tune, DetectConfig(target_far_per_vehicle_day=5.0,
                                                                  h_floor=0.0), PARAMS_DIR)
    _, tight = run_detectors(out, ["f2"], tune, tune,
                             DetectConfig(target_far_per_vehicle_day=0.01, h_floor=0.0),
                             PARAMS_DIR)
    assert loose["thresholds"]["f2"]["F2"] <= tight["thresholds"]["f2"]["F2"]


def test_unknown_seed_is_an_error(batch: tuple[Path, list[int], list[int]]) -> None:
    out, tune, _ = batch
    with pytest.raises(FileNotFoundError):
        run_detectors(out, ["f2"], tune, [999], DetectConfig(), PARAMS_DIR)


def test_workers_give_identical_alarms(batch: tuple[Path, list[int], list[int]],
                                       detected: list[Alarm]) -> None:
    out, tune, test = batch
    alarms, _ = run_detectors(out, ["f2", "f8", "f9", "f10", "wheels"], tune, tune + test,
                              CFG, PARAMS_DIR, workers=2)
    assert alarms == detected


def test_score_known_alarms(batch: tuple[Path, list[int], list[int]], tmp_path: Path) -> None:
    out, _, test = batch
    f2_seed, f8_seed, none_seed = test[0], test[3], test[6]
    alarms = [
        Alarm(f2_seed, "x", "SAT-2", ARM_S + 100, "F2", 50.0),       # detection, delay 100
        Alarm(f2_seed, "x", "SAT-1", 5000.0, "F2", 12.0),            # false alarm, other vehicle
        Alarm(f2_seed, "x", "SAT-2", 1000.0, "F9", 12.0),            # false alarm, pre-arm
        Alarm(f8_seed, "x", "SAT-2", ARM_S + 60, "F2", 9.0),         # wrong id first
        Alarm(f8_seed, "x", "SAT-2", ARM_S + 90, "F8", 30.0),
        Alarm(none_seed, "x", "SAT-3", 100.0, "F8", 9.0),            # false alarm in a clean run
    ]
    res = score_seeds(out, alarms, [f2_seed, f8_seed, none_seed])
    rows = {r["fault"]: r for r in res["rows"]}
    assert rows["F2"]["detected"] == 1
    assert rows["F2"]["delay_median_s"] == pytest.approx(100.0)
    assert rows["F2"]["lead_median_s"] == pytest.approx(TAKE_S + 30.0 - (ARM_S + 100.0))
    assert rows["F8"]["delay_median_s"] == pytest.approx(90.0)
    assert rows["F8"]["isolated_first"] == 0  # the first alarm named F2
    assert res["confusion"]["counts"]["F8"] == {"F2": 1}
    assert res["confusion"]["counts"]["none"] == {"F2": 1, "F9": 1, "F8": 1}
    assert res["false_alarms"] == 3
    # exposure: 3 vehicles x 1 day per seed, less the faulted vehicle's post-arm time
    days = (3 * 3 * 86400.0 - 2 * (86400.0 - ARM_S)) / 86400.0
    assert res["vehicle_days"] == pytest.approx(days, rel=1e-3)
    assert res["false_alarms_per_vehicle_day"] == pytest.approx(3 / days, rel=1e-3)
    write_score(tmp_path, {"test": res})
    assert "F2" in (tmp_path / "score.md").read_text()


def test_alarm_csv_round_trip(batch: tuple[Path, list[int], list[int]], tmp_path: Path) -> None:
    out, tune, test = batch
    alarms = detect_batch(out, ["f9"], tune, test, tmp_path, DetectConfig(), PARAMS_DIR)
    assert read_alarms(tmp_path / "alarms.csv") == alarms
    assert (tmp_path / "calibration.json").is_file()
