"""Detectors and scoring on a small batch from the simulator itself (two forced faults)."""

from pathlib import Path

import pytest

from expand3.cli import main
from expand3.detect.base import Alarm
from expand3.detect.config import DetectConfig
from expand3.detect.harness import read_alarms, run_detectors
from expand3.faults import FaultConfig
from expand3.run import run_batch

PARAMS_DIR = Path(__file__).resolve().parents[2] / "params"
DURATION_H = 34.0  # past the take, which is at 30 h or just after
CFG = DetectConfig(h_floor=12.0)


@pytest.fixture(scope="module")
def batch(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("real")
    run_batch(PARAMS_DIR, [0], out, FaultConfig(mode="forced", fault_id="F2"), DURATION_H, 1)
    run_batch(PARAMS_DIR, [1], out, FaultConfig(mode="forced", fault_id="F8"), DURATION_H, 1)
    return out


@pytest.mark.slow
def test_forced_faults_are_found_and_isolated(batch: Path) -> None:
    alarms, _ = run_detectors(batch, ["f2", "f8", "f9", "f10", "wheels"], [0, 1], [0, 1], CFG,
                              PARAMS_DIR)
    first: dict[int, Alarm] = {}
    for a in alarms:
        first.setdefault(a.seed, a)
    assert (first[0].sat_id, first[0].fault_id) == ("SAT-2", "F2")
    assert (first[1].sat_id, first[1].fault_id) == ("SAT-2", "F8")


@pytest.mark.slow
def test_cli_detect_then_score(batch: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["detect", str(batch), "--detector", "f2,f8", "--tune", "0:2",
                 "--params", str(PARAMS_DIR)]) == 0
    alarms = batch / "detect" / "f2+f8" / "alarms.csv"
    assert {a.fault_id for a in read_alarms(alarms)} >= {"F2", "F8"}
    assert main(["score", str(batch), "--tune", "0:2", "--test", "2:3"]) == 0
    out = capsys.readouterr().out
    assert "F2 1/1" in out
    assert "F8 1/1" in out
    assert (batch / "detect" / "f2+f8" / "score.md").is_file()
