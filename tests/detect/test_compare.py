"""Observability ceiling, isolation before the take, detection-only scoring and the tables."""

from pathlib import Path

import pytest
from detect_synth import ARM_S, TAKE_S, make_batch

from expand3.detect.base import Alarm
from expand3.detect.data import SeedData
from expand3.detect.registry import describe
from expand3.detect.score import compare, evidence_time, score_seeds, write_comparison

KINDS = ["F10", "F3", "F9", "F2", "F7"]


@pytest.fixture(scope="module")
def batch(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("cmp")
    make_batch(out, KINDS)
    return out


def test_evidence_times(batch: Path) -> None:
    ev = {k: evidence_time(SeedData(batch / f"seed_{i:05d}"), v, k, ARM_S)
          for i, (k, v) in enumerate(zip(KINDS, ["SAT-2", "SAT-3", "SAT-2", "SAT-2", "SAT-2"],
                                         strict=True))}
    assert ev["F10"] == TAKE_S + 30.0            # the take itself
    assert ev["F3"] == TAKE_S + 60.0 - 40.0      # slew in of SAT-3, second take
    assert ev["F2"] == ARM_S
    assert ev["F7"] == ARM_S
    assert ev["F9"] is not None
    assert ev["F9"] >= ARM_S  # first SAA tick after arm


def test_ceiling_and_isolation_before_take(batch: Path) -> None:
    alarms = [
        Alarm(0, "d", "SAT-2", TAKE_S + 30.0, "F10", 9.0),   # at the take: not before it
        Alarm(1, "d", "SAT-3", TAKE_S + 40.0, "F3", 9.0),    # inside the slew in: before the take
        Alarm(3, "d", "SAT-2", ARM_S + 10, "F7", 9.0),       # wrong id first
        Alarm(3, "d", "SAT-2", ARM_S + 20, "F2", 9.0),
    ]
    res = score_seeds(batch, alarms, list(range(len(KINDS))))
    rows = {r["fault"]: r for r in res["rows"]}
    assert rows["F10"]["detected"] == 1
    assert rows["F10"]["isolated_before_take"] == 0
    assert rows["F10"]["ceiling"] == 0.0           # the window never opens before the take
    assert rows["F3"]["isolated_before_take"] == 1
    assert rows["F3"]["ceiling"] == 1.0
    assert rows["F2"]["isolated_before_take_rate"] == 1.0
    assert rows["F2"]["isolated_first"] == 0       # F7 came first
    assert rows["F7"]["detected"] == 0
    assert rows["F2"]["ceiling_lead_median_s"] == pytest.approx(TAKE_S + 30.0 - ARM_S)


def test_detection_only_alarms_count_for_any_fault(batch: Path) -> None:
    alarms = [Alarm(3, "u", "SAT-2", ARM_S + 100, "T2:eps", 50.0),
              Alarm(4, "u", "SAT-1", 500.0, "T2:eps", 50.0)]
    strict = score_seeds(batch, alarms, [3, 4])
    lenient = score_seeds(batch, alarms, [3, 4], detect_only=True)
    assert {r["fault"]: r["detected"] for r in strict["rows"]}["F2"] == 0
    rows = {r["fault"]: r for r in lenient["rows"]}
    assert rows["F2"]["detected"] == 1
    assert rows["F2"]["delay_median_s"] == 100
    assert lenient["false_alarms"] == 1


def test_comparison_tables(batch: Path, tmp_path: Path) -> None:
    alarms = [Alarm(3, "f2", "SAT-2", ARM_S + 5, "F2", 99.0),
              Alarm(3, "unsup_pca", "SAT-2", ARM_S + 60, "T2:eps", 9.0),
              Alarm(2, "f2", "SAT-1", 100.0, "F2", 20.0)]
    results = compare(batch, alarms, list(range(len(KINDS))), {"unsup_pca"})
    names = list(results)
    info = {n: describe(n) for n in names}
    write_comparison(tmp_path, results, {n: info[n][0] for n in names},
                     {n: info[n][2] for n in names}, "test")
    text = (tmp_path / "compare_test.md").read_text()
    assert "## F2" in text
    assert "f2" in text
    assert "unsup_pca" in text
    assert "Observability ceiling" in text
    csv = (tmp_path / "compare_test_F2.csv").read_text().splitlines()
    assert csv[0].startswith("detector,family,detection_rate")
    assert len(csv) == 3  # f2 and unsup_pca; no other detector covers F2 here
    assert describe("clf")[0] == "classifier"
    assert describe("wheels")[2] == {"F3", "F4"}
