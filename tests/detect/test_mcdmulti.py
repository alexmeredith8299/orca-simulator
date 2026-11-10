"""Multi-satellite MC dropout: slot order, held data with ages, and the two variants."""

from pathlib import Path

import numpy as np
import pytest
from detect_synth import ARM_S, FAULT_VEHICLE, make_batch

from expand3.detect.config import DetectConfig
from expand3.detect.harness import detector_names, run_detectors

pytest.importorskip("torch")

from expand3.detect.mcdmulti import MCDM, assemble, slot_order

PARAMS_DIR = Path(__file__).resolve().parents[2] / "params"
CLUSTERS = [["SAT-1", "SAT-2", "SAT-3"], ["SAT-4", "SAT-5", "SAT-6"]]
CFG = DetectConfig(h_floor=12.0, mcd_epochs=8, mcd_hidden=16, mcd_lags=8, mcd_samples=10,
                   mcd_window=6, mcd_threads=2, mcd_batch=128, mcd_device="cpu")
TRAIN = ["none"] * 6 + ["F1", "F7", "none", "none"]
TEST = ["F1", "F7", "none"]


def test_slot_order_is_self_then_cluster_mates_then_the_other_cluster() -> None:
    assert slot_order("SAT-2", CLUSTERS) == ["SAT-2", "SAT-1", "SAT-3", "SAT-4", "SAT-5",
                                             "SAT-6"]
    assert slot_order("SAT-5", CLUSTERS) == ["SAT-5", "SAT-4", "SAT-6", "SAT-1", "SAT-2",
                                             "SAT-3"]


def _feats(sats: list[str], n: int) -> dict[str, np.ndarray]:
    """One feature per satellite whose value is its number times 100 plus the tick."""
    return {s: (100.0 * int(s[-1]) + np.arange(n, dtype=float))[:, None] for s in sats}


def test_instantaneous_view_holds_every_satellites_current_features() -> None:
    sats = [s for c in CLUSTERS for s in c]
    n = 5
    order = slot_order("SAT-2", CLUSTERS)
    idx = {s: np.arange(n) for s in sats}
    age = {s: np.zeros(n) for s in sats}
    x = assemble(_feats(sats, n), idx, age, order, {s: 0.0 for s in sats}, cap_s=600.0)
    assert x.shape == (n, 6 * 3)  # per slot: one feature, the age and the bus flag
    assert x[:, 0].tolist() == [200.0 + k for k in range(n)]    # self
    assert x[:, 3].tolist() == [100.0 + k for k in range(n)]    # SAT-1
    assert x[:, 15].tolist() == [600.0 + k for k in range(n)]   # SAT-6
    assert np.all(x[:, 1::3] == 0.0)  # nothing is stale


def test_held_features_are_the_old_values_with_their_age_and_missing_ones_are_zero() -> None:
    sats = [s for c in CLUSTERS for s in c]
    n = 4
    order = slot_order("SAT-2", CLUSTERS)
    idx = {s: np.arange(n) for s in sats}
    age = {s: np.zeros(n) for s in sats}
    idx["SAT-1"] = np.array([-1, 0, 0, 0])   # first message arrives at tick 1, then none
    age["SAT-1"] = np.array([np.inf, 0.0, 30.0, 60.0])
    x = assemble(_feats(sats, n), idx, age, order, {s: 0.0 for s in sats}, cap_s=600.0)
    assert x[:, 3].tolist() == [0.0, 100.0, 100.0, 100.0]
    assert x[:, 4].tolist() == [1.0, 0.0, 30.0 / 600.0, 60.0 / 600.0]  # age / cap, capped at 1


def test_a_satellite_absent_from_the_data_is_a_stale_zero_slot() -> None:
    sats = ["SAT-1", "SAT-2", "SAT-3"]  # the synthetic batches carry one cluster only
    n = 3
    x = assemble(_feats(sats, n), {s: np.arange(n) for s in sats},
                 {s: np.zeros(n) for s in sats}, slot_order("SAT-1", CLUSTERS),
                 {s: 0.0 for s in sats}, cap_s=600.0)
    assert x.shape == (n, 18)
    assert np.all(x[:, 9::3] == 0.0)    # features of slots 3-5
    assert np.all(x[:, 10::3] == 1.0)   # age at the cap
    assert np.all(x[:, 11::3] == 0.0)   # bus flag
    assert np.all(np.isfinite(x))


def test_bus_flag_marks_bus_b() -> None:
    sats = [s for c in CLUSTERS for s in c]
    n = 2
    x = assemble(_feats(sats, n), {s: np.arange(n) for s in sats},
                 {s: np.zeros(n) for s in sats}, slot_order("SAT-5", CLUSTERS),
                 {s: float(s in ("SAT-5", "SAT-6")) for s in sats}, cap_s=600.0)
    assert x[0, 2::3].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def test_registry_names() -> None:
    assert set(MCDM) == {"mcdm_thermal", "mcdm_eps", "mcdm_thermal_comms", "mcdm_eps_comms"}


def test_all_leaves_out_the_multi_satellite_variants_but_names_pick_them() -> None:
    assert not set(detector_names("all")) & set(MCDM)
    assert detector_names("mcdm_eps,mcdm_eps_comms") == ["mcdm_eps", "mcdm_eps_comms"]


@pytest.mark.slow
@pytest.mark.parametrize("suffix", ["", "_comms"])
def test_f7_and_f1_scores_rise_after_arm(tmp_path: Path, suffix: str) -> None:
    seeds = make_batch(tmp_path, [*TRAIN, *TEST])
    tune, test = seeds[: len(TRAIN)], seeds[len(TRAIN):]
    keep: dict[str, object] = {}
    run_detectors(tmp_path, [f"mcdm_thermal{suffix}", f"mcdm_eps{suffix}"], tune, test, CFG,
                  PARAMS_DIR, keep=keep)
    by_seed = keep["traces"]
    for seed, kind, name in ((test[0], "F1", f"mcdm_eps{suffix}"),
                             (test[1], "F7", f"mcdm_thermal{suffix}")):
        traces = by_seed[seed][0][name]  # type: ignore[index]
        tr = next(t for t in traces if t.sat_id == FAULT_VEHICLE[kind])
        before, after = tr.score[tr.t_s < ARM_S], tr.score[tr.t_s >= ARM_S]
        assert after.max() > 3.0 * max(before.max(), 1.0), (kind, before.max(), after.max())
