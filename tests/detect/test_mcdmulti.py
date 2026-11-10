"""Multi-satellite MC dropout: slot order, held data with ages, and the two variants."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from detect_synth import ARM_S, FAULT_VEHICLE, make_batch

from expand3.detect.config import DetectConfig
from expand3.detect.data import SeedData
from expand3.detect.harness import detector_names, run_detectors

pytest.importorskip("torch")

from expand3.detect.mcdmulti import MCDM, assemble, slot_order
from expand3.detect.mcdrop import McdModel

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


def test_a_vehicle_missing_a_thermistor_pads_to_the_shared_target_columns(tmp_path: Path) -> None:
    seed_dir = tmp_path / f"seed_{make_batch(tmp_path, ['none'])[0]:05d}"
    table = pq.read_table(seed_dir / "measurements.parquet").to_pandas()
    gone = "T_wing_ny_K_meas"  # Bus B has no such thermistor
    table.loc[table.sat_id == "SAT-3", gone] = np.nan
    pq.write_table(pa.Table.from_pandas(table, preserve_index=False),
                   seed_dir / "measurements.parquet")
    from expand3.detect.mcdmulti import McdMultiThermal
    out = McdMultiThermal(CFG, str(PARAMS_DIR)).summarize(SeedData(seed_dir))["all"]
    assert all(y.shape[1] == len(out["cols"]) for y in out["y"])
    k = out["cols"].index(gone)
    sat3 = out["y"][SeedData(seed_dir).sat_ids.index("SAT-3")]
    assert np.all(np.isnan(sat3[:, k]))
    assert np.any(np.isfinite(out["y"][0][:, k]))


def _signals(sats: list[str], n: int) -> dict[str, np.ndarray]:
    """Two measured signals per satellite: its number times 1000 plus the tick, and its negative."""
    return {s: np.stack([1000.0 * int(s[-1]) + np.arange(n, dtype=float),
                         -1000.0 * int(s[-1]) - np.arange(n, dtype=float)], axis=1)
            for s in sats}


def test_measured_signals_of_the_peers_are_inputs_and_the_vehicles_own_are_masked() -> None:
    sats = [s for c in CLUSTERS for s in c]
    n = 4
    ident = {s: np.arange(n) for s in sats}
    zero = {s: np.zeros(n) for s in sats}
    x = assemble(_feats(sats, n), ident, zero, slot_order("SAT-2", CLUSTERS),
                 {s: 0.0 for s in sats}, 600.0, meas=_signals(sats, n), self_sat="SAT-2")
    # self slot: 1 feature + age + bus; five peer slots: 1 feature + 2 signals + age + bus
    assert x.shape == (n, 3 + 5 * 5)
    assert x[:, 0].tolist() == [200.0 + k for k in range(n)]
    assert x[:, 3].tolist() == [100.0 + k for k in range(n)]          # SAT-1 feature
    assert x[:, 4].tolist() == [1000.0 + k for k in range(n)]         # SAT-1 signals
    assert x[:, 5].tolist() == [-1000.0 - k for k in range(n)]
    assert not np.any(x == 2000.0)                                     # SAT-2's own: absent


def test_the_literal_all_variant_keeps_the_vehicles_own_signals() -> None:
    sats = [s for c in CLUSTERS for s in c]
    n = 3
    x = assemble(_feats(sats, n), {s: np.arange(n) for s in sats},
                 {s: np.zeros(n) for s in sats}, slot_order("SAT-2", CLUSTERS),
                 {s: 0.0 for s in sats}, 600.0, meas=_signals(sats, n), self_sat="SAT-2",
                 own_meas=True)
    assert x.shape == (n, 6 * 5)
    assert x[:, 1].tolist() == [2000.0 + k for k in range(n)]


def test_signals_not_yet_received_are_missing_not_zero() -> None:
    sats = [s for c in CLUSTERS for s in c]
    n = 4
    idx = {s: np.arange(n) for s in sats}
    age = {s: np.zeros(n) for s in sats}
    idx["SAT-1"] = np.array([-1, -1, 1, 1])
    age["SAT-1"] = np.array([np.inf, np.inf, 0.0, 30.0])
    x = assemble(_feats(sats, n), idx, age, slot_order("SAT-2", CLUSTERS),
                 {s: 0.0 for s in sats}, 600.0, meas=_signals(sats, n), self_sat="SAT-2")
    assert np.all(np.isnan(x[:2, 4:6]))
    assert x[2:, 4].tolist() == [1001.0, 1001.0]
    assert x[:2, 3].tolist() == [0.0, 0.0]  # context features keep the zero convention
    # a vehicle absent from the data is all missing
    sub = ["SAT-1", "SAT-2", "SAT-3"]
    y = assemble(_feats(sub, 2), {s: np.arange(2) for s in sub}, {s: np.zeros(2) for s in sub},
                 slot_order("SAT-1", CLUSTERS), {s: 0.0 for s in sub}, 600.0,
                 meas=_signals(sub, 2), self_sat="SAT-1")
    assert np.all(np.isnan(y[:, 4 + 2 * 5: 6 + 2 * 5]))


def test_model_fills_missing_inputs_with_the_training_mean() -> None:
    rng = np.random.default_rng(1)
    series, targets = [], []
    for _ in range(6):
        t = np.arange(200.0)
        x = np.stack([np.sin(t / 15.0), np.cos(t / 25.0), 5.0 + rng.normal(size=200)], axis=1)
        x = x.astype(np.float32)
        y = (1.5 * x[:, :1] + 0.1 * rng.normal(size=(200, 1)))
        x[:30, 2] = np.nan  # a signal not yet received
        series.append(x)
        targets.append(y)
    model = McdModel(CFG, ["y"]).fit(series, targets, list(range(6)))
    assert np.all(np.isfinite(model.x_mu))
    assert np.all(model.x_sd > 0)
    mean, std = model.predict(series[0], seed=0)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(std))


def test_registry_names() -> None:
    base = {"mcdm_thermal", "mcdm_eps", "mcdm_thermal_comms", "mcdm_eps_comms"}
    sig = {f"{n}_sig" for n in ("mcdm_thermal", "mcdm_eps")}
    sig |= {f"{n}_sig_comms" for n in ("mcdm_thermal", "mcdm_eps")}
    own = {f"{n}_all" for n in ("mcdm_thermal", "mcdm_eps")}
    assert set(MCDM) == base | sig | own


def test_all_leaves_out_the_multi_satellite_variants_but_names_pick_them() -> None:
    assert not set(detector_names("all")) & set(MCDM)
    assert detector_names("mcdm_eps,mcdm_eps_comms") == ["mcdm_eps", "mcdm_eps_comms"]


@pytest.mark.slow
@pytest.mark.parametrize("suffix", ["", "_comms", "_sig", "_sig_comms", "_all"])
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
