"""MC dropout detectors for F2, F3, F4, F8, F9 and F10, alone and with the other vehicles."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from detect_synth import (
    A_CMD,
    A_KM,
    BURNS,
    FAULT_VEHICLE,
    PERIOD_S,
    make_batch,
)

from expand3.detect.config import DetectConfig
from expand3.detect.data import SeedData
from expand3.detect.harness import detector_names, run_detectors

pytest.importorskip("torch")

from expand3.detect.mcdnn import MCDN, blocks, gnss_elements, trailing_sum

PARAMS_DIR = Path(__file__).resolve().parents[2] / "params"
CFG = DetectConfig(h_floor=12.0, mcd_epochs=10, mcd_hidden=16, mcd_lags=8, mcd_samples=10,
                   mcd_window=6, mcd_threads=2, mcd_batch=128, mcd_device="cpu",
                   mcd_stride=5)


def test_blocks_sum_every_stride_and_keep_a_partial_last_block() -> None:
    x = np.arange(11, dtype=float)
    start, end = blocks(11, 5)
    assert start.tolist() == [0, 5, 10]
    assert end.tolist() == [4, 9, 10]  # last tick of each block
    assert np.add.reduceat(x, start).tolist() == [10.0, 35.0, 10.0]


def test_trailing_sum_is_causal() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert trailing_sum(x, 3).tolist() == [1.0, 3.0, 6.0, 9.0, 12.0]


def test_registry_names() -> None:
    faults = ("mcd_f2", "mcdn_f8", "mcdn_f9", "mcdn_f10", "mcdn_wheels", "mcdn_burn", "mcdn_gnss")
    assert set(MCDN) == {f"{f}{s}" for f in faults for s in ("", "_sig", "_sig_comms")}
    assert not set(detector_names("all")) & set(MCDN)


def _samples(tmp_path: Path, kind: str, name: str) -> tuple[SeedData, dict[str, object]]:
    seed = make_batch(tmp_path, [kind])[0]
    d = SeedData(tmp_path / f"seed_{seed:05d}")
    det = MCDN[name](CFG, str(PARAMS_DIR))
    return d, {sat: det.samples(d, sat) for sat in d.sat_ids}  # type: ignore[attr-defined]


def test_upset_blocks_hold_the_counts_and_only_saa_blocks_are_targets(tmp_path: Path) -> None:
    d, sam = _samples(tmp_path, "none", "mcdn_f9")
    m = d.measurements("SAT-1")
    s = sam["SAT-1"]
    assert s.y[:, 0][np.isfinite(s.y[:, 0])].sum() <= m["upsets"].sum()  # type: ignore[attr-defined]
    inside = s.aux["inside"]  # type: ignore[attr-defined]
    assert np.all(np.isnan(s.y[inside == 0, 0]))  # type: ignore[attr-defined]
    assert np.all(np.isfinite(s.y[inside > 0, 0]))  # type: ignore[attr-defined]


def test_dark_current_has_one_sample_per_take_at_its_log2_value(tmp_path: Path) -> None:
    d, sam = _samples(tmp_path, "none", "mcdn_f10")
    takes = d.sat_columns("takes", "SAT-2", ["t_take_s", "dark_current_e_s_px"])
    s = sam["SAT-2"]
    assert s.t.tolist() == [float(v) for v in takes["t_take_s"]]  # type: ignore[attr-defined]
    assert np.allclose(s.y[:, 0], np.log2(takes["dark_current_e_s_px"]))  # type: ignore[attr-defined]


def test_wheel_steps_target_the_momentum_change_which_is_the_command_when_healthy(
        tmp_path: Path) -> None:
    _, sam = _samples(tmp_path, "none", "mcdn_wheels")
    s = sam["SAT-1"]
    u = s.local[:, :4]  # type: ignore[attr-defined]
    assert s.y.shape == (len(s.t), 4)  # type: ignore[attr-defined]
    assert np.allclose(s.y, u, atol=1e-6)  # type: ignore[attr-defined]


def test_orbit_elements_come_back_from_the_fixes(tmp_path: Path) -> None:
    seed = make_batch(tmp_path, ["none"])[0]
    m = SeedData(tmp_path / f"seed_{seed:05d}").measurements("SAT-1")
    a, u = gnss_elements(m, 398600.4418)
    ok = np.isfinite(a)
    assert 0.4 < ok.mean() < 0.6  # fixes on every other tick
    expected_u = 2 * np.pi * m["t_s"] / PERIOD_S
    assert np.abs(np.angle(np.exp(1j * (u[ok] - expected_u[ok])))).max() < 0.01
    assert np.abs(a[ok] - (A_KM + 5.0 * np.sin(2 * expected_u[ok]))).max() < 0.5


def test_burn_input_is_the_delta_v_ordered_so_far(tmp_path: Path) -> None:
    _, sam = _samples(tmp_path, "none", "mcdn_burn")
    s = sam["SAT-3"]
    b0, b1 = BURNS["SAT-3"]
    expected = A_CMD * (np.clip(s.t, b0, b1) - b0)  # type: ignore[attr-defined]
    assert np.abs(s.local[:, 4] - expected).max() < 3 * A_CMD * 30.0  # type: ignore[attr-defined]
    assert np.all(sam["SAT-1"].local[:, 4] == 0.0)  # type: ignore[attr-defined]
    assert s.y.shape == (len(s.t), 1)  # type: ignore[attr-defined]


@pytest.mark.slow
@pytest.mark.parametrize("suffix", ["", "_sig_comms"])
@pytest.mark.parametrize(("name", "kind"), [("mcd_f2", "F2"), ("mcdn_f8", "F8"), ("mcdn_f9", "F9"),
                                       ("mcdn_f10", "F10"), ("mcdn_wheels", "F3"),
                                       ("mcdn_wheels", "F4"), ("mcdn_burn", "F5"),
                                       ("mcdn_burn", "F6")])
def test_the_faulted_vehicles_score_stands_out(tmp_path: Path, name: str, kind: str,
                                               suffix: str) -> None:
    seeds = make_batch(tmp_path, [*(["none"] * 8), kind])
    tune, test = seeds[:8], seeds[8:]
    cfg = replace(CFG, mcd_epochs=40, mcd_batch=64) if name != "mcd_f2" else CFG
    keep: dict[str, object] = {}
    run_detectors(tmp_path, [name + suffix], tune, test, cfg, PARAMS_DIR, keep=keep)
    traces = keep["traces"][test[0]][0][name + suffix]  # type: ignore[index]
    mine = [t for t in traces if t.fault_id == kind]
    top: dict[str, float] = {}
    for t in mine:  # a vehicle can have several traces (strings): its score is the largest
        top[t.sat_id] = max(top.get(t.sat_id, 0.0), float(np.max(t.score)))
    others = max((v for s, v in top.items() if s != FAULT_VEHICLE[kind]), default=0.0)
    assert top[FAULT_VEHICLE[kind]] > 3.0 * max(others, 1.0), (kind, top)


def test_gnss_miss_is_the_fix_minus_the_vehicles_own_solution_and_peers_add_theirs(
        tmp_path: Path) -> None:
    seed = make_batch(tmp_path, ["none"], kepler=True)[0]
    d = SeedData(tmp_path / f"seed_{seed:05d}")
    det = MCDN["mcdn_gnss"](CFG, str(PARAMS_DIR))
    own = det.samples(d, "SAT-1")  # type: ignore[attr-defined]
    assert own.y.shape[1] == 6
    m = d.measurements("SAT-1")
    k = own.tick[0]
    want = (m["gnss_r_x_km_meas"][k] - m["nav_r_x_km_meas"][k]) * 1000.0
    assert own.y[0, 0] == pytest.approx(want)
    assert own.y[:, :3].std() < 1.5  # the solution has averaged part of the fix noise away
    block = det.signal_block(d, "SAT-1")  # type: ignore[attr-defined]
    assert block.shape[1] == 18  # fix (6), solution (6), fix minus solution (6)
    assert not np.isnan(block[k]).any()
    sig = MCDN["mcdn_gnss_sig"](CFG, str(PARAMS_DIR))
    assert sig.x(d, "SAT-1", own).shape[1] > det.x(d, "SAT-1", own).shape[1]  # type: ignore[attr-defined]


@pytest.mark.slow
@pytest.mark.parametrize(("name", "kind", "fid"), [
    ("mcdn_gnss", "gnss_noise", "gnss_noise"), ("mcdn_gnss_sig", "gnss_noise", "gnss_noise"),
    ("mcdn_gnss_sig_comms", "gnss_noise", "gnss_noise"),
    ("mcdn_gnss", "gnss_jump", "gnss_jump"), ("mcdn_gnss_sig", "gnss_jump", "gnss_jump"),
    ("mcdn_gnss_sig_comms", "gnss_jump", "gnss_jump")])
def test_the_gnss_faulted_vehicles_score_stands_out(tmp_path: Path, name: str, kind: str,
                                                    fid: str) -> None:
    seeds = make_batch(tmp_path, [*(["none"] * 6), kind], kepler=True)
    keep: dict[str, object] = {}
    cfg = replace(CFG, mcd_epochs=40, mcd_batch=64)
    run_detectors(tmp_path, [name], seeds[:6], seeds[6:], cfg, PARAMS_DIR, keep=keep)
    traces = keep["traces"][seeds[6]][0][name]  # type: ignore[index]
    top = {t.sat_id: float(np.max(t.score)) for t in traces if t.fault_id == fid}
    others = max(v for s, v in top.items() if s != FAULT_VEHICLE[kind])
    assert top[FAULT_VEHICLE[kind]] > 3.0 * max(others, 1.0), (kind, top)
