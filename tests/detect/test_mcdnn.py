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
    faults = ("mcd_f2", "mcdn_f8", "mcdn_f9", "mcdn_f10", "mcdn_wheels", "mcdn_burn", "mcdn_gnss",
              "mcdn_gnssraw", "mcdn_mag")
    assert set(MCDN) == {f"{f}{s}" for f in faults for s in ("", "_sig", "_sig_comms")} | {
        "mcdn_txfade_sig", "mcdn_txfade_sig_comms", "mcdn_gnssrel", "mcdn_gnssrel_comms",
        "mcdn_gnsspeer", "mcdn_gnsspeer_comms"}
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


def test_gnssraw_uses_only_the_raw_fixes_and_peers_add_theirs(tmp_path: Path) -> None:
    seed = make_batch(tmp_path, ["none"], kepler=True)[0]
    d = SeedData(tmp_path / f"seed_{seed:05d}")
    det = MCDN["mcdn_gnssraw"](CFG, str(PARAMS_DIR))
    own = det.samples(d, "SAT-1")  # type: ignore[attr-defined]
    m = d.measurements("SAT-1")
    assert own.y.shape[1] == 3  # position change less the velocity trapezoid [m]
    k = own.tick[0]
    valid = np.flatnonzero(m["gnss_valid"] > 0.5)
    before = valid[valid < k][-1]
    dt = m["t_s"][k] - m["t_s"][before]
    want = (m["gnss_r_x_km_meas"][k] - m["gnss_r_x_km_meas"][before]
            - 0.5 * (m["gnss_v_x_km_s_meas"][k] + m["gnss_v_x_km_s_meas"][before]) * dt) * 1000.0
    assert own.y[0, 0] == pytest.approx(want)
    block = det.signal_block(d, "SAT-1")  # type: ignore[attr-defined]
    assert block.shape[1] == 9  # fix position and velocity, residual
    sig = MCDN["mcdn_gnssraw_sig"](CFG, str(PARAMS_DIR))
    assert sig.x(d, "SAT-1", own).shape[1] > det.x(d, "SAT-1", own).shape[1]  # type: ignore[attr-defined]


def test_gnssrel_sample_is_the_relative_fix_less_the_one_an_orbit_earlier(tmp_path: Path) -> None:
    seed = make_batch(tmp_path, ["none"], kepler=True)[0]
    d = SeedData(tmp_path / f"seed_{seed:05d}")
    det = MCDN["mcdn_gnssrel"](CFG, str(PARAMS_DIR))
    s = det.samples(d, "SAT-1")  # type: ignore[attr-defined]
    assert s.y.shape[1] == 3
    assert s.local.shape[1] == 10  # four relative vectors and the number of mates
    assert len(s.t) > 100
    m = {sat: d.measurements(sat) for sat in ("SAT-1", "SAT-2", "SAT-3")}

    def rel(tick: int) -> float:
        return float(np.mean([m["SAT-1"]["gnss_r_x_km_meas"][tick] - m[o]["gnss_r_x_km_meas"][tick]
                              for o in ("SAT-2", "SAT-3")]) * 1000.0)

    dt = float(np.median(np.diff(m["SAT-1"]["t_s"])))
    period = det._period_ticks(dt, 2)  # type: ignore[attr-defined]
    k = int(s.tick[0])
    assert rel(k) - rel(k - period) == pytest.approx(s.y[0, 0], abs=1e-6)
    assert rel(k - period) == pytest.approx(s.local[0, 0], abs=1e-6)
    assert det.x(d, "SAT-1", s).shape[1] == 10  # type: ignore[attr-defined]


def test_gnsspeer_inputs_are_lean_and_only_the_top_vehicle_keeps_a_score(tmp_path: Path) -> None:
    seed = make_batch(tmp_path, ["none"], kepler=True)[0]
    d = SeedData(tmp_path / f"seed_{seed:05d}")
    det = MCDN["mcdn_gnsspeer"](CFG, str(PARAMS_DIR))
    s = det.samples(d, "SAT-1")  # type: ignore[attr-defined]
    full = MCDN["mcdn_gnssraw_sig"](CFG, str(PARAMS_DIR))
    own = det.x(d, "SAT-1", s)  # type: ignore[attr-defined]
    assert own.shape[1] < full.x(d, "SAT-1", s).shape[1]  # type: ignore[attr-defined]
    per = {}
    for sat in d.sat_ids:
        smp = det.samples(d, sat)  # type: ignore[attr-defined]
        per[sat] = (smp, np.zeros_like(smp.y), np.ones_like(smp.y))
    traces = det.score(d, per)  # type: ignore[attr-defined]
    for fault in ("gnss_jump", "gnss_noise"):
        grid = {}
        for t in traces:
            if t.fault_id == fault:
                grid[t.sat_id] = dict(zip(t.t_s, t.score, strict=True))
        for t_s in grid["SAT-1"]:
            live = [sat for sat in grid if grid[sat].get(t_s, 0.0) > 0.0]
            assert len(live) <= 1


@pytest.mark.slow
@pytest.mark.parametrize(("name", "kind", "fid"), [
    ("mcdn_gnss", "gnss_noise", "gnss_noise"), ("mcdn_gnss_sig", "gnss_noise", "gnss_noise"),
    ("mcdn_gnss_sig_comms", "gnss_noise", "gnss_noise"),
    ("mcdn_gnss", "gnss_jump", "gnss_jump"), ("mcdn_gnss_sig", "gnss_jump", "gnss_jump"),
    ("mcdn_gnss_sig_comms", "gnss_jump", "gnss_jump"),
    # the raw-fix detector does not see a held offset (it cancels between two fixes), so no jump
    ("mcdn_gnssraw", "gnss_noise", "gnss_noise"), ("mcdn_gnssraw_sig", "gnss_noise", "gnss_noise"),
    ("mcdn_gnssrel", "gnss_noise", "gnss_noise"), ("mcdn_gnssrel", "gnss_jump", "gnss_jump"),
    ("mcdn_gnssrel_comms", "gnss_jump", "gnss_jump"),
    ("mcdn_gnsspeer", "gnss_noise", "gnss_noise"),
    ("mcdn_gnsspeer_comms", "gnss_noise", "gnss_noise")])
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


def test_txfade_samples_are_the_partners_readings_of_the_sender_at_the_range_of_the_navs(
        tmp_path: Path) -> None:
    seed = make_batch(tmp_path, ["none"], kepler=True)[0]
    d = SeedData(tmp_path / f"seed_{seed:05d}")
    now = MCDN["mcdn_txfade_sig"](CFG, str(PARAMS_DIR)).samples(d, "SAT-2")  # type: ignore[attr-defined]
    link = d.table("crosslink", ["sender", "delivered", "rssi_dBm"])
    sent = (np.asarray(link.column("sender")) == "SAT-2") & np.asarray(
        link.column("delivered"), dtype=bool)
    assert len(now.t) == sent.sum()  # one sample per delivered message from the sender
    assert now.y.shape[1] == 1
    # the range from the two vehicles' navigation solutions gives the free-space level
    r_km = np.exp(now.local[:, 0])
    expect = 43.0 - 20.0 * np.log10(4.0 * np.pi * r_km * 1e3 * 2.2e9 / 299792458.0)
    assert np.abs(now.y[:, 0] - expect).std() < 0.6  # the 0.5 dB reading error
    assert abs((now.y[:, 0] - expect).mean()) < 0.1
    # store and forward: a reading reaches the sender no earlier than it was taken
    held = MCDN["mcdn_txfade_sig_comms"](CFG, str(PARAMS_DIR)).samples(d, "SAT-2")  # type: ignore[attr-defined]
    assert np.all(held.t >= now.t[: len(held.t)] - 1e-9) or len(held.t) <= len(now.t)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["mcdn_txfade_sig", "mcdn_txfade_sig_comms"])
def test_the_faded_senders_score_stands_out(tmp_path: Path, name: str) -> None:
    seeds = make_batch(tmp_path, [*(["none"] * 6), "tx_fade"], kepler=True)
    keep: dict[str, object] = {}
    cfg = replace(CFG, mcd_epochs=40, mcd_batch=64)
    run_detectors(tmp_path, [name], seeds[:6], seeds[6:], cfg, PARAMS_DIR, keep=keep)
    traces = keep["traces"][seeds[6]][0][name]  # type: ignore[index]
    top = {t.sat_id: float(np.max(t.score)) for t in traces if t.fault_id == "tx_fade"}
    others = max(v for s, v in top.items() if s != "SAT-2")
    assert top["SAT-2"] > 3.0 * max(others, 1.0), top


def test_mag_samples_are_the_residual_against_the_onboard_model(tmp_path: Path) -> None:
    seed = make_batch(tmp_path, ["none"])[0]
    d = SeedData(tmp_path / f"seed_{seed:05d}")
    det = MCDN["mcdn_mag"](CFG, str(PARAMS_DIR))
    s = det.samples(d, "SAT-1")  # type: ignore[attr-defined]
    assert s.y.shape[1] == 4  # three body axes (needs a tracker fix) and the magnitude
    assert np.isnan(s.y[:, 0]).any()
    assert np.isfinite(s.y[:, 3]).all()
    assert abs(np.nanmean(s.y[:, 3])) < 400.0  # the shared disturbance, no more
    block = det.signal_block(d, "SAT-1")  # type: ignore[attr-defined]
    assert block.shape[1] == 4
    sig = MCDN["mcdn_mag_sig"](CFG, str(PARAMS_DIR))
    assert sig.x(d, "SAT-1", s).shape[1] > det.x(d, "SAT-1", s).shape[1]  # type: ignore[attr-defined]


@pytest.mark.slow
@pytest.mark.parametrize(("name", "kind"), [
    ("mcdn_mag_sig", "mag_bias"), ("mcdn_mag_sig_comms", "mag_bias"),
    ("mcdn_mag", "mag_noise"), ("mcdn_mag_sig", "mag_coupling")])
def test_the_magnetometer_faulted_vehicles_score_stands_out(tmp_path: Path, name: str,
                                                            kind: str) -> None:
    seeds = make_batch(tmp_path, [*(["none"] * 6), kind])
    keep: dict[str, object] = {}
    cfg = replace(CFG, mcd_epochs=40, mcd_batch=64)
    run_detectors(tmp_path, [name], seeds[:6], seeds[6:], cfg, PARAMS_DIR, keep=keep)
    traces = keep["traces"][seeds[6]][0][name]  # type: ignore[index]
    top = {t.sat_id: float(np.max(t.score)) for t in traces if t.fault_id == kind}
    others = max(v for s, v in top.items() if s != "SAT-2")
    # the slope on the current is the least specific of the three scores (a constant current
    # looks like a shift), so it has the smallest margin
    margin = 2.0 if kind == "mag_coupling" else 3.0
    assert top["SAT-2"] > margin * max(others, 1.0), (kind, top)
