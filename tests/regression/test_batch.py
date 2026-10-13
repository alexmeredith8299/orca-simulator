"""M8: per-seed Parquet output, manifests, CLI batch runs and the validation report."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest

from expand3.cli import main, parse_seeds
from expand3.faults import FaultConfig
from expand3.mission.thread import run_mission
from expand3.params import Params
from expand3.recorder import DATA_FILES
from expand3.report import build_report
from expand3.run import run_batch, run_seed, seed_dir

PARAMS_DIR = Path(__file__).resolve().parents[2] / "params"
SHORT_H = 2.0
TIMING_KEYS = {"wall_s", "started_utc", "finished_utc", "workers"}


def strip_timing(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_timing(v) for k, v in obj.items() if k not in TIMING_KEYS}
    if isinstance(obj, list):
        return [strip_timing(v) for v in obj]
    return obj


@pytest.fixture(scope="module")
def batch(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("batch")
    run_batch(PARAMS_DIR, [0, 1], out, FaultConfig(), duration_h=SHORT_H, workers=1)
    return out


def test_parse_seeds() -> None:
    assert parse_seeds("0:4") == [0, 1, 2, 3]
    assert parse_seeds("7") == [7]
    assert parse_seeds("1,5,9") == [1, 5, 9]


def test_layout_and_manifest(batch: Path) -> None:
    d = seed_dir(batch, 0)
    for name in DATA_FILES:
        assert (d / name).is_file(), name
    man = json.loads((d / "manifest.json").read_text())
    for key in ("seed", "fault_config", "draws", "git", "params_sha256", "versions", "files",
                "wall_s"):
        assert key in man, key
    assert man["seed"] == 0
    assert set(man["files"]) == set(DATA_FILES)
    assert {"epoch_utc", "run_length_s", "rho0_kg_m3", "H_km", "burn_command_s",
            "burn_length_s", "fault_id", "fault_lead_s"} <= set(man["draws"])
    bm = json.loads((batch / "manifest.json").read_text())
    assert [s["seed"] for s in bm["seeds"]] == [0, 1]
    assert all(s["status"] == "ok" for s in bm["seeds"])


def test_determinism_across_workers(batch: Path, tmp_path: Path) -> None:
    other = tmp_path / "again"
    run_batch(PARAMS_DIR, [0, 1], other, FaultConfig(), duration_h=SHORT_H, workers=2)
    for seed in (0, 1):
        a, b = seed_dir(batch, seed), seed_dir(other, seed)
        for name in DATA_FILES:
            assert (a / name).read_bytes() == (b / name).read_bytes(), name
        ma = json.loads((a / "manifest.json").read_text())
        mb = json.loads((b / "manifest.json").read_text())
        assert strip_timing(ma) == strip_timing(mb)
    ba = json.loads((batch / "manifest.json").read_text())
    bb = json.loads((other / "manifest.json").read_text())
    assert strip_timing(ba) | {"out": None} == strip_timing(bb) | {"out": None}


def test_schema_units_and_rates(batch: Path, params: Params) -> None:
    d = seed_dir(batch, 0)
    truth = pq.read_table(d / "truth.parquet")
    meas = pq.read_table(d / "measurements.parquet")
    for table in (truth, meas):
        units = json.loads(table.schema.metadata[b"units"])
        assert units["t_s"] == "s"
        assert table.schema.field("sat_id").type == "string"
        assert table.schema.field("t_s").type == "double"
    tcols = set(truth.column_names)
    for col in ("r_x_km", "v_z_km_s", "q_w", "omega_y_rad_s", "h_wheel_4_Nms", "mass_kg",
                "propellant_kg", "E_J", "soc", "v_V", "i_A", "p_gen_W", "p_string_1_W",
                "shed", "uv_hold", "T_battery_K", "T_wing_ny_K", "battery_heater", "gate_pass",
                "gate_take", "gate_slew", "gate_burn", "in_saa", "B_nT", "upsets", "resets",
                "fault_s", "fault_theta_0", "fault_theta_1"):
        assert col in tcols, col
    assert truth.schema.field("shed").type == "bool"
    assert truth.schema.field("upsets").type == "int64"
    units = json.loads(truth.schema.metadata[b"units"])
    assert units["r_x_km"] == "km"
    assert units["T_battery_K"] == "K"
    cols = set(meas.column_names)
    pairs = ["gnss_r_x_km", "gnss_v_x_km_s", "st_q_w", "imu_omega_x_rad_s", "pack_v_V",
             "pack_i_A", "string_p_1_W", "T_battery_K", "soc_est", "tank_kg"]
    for base in pairs:  # truth beside every measurement
        assert f"{base}_meas" in cols, base
        assert f"{base}_true" in cols, base
    for flag in ("gnss_valid", "st_available", "tank_valid", "heater_battery_cmd",
                 "heater_battery_delivered", "field_nT", "upsets", "resets", "gate_saa"):
        assert flag in cols, flag
    t = np.asarray(meas.column("t_s"))
    sat = np.asarray(meas.column("sat_id"))
    sel = sat == "SAT-1"
    gnss = np.asarray(meas.column("gnss_valid"))[sel]
    np.testing.assert_array_equal(gnss, np.isclose(np.mod(t[sel], 60.0), 0))
    gx = np.asarray(meas.column("gnss_r_x_km_meas"), dtype=float)[sel]
    assert np.all(np.isnan(gx[~gnss]))
    assert not np.any(np.isnan(gx[gnss]))
    for col in ("imu_omega_x_rad_s_meas", "pack_v_V_meas", "T_battery_K_meas", "soc_est_meas"):
        assert not np.any(np.isnan(np.asarray(meas.column(col), dtype=float)[sel])), col
    st = np.asarray(meas.column("st_available"))[sel]
    sq = np.asarray(meas.column("st_q_w_meas"), dtype=float)[sel]
    assert np.all(np.isnan(sq[~st]))
    assert not np.any(np.isnan(sq[st]))
    xl = pq.read_table(d / "crosslink.parquet")
    assert {"t_s", "sender", "receiver", "sent", "delivered", "age_s"} <= set(xl.column_names)
    wh = pq.read_table(d / "wheel_steps.parquet")
    assert {"sat_id", "t_s", "slew", "h_1_Nms", "u_1_Nm", "tau_x_Nm", "power_W"} <= set(
        wh.column_names)


def test_round_trip(batch: Path, params: Params) -> None:
    run = run_mission(params, 1, duration_s=SHORT_H * 3600)
    d = seed_dir(batch, 1)
    truth = pq.read_table(d / "truth.parquet")
    meas = pq.read_table(d / "measurements.parquet")
    sat = np.asarray(truth.column("sat_id"))
    for k, s in enumerate(run.sats):
        sel = sat == s
        np.testing.assert_array_equal(np.asarray(truth.column("t_s"))[sel], run.t_s)
        np.testing.assert_array_equal(np.asarray(truth.column("r_y_km"))[sel], run.r_km[:, k, 1])
        np.testing.assert_array_equal(np.asarray(truth.column("q_z"))[sel], run.q[:, k, 3])
        np.testing.assert_array_equal(np.asarray(truth.column("v_V"))[sel], run.eps[s].v_V)
        np.testing.assert_array_equal(np.asarray(truth.column("T_payload_K"))[sel],
                                      run.thermal[s].node("payload"))
        np.testing.assert_array_equal(np.asarray(truth.column("upsets"))[sel], run.saa[s].upsets)
        msel = np.asarray(meas.column("sat_id")) == s
        sig = run.signals[s]
        got = np.asarray(meas.column("pack_i_A_meas"))[msel]
        np.testing.assert_array_equal(got, sig.pack_current.meas)
        gnss = np.asarray(meas.column("gnss_valid"))[msel]
        np.testing.assert_array_equal(
            np.asarray(meas.column("gnss_r_z_km_meas"), dtype=float)[msel][gnss],
            sig.gnss_r.meas[:, 2])


def test_labels_none(batch: Path) -> None:
    for seed in (0, 1):
        lab = json.loads((seed_dir(batch, seed) / "labels.json").read_text())
        man = json.loads((seed_dir(batch, seed) / "manifest.json").read_text())
        if lab["fault"] == "none":
            truth = pq.read_table(seed_dir(batch, seed) / "truth.parquet")
            assert np.all(np.asarray(truth.column("fault_s")) == 0)
        else:  # a seeded fault that has not armed inside a short run
            assert lab["fault_id"] == man["draws"]["fault_id"]


@pytest.mark.slow
def test_forced_fault_labels(tmp_path: Path, params: Params) -> None:
    run_seed(PARAMS_DIR, 20270, tmp_path, FaultConfig(mode="forced", fault_id="F9",
                                                      lead_s=10_000.0), duration_h=29.0)
    d = seed_dir(tmp_path, 20270)
    lab = json.loads((d / "labels.json").read_text())
    assert (lab["fault_id"], lab["vehicle"], lab["lead_s"]) == ("F9", "SAT-2", 10_000.0)
    assert lab["paths"] == ["cdh_upsets.lambda_in_per_s"]
    truth = pq.read_table(d / "truth.parquet")
    sat = np.asarray(truth.column("sat_id")) == "SAT-2"
    t = np.asarray(truth.column("t_s"))[sat]
    theta = np.asarray(truth.column("fault_theta_0"))[sat]
    s = np.asarray(truth.column("fault_s"))[sat]
    after = t >= lab["t_arm_s"]
    assert after.any()
    np.testing.assert_allclose(theta[after], lab["theta0"][0] * lab["magnitude"])
    np.testing.assert_allclose(theta[~after], lab["theta0"][0])
    assert np.all(s[after] == 1)
    other = np.asarray(truth.column("sat_id")) == "SAT-1"
    assert np.all(np.isnan(np.asarray(truth.column("fault_theta_0"))[other]))
    none = tmp_path / "none"
    run_seed(PARAMS_DIR, 20270, none, FaultConfig(mode="none"), duration_h=SHORT_H)
    lab_none = json.loads((seed_dir(none, 20270) / "labels.json").read_text())
    assert lab_none == {"fault": "none"}


def failing_runner(params_dir: Path, seed: int, out: Path, fault: FaultConfig,
                   duration_h: float | None = None, force: bool = False) -> dict[str, Any]:
    if seed == 1:
        raise RuntimeError("boom")
    return run_seed(params_dir, seed, out, fault, duration_h, force)


def test_resume_force_and_failures(tmp_path: Path) -> None:
    out = tmp_path / "b"
    bm = run_batch(PARAMS_DIR, [0, 1], out, FaultConfig(mode="none"), duration_h=1.0,
                   runner=failing_runner)
    status = {s["seed"]: s for s in bm["seeds"]}
    assert status[0]["status"] == "ok"
    assert status[1]["status"] == "failed"
    assert "boom" in status[1]["error"]
    again = run_batch(PARAMS_DIR, [0], out, FaultConfig(mode="none"), duration_h=1.0)
    assert again["seeds"][0]["status"] == "skipped"
    forced = run_batch(PARAMS_DIR, [0], out, FaultConfig(mode="none"), duration_h=1.0,
                       force=True)
    assert forced["seeds"][0]["status"] == "ok"


def test_cli_and_report(tmp_path: Path) -> None:
    out = tmp_path / "cli"
    assert main(["run", "--seeds", "0:2", "--out", str(out), "--duration-h", "1",
                 "--workers", "2"]) == 0
    assert main(["run", "--seeds", "0:2", "--out", str(out), "--duration-h", "1"]) == 0
    assert main(["report", str(out)]) == 0
    rep = out / "report"
    assert (rep / "report.md").is_file()
    for csv in ("fault_counts.csv", "invariants.csv", "runtime.csv"):
        assert (rep / csv).is_file(), csv
    summary = build_report(out)
    assert summary["runs"] == 2
    assert summary["invariant_pass_rate"] == 1.0
    text = (rep / "report.md").read_text()
    for heading in ("Fault draws", "Invariants", "Thermal limits and undervoltage",
                    "Runtime", "F5/F6 coverage", "Known gaps"):
        assert heading in text, heading


@pytest.mark.slow
@pytest.mark.batch100
def test_hundred_seed_batch(tmp_path_factory: pytest.TempPathFactory) -> None:
    out = Path(tmp_path_factory.mktemp("b100"))
    assert main(["run", "--seeds", "0:100", "--out", str(out), "--workers", "4"]) == 0
    assert main(["report", str(out)]) == 0
