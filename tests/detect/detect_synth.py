"""Synthetic seed directories in the batch layout, with known faults, for detector tests."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import lfilter

SATS = ["SAT-1", "SAT-2", "SAT-3"]
TICK_S = 30.0
END_S = 86400.0
ARM_S = 70000.0
TAKE_S = 80000.0
LAMBDA_IN = 5.5e-4
D0, T_REF, DT_D = 10.0, 293.15, 6.5
FAULT_VEHICLE = {"F1": "SAT-2", "F2": "SAT-2", "F3": "SAT-3", "F4": "SAT-3", "F7": "SAT-2",
                 "F8": "SAT-2", "F9": "SAT-2", "F10": "SAT-2"}
NODES = ["bus_px", "bus_nx", "bus_py", "bus_ny", "bus_pz", "bus_nz", "wing_px", "wing_nx",
         "wing_py", "wing_ny", "battery", "obc", "radio_pa", "payload", "bus_core"]
R_PACK = 0.08


def _wheel_rows(sat: str, slew: int, t0: float, k: np.ndarray, c: np.ndarray) -> list[dict]:
    h = np.zeros(4)
    rows = []
    for i in range(1, 41):
        u = np.array([0.07, -0.04, -0.06, 0.05]) * (1.0 if i < 20 else -1.0)
        h = h - (k * u + c * h)
        rows.append({"sat_id": sat, "slew": slew, "t_s": t0 + i,
                     **{f"h_{j + 1}_Nms": h[j] for j in range(4)},
                     **{f"u_{j + 1}_Nm": u[j] for j in range(4)}})
    return rows


def make_seed(out: Path, seed: int, fault: str) -> Path:
    """Three vehicles, a day. Fault `none` or F2, F3, F4, F8, F9, F10 on its vehicle from ARM_S."""
    rng = np.random.default_rng(seed)
    d = out / f"seed_{seed:05d}"
    d.mkdir(parents=True)
    t = np.arange(0.0, END_S + TICK_S, TICK_S)
    n = t.size
    vehicle = FAULT_VEHICLE.get(fault)
    post = t >= ARM_S
    meas: dict[str, list[np.ndarray]] = {}
    cols: dict[str, list] = {"sat_id": [], "t_s": [], "gate_saa": [], "upsets": []}
    for j in range(4):
        cols[f"string_p_{j + 1}_W_meas"] = []
    extra = ["pack_v_V_meas", "pack_i_A_meas", "soc_est_meas", "gate_pass", "gate_take",
             "gate_slew", "gate_burn", "gate_shed", "heater_battery_delivered",
             "heater_payload_delivered", "resets", "field_nT", "st_available",
             "imu_omega_x_rad_s_meas", "imu_omega_y_rad_s_meas", "imu_omega_z_rad_s_meas",
             *[f"T_{n}_K_meas" for n in NODES]]
    for c in extra:
        cols[c] = []
    for sat in SATS:
        sun = np.clip(np.sin(2 * np.pi * t / 5400.0), 0.0, None)
        cols["sat_id"] += [sat] * n
        cols["t_s"] += list(t)
        for j in range(4):
            p = 120.0 * sun * (1 + 0.01 * rng.standard_normal(n))
            if fault == "F2" and sat == vehicle and j == 0:
                p = np.where(post, 0.0, p)
            cols[f"string_p_{j + 1}_W_meas"] += list(p)
        saa = (t % 6000.0) < 900.0
        lam = np.full(n, LAMBDA_IN)
        if fault == "F9" and sat == vehicle:
            lam = np.where(post, 50 * LAMBDA_IN, lam)
        cols["gate_saa"] += list(saa)
        cols["upsets"] += list(np.where(saa, rng.poisson(lam * TICK_S), 0))
        sunf = (sun > 0).astype(float)
        slow = lfilter([0.02], [1, -0.98], sunf, zi=[0.98 * 0.6])[0]
        cur = 3.0 * sunf - 2.0
        r_pack = np.where((fault == "F1") & (sat == vehicle) & post, 3 * R_PACK, R_PACK)
        soc = 0.7 + 0.1 * np.sin(2 * np.pi * t / 20000.0) + 0.02
        cols["pack_i_A_meas"] += list(cur + 0.02 * rng.standard_normal(n))
        cols["pack_v_V_meas"] += list(28.0 + 8 * soc - cur * r_pack + 0.04 * rng.standard_normal(n))
        cols["soc_est_meas"] += list(soc)
        shift = np.where((fault == "F7") & (sat == vehicle) & post, -1.5, 0.0)
        for k, node in enumerate(NODES):
            temp = 285.0 + k + (4.0 + 0.2 * k) * slow + shift + 0.5 * rng.standard_normal(n)
            cols[f"T_{node}_K_meas"] += list(temp)
        pas = (t % 6000.0 >= 3000.0) & (t % 6000.0 < 3600.0)
        take = (t >= TAKE_S) & (t < TAKE_S + 60.0)
        for g, v in (("gate_pass", pas), ("gate_take", take), ("gate_slew", take),
                     ("gate_burn", np.zeros(n, bool)), ("gate_shed", np.zeros(n, bool)),
                     ("heater_battery_delivered", np.zeros(n, bool)),
                     ("heater_payload_delivered", np.zeros(n, bool))):
            cols[g] += list(v.astype(float))
        cols["resets"] += [0.0] * n
        cols["field_nT"] += list(np.where(saa, 21000.0, 30000.0) + rng.standard_normal(n))
        cols["st_available"] += list((rng.random(n) < 0.5).astype(float))
        for ax in "xyz":
            cols[f"imu_omega_{ax}_rad_s_meas"] += list(5e-5 * rng.standard_normal(n))
    del meas
    pq.write_table(pa.table(cols), d / "measurements.parquet")

    link: dict[str, list] = {k: [] for k in ("t_s", "sender", "receiver", "sent", "delivered")}
    for a in SATS:
        for b in SATS:
            if a == b:
                continue
            p_drop = np.zeros(n)
            if fault == "F8" and vehicle in (a, b):
                p_drop = np.where(post, 0.9, 0.0)
            link["t_s"] += list(t)
            link["sender"] += [a] * n
            link["receiver"] += [b] * n
            link["sent"] += [True] * n
            link["delivered"] += list(rng.random(n) >= p_drop)
    pq.write_table(pa.table(link), d / "crosslink.parquet")

    take_rows, wheel_rows = [], []
    for i, sat in enumerate(SATS):
        temp = 283.0 + rng.uniform(-3, 3)
        model = D0 * 2.0 ** ((temp - T_REF) / DT_D)
        dark = model * (4.0 if fault == "F10" and sat == vehicle else 1.0)
        take_rows.append({"sat_id": sat, "t_take_s": TAKE_S + 30.0 * i, "payload_K": temp,
                          "dark_current_e_s_px": dark,
                          "slew_in_start_s": TAKE_S + 30.0 * i - 40.0})
        k, c = np.ones(4), np.zeros(4)
        if sat == vehicle and fault == "F3":
            c[0] = 0.02
        if sat == vehicle and fault == "F4":
            k[1] = 0.5
        for s in range(2):
            wheel_rows += _wheel_rows(sat, s, TAKE_S - 40.0 + 100.0 * s + 30.0 * i, k, c)
    pq.write_table(pa.table({k: [r[k] for r in take_rows] for k in take_rows[0]}),
                   d / "takes.parquet")
    pq.write_table(pa.table({k: [r[k] for r in wheel_rows] for k in wheel_rows[0]}),
                   d / "wheel_steps.parquet")
    pq.write_table(pa.table({"sat_id": ["SAT-2"], "t_s": [ARM_S + 600.0]}), d / "burns.parquet")
    label: dict = {"fault": "none"}
    if fault != "none":
        label = {"fault": fault, "fault_id": fault, "vehicle": vehicle, "t_arm_s": ARM_S}
    (d / "labels.json").write_text(json.dumps(label))
    (d / "manifest.json").write_text("{}")
    return d


def make_batch(out: Path, faults: list[str]) -> list[int]:
    for seed, fault in enumerate(faults):
        make_seed(out, seed, fault)
    return list(range(len(faults)))
