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
FAULT_VEHICLE = {"gnss_jump": "SAT-2", "gnss_noise": "SAT-2", "tx_fade": "SAT-2",
                 "gyro_bias": "SAT-2", "gyro_scale": "SAT-2", "gyro_noise": "SAT-3",
                 "gyro_stuck": "SAT-3", "mag_bias": "SAT-2", "mag_noise": "SAT-2",
                 "mag_coupling": "SAT-2",
                 "F1": "SAT-2", "F2": "SAT-2", "F3": "SAT-3", "F4": "SAT-3", "F5": "SAT-2",
                 "F6": "SAT-2", "F7": "SAT-2", "F8": "SAT-2", "F9": "SAT-2",
                 "F10": "SAT-2"}
MU, A_KM, PERIOD_S = 398600.4418, 6903.137, 5700.0
INCL = np.deg2rad(97.5)
BURNS = {"SAT-2": (ARM_S + 600.0, END_S), "SAT-3": (30000.0, 50000.0)}  # [start, end) in s
A_CMD = 8.0e-5  # ordered acceleration [m/s^2]: the synthetic burn is shorter than a real one
THRUST_FAULT = 0.7  # delivered fraction after arm under F5 and F6

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


def _burn_rows(t: np.ndarray) -> dict[str, np.ndarray]:
    """Burn table rows (the ordered acceleration along +x) for every burning vehicle."""
    rows: dict[str, list] = {k: [] for k in ("sat_id", "t_s", "a_cmd_x_m_s2", "a_cmd_y_m_s2",
                                             "a_cmd_z_m_s2")}
    for sat, (b0, b1) in BURNS.items():
        tb = t[(t >= b0) & (t < b1)]
        rows["sat_id"] += [sat] * tb.size
        rows["t_s"] += list(tb)
        rows["a_cmd_x_m_s2"] += [A_CMD] * tb.size
        rows["a_cmd_y_m_s2"] += [0.0] * tb.size
        rows["a_cmd_z_m_s2"] += [0.0] * tb.size
    return {k: np.array(v) for k, v in rows.items()}


def _gnss(t: np.ndarray, k: int, fault: str, faulted: bool, sat: str, rng: np.random.Generator
          ) -> dict[str, np.ndarray]:
    """Noisy position and velocity fixes on a circular orbit whose semi-major axis has a short
    period oscillation and grows with the delta-v the burn delivers (less after an F5 or F6)."""
    u = 2 * np.pi * t / PERIOD_S + 0.03 * k
    dv = np.zeros_like(t)
    if sat in BURNS:
        b0, b1 = BURNS[sat]
        factor = np.where((fault in ("F5", "F6")) & faulted & (t >= ARM_S), THRUST_FAULT, 1.0)
        dv = np.cumsum(np.where((t >= b0) & (t < b1), A_CMD * factor * TICK_S, 0.0))
    a_osc = A_KM + 5.0 * np.sin(2 * u) + 1.816 * dv  # km; 2 a dv / v with v = 7.6 km/s
    r = a_osc  # circular: the radius is the semi-major axis here
    v = np.sqrt(MU * (2.0 / r - 1.0 / a_osc))
    pos = np.stack([r * np.cos(u), r * np.sin(u) * np.cos(INCL), r * np.sin(u) * np.sin(INCL)])
    vel = np.stack([-v * np.sin(u), v * np.cos(u) * np.cos(INCL), v * np.cos(u) * np.sin(INCL)])
    out = {"gnss_valid": (np.arange(t.size) % 2 == 0).astype(float)}
    for i, ax in enumerate("xyz"):
        out[f"gnss_r_{ax}_km_meas"] = pos[i] + 1.5e-3 * rng.standard_normal(t.size)
        out[f"gnss_v_{ax}_km_s_meas"] = vel[i] + 5e-5 * rng.standard_normal(t.size)
    return out


def _kepler(t: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Truth position and velocity [km, km/s] on a circular orbit, one phase per vehicle."""
    w = 2 * np.pi / PERIOD_S
    u = w * t + 0.03 * k
    r = A_KM * np.stack([np.cos(u), np.sin(u) * np.cos(INCL), np.sin(u) * np.sin(INCL)], axis=1)
    v = A_KM * w * np.stack([-np.sin(u), np.cos(u) * np.cos(INCL), np.cos(u) * np.sin(INCL)],
                            axis=1)
    return r, v


def _mag_disturbance(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A smooth field error shared by every vehicle, about 300 nT rms, drifting over hours."""
    out = np.zeros((t.size, 3))
    for period_h in (1.5, 3.0, 5.0, 8.0):
        out += rng.normal(0.0, 130.0, 3) * np.sin(2 * np.pi * t[:, None] / (period_h * 3600.0)
                                                  + rng.uniform(0.0, 6.28, 3))
    return out


def _mag(t: np.ndarray, k: int, fault: str, faulted: bool, common: np.ndarray,
         amps: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Magnetometer and onboard-model columns on every other tick: the model is the ideal field,
    the truth adds the common disturbance (plus 30 nT of the vehicle's own), the reading 30 nT of
    noise, and the faults of the table from ARM_S (bias 300 nT, noise x5, 100 nT per A)."""
    ideal = np.array([20000.0, -5000.0, 30000.0]) + 3000.0 * np.sin(
        2 * np.pi * t[:, None] / PERIOD_S + 0.03 * k + np.array([0.0, 1.0, 2.0]))
    own = 30.0 * np.sin(2 * np.pi * t[:, None] / 14000.0 + k)
    post = ((t >= ARM_S) & faulted)[:, None]
    mult = np.where(post & (fault == "mag_noise"), 5.0, 1.0)
    bias = np.where(post & (fault == "mag_bias"), 300.0, 0.0)
    coupling = np.where(post & (fault == "mag_coupling"), 100.0 * amps[:, None], 0.0)
    meas = ideal + common + own + 30.0 * rng.standard_normal((t.size, 3)) * mult + bias + coupling
    on = np.arange(t.size) % 2 == 0
    tracker = on & (rng.random(t.size) < 0.5)
    out: dict[str, np.ndarray] = {}
    for i, ax in enumerate("xyz"):
        out[f"mag_{ax}_nT_meas"] = np.where(on, meas[:, i], np.nan)
        out[f"mag_model_{ax}_nT_meas"] = np.where(tracker, ideal[:, i], np.nan)
    out["mag_model_abs_nT_meas"] = np.where(on, np.linalg.norm(ideal, axis=1), np.nan)
    return out


def _gnss_kepler(t: np.ndarray, k: int, fault: str, faulted: bool, rng: np.random.Generator,
                 ) -> dict[str, np.ndarray]:
    """Fixes of the consistent orbit with the GNSS faults: a 10 m offset on every axis from ARM_S
    (gnss_jump), or five times the noise (gnss_noise)."""
    r, v = _kepler(t, k)
    post = (t >= ARM_S) & faulted
    mult = np.where(post & (fault == "gnss_noise"), 5.0, 1.0)[:, None]
    offset = np.where(post & (fault == "gnss_jump"), 10.0e-3, 0.0)[:, None]
    r = r + 1.5e-3 * rng.standard_normal(r.shape) * mult + offset
    v = v + 5e-5 * rng.standard_normal(v.shape) * mult
    valid = np.arange(t.size) % 2 == 0
    out = {"gnss_valid": valid.astype(float)}
    # the vehicle's navigation solution: the truth plus a first-order filter (gain 0.3) of the
    # fixes' error, so noise is averaged down and an offset is taken in over a few fixes
    truth_r, truth_v = _kepler(t, k)
    nr, nv = truth_r.copy(), truth_v.copy()
    ok = np.flatnonzero(valid)
    er, ev = np.zeros(3), np.zeros(3)
    for j, i in enumerate(ok):
        gain = 1.0 if j == 0 else 0.3
        er = er + gain * ((r[i] - truth_r[i]) - er)
        ev = ev + gain * ((v[i] - truth_v[i]) - ev)
        nr[i], nv[i] = truth_r[i] + er, truth_v[i] + ev
    for i, ax in enumerate("xyz"):
        out[f"gnss_r_{ax}_km_meas"] = r[:, i]
        out[f"gnss_v_{ax}_km_s_meas"] = v[:, i]
        out[f"nav_r_{ax}_km_meas"] = nr[:, i]
        out[f"nav_v_{ax}_km_s_meas"] = nv[:, i]
    return out


def _gyro(t: np.ndarray, k: int, fault: str, faulted: bool, available: np.ndarray,
          rng: np.random.Generator) -> dict[str, np.ndarray]:
    """The body rate, the gyro that reads it and the star tracker's attitude fixes. The rate is a
    slow oscillation of the size of the orbit rate; the fixes are the attitude that rate
    integrates to, with the tracker's noise, and absent where the tracker has no fix."""
    n = t.size
    phase = rng.uniform(0.0, 2 * np.pi, 3)
    omega = 1.5e-4 * np.sin(2 * np.pi * t[:, None] / PERIOD_S + phase)
    q = np.zeros((n, 4))
    q[0] = [1.0, 0.0, 0.0, 0.0]
    for i in range(n - 1):
        w = np.linalg.norm(omega[i])
        half = 0.5 * w * TICK_S
        dq = np.concatenate([[np.cos(half)], np.sin(half) * omega[i] / max(w, 1e-12)])
        a = q[i]
        q[i + 1] = [a[0] * dq[0] - a[1] * dq[1] - a[2] * dq[2] - a[3] * dq[3],
                    a[0] * dq[1] + a[1] * dq[0] + a[2] * dq[3] - a[3] * dq[2],
                    a[0] * dq[2] - a[1] * dq[3] + a[2] * dq[0] + a[3] * dq[1],
                    a[0] * dq[3] + a[1] * dq[2] - a[2] * dq[1] + a[3] * dq[0]]
    q += 1.0e-5 * rng.standard_normal((n, 4))  # tracker noise, about 2 arcsec
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    post = (t >= ARM_S) & faulted
    scale = np.where(post & (fault == "gyro_scale"), 1.25, 1.0)[:, None]
    bias = np.where(post & (fault == "gyro_bias"), 5.0e-4, 0.0)[:, None]
    mult = np.where(post & (fault == "gyro_noise"), 5.0, 1.0)[:, None]
    meas = omega * scale + bias + 5.0e-5 * mult * rng.standard_normal((n, 3))
    if faulted and fault == "gyro_stuck":
        first = int(np.argmax(t >= ARM_S))
        meas[first:] = meas[first - 1]
    out = {f"imu_omega_{ax}_rad_s_meas": meas[:, i] for i, ax in enumerate("xyz")}
    for i, c in enumerate("wxyz"):
        out[f"st_q_{c}_meas"] = np.where(available > 0, q[:, i], np.nan)
    return out


def make_seed(out: Path, seed: int, fault: str, kepler: bool = False) -> Path:
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
    gyro_rng = np.random.default_rng(seed + 50_000)  # its own stream, as the fixes
    for k, sat in enumerate(SATS):
        g = _gyro(t, k, fault, sat == vehicle,
                  np.array(cols["st_available"][k * n:(k + 1) * n]), gyro_rng)
        for name, v in g.items():
            if name not in cols:
                cols[name] = [0.0] * (n * len(SATS))
            cols[name][k * n:(k + 1) * n] = list(v)
    nav_rng = np.random.default_rng(seed + 10_000)  # its own stream: the others are unchanged
    nav = [_gnss_kepler(t, k, fault, sat == vehicle, nav_rng) if kepler
           else _gnss(t, k, fault, sat == vehicle, sat, nav_rng) for k, sat in enumerate(SATS)]
    for name in nav[0]:
        cols[name] = [v for fixes in nav for v in fixes[name]]
    mag_rng = np.random.default_rng(seed + 40_000)  # its own stream, as the fixes
    common = _mag_disturbance(t, mag_rng)
    mags = [_mag(t, k, fault, sat == vehicle, common,
                 np.array(cols["pack_i_A_meas"][k * n:(k + 1) * n]), mag_rng)
            for k, sat in enumerate(SATS)]
    for name in mags[0]:
        cols[name] = [v for m in mags for v in m[name]]
    pq.write_table(pa.table(cols), d / "measurements.parquet")

    link: dict[str, list] = {k: [] for k in ("t_s", "sender", "receiver", "sent", "delivered")}
    if kepler:
        link["rssi_dBm"] = []
    rssi_rng = np.random.default_rng(seed + 30_000)  # its own stream, as the fixes
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
            if kepler:  # 37 dBm and 3 dBi each end, 2.2 GHz free space, 0.5 dB reading error
                dist = np.linalg.norm(_kepler(t, SATS.index(a))[0] - _kepler(t, SATS.index(b))[0],
                                      axis=1)
                fspl = 20.0 * np.log10(4.0 * np.pi * dist * 1e3 * 2.2e9 / 299792458.0)
                fade = np.where(post & (fault == "tx_fade") & (a == vehicle), 6.0, 0.0)
                link["rssi_dBm"] += list(43.0 - fspl - fade + 0.5 * rssi_rng.standard_normal(n))
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
    pq.write_table(pa.table(_burn_rows(t)), d / "burns.parquet")
    label: dict = {"fault": "none"}
    if fault != "none":
        label = {"fault": fault, "fault_id": fault, "vehicle": vehicle, "t_arm_s": ARM_S}
    (d / "labels.json").write_text(json.dumps(label))
    (d / "manifest.json").write_text("{}")
    return d


def make_batch(out: Path, faults: list[str], kepler: bool = False) -> list[int]:
    for seed, fault in enumerate(faults):
        make_seed(out, seed, fault, kepler)
    return list(range(len(faults)))
