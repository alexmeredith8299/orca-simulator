"""Monte Carlo dropout detectors for the faults that are not smooth signals (extra `detect-dl`).

`mcdrop.py` forecasts a continuous measured signal with a dropout LSTM. F2 is still that kind of
signal (string power while sunlit) and keeps the LSTM. The rest are events, counts or one value per
take, so a dropout MLP predicts the expected quantity of each sample from exogenous inputs, the
spread of its stochastic passes plus an aleatoric term fitted on held-out seeds is its uncertainty,
and the GLR that fits the quantity scores the miss:

- F3, F4: per slew step and wheel, the momentum change. A healthy wheel has dh = u dt (Model:
  actuators, k = 1, c = 0), so F4 shows as a slope of the miss on the command and F3 as a slope on
  the momentum, both per wheel.
- F8: per block of ticks, the fraction of the vehicle's incident messages that dropped (binomial).
  Like the physics detector, only the vehicle with the largest score can alarm, because a drop on
  a link counts for both its ends.
- F9: per block, the upset count while inside the SAA (Poisson). The expected count is the
  network's mean plus its epistemic std.
- F10: per take, log2 of the dark current from the payload temperature (Gaussian, one sample).
- gnss_jump, gnss_noise (extended faults): the GNSS fix against the vehicle's own navigation
  solution (`McdGnss`), with the other vehicles' fixes and solutions in the multi-satellite forms.

Each has the single-vehicle form (the vehicle's own context features and its own inputs) and, with
`_sig`, the multi-satellite form of `mcdmulti.py`: the context of all six vehicles and the other
five vehicles' measured signals, the vehicle's own masked; `_sig_comms` through the crosslink and
ground relay. For F8 and F9 the same-kind signal of a peer (its recent drop fraction, its recent
upsets) is added to what the peers contribute. A sample's inputs come from the tick at or before
its time.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from expand3.detect.base import Detector, Trace
from expand3.detect.comms import hold
from expand3.detect.data import SeedData
from expand3.detect.features import context, sunlit
from expand3.detect.glr import binomial_glr, gaussian_glr, poisson_glr, slope_glr, variance_glr
from expand3.detect.mcdmulti import ALL, _MultiView, measured_block
from expand3.detect.mcdrop import McdModel, McdThermal
from expand3.params import load_params

F64 = NDArray[np.float64]
F32 = NDArray[np.float32]
IntArr = NDArray[np.int64]
N_WHEELS = 4
N_STRINGS = 4
M_PER_KM = 1000.0  # the batch tables carry km and km/s; the misses are scored in m and m/s


def blocks(n: int, stride: int) -> tuple[IntArr, IntArr]:
    """First and last tick of each block of `stride` ticks; the last block may be partial."""
    start = np.arange(0, n, stride, dtype=np.int64)
    return start, np.minimum(start + stride, n) - 1


def trailing_sum(x: F64, window: int) -> F64:
    """Sum of the last `window` values at each tick (fewer at the start); uses no later tick."""
    c = np.cumsum(x)
    out = np.array(c, dtype=np.float64)
    out[window:] -= c[:-window]
    return out


@dataclass
class Samples:
    """One vehicle's samples of one fault: times, the tick at or before each, the fault's own
    inputs, targets (NaN where not usable) and per-sample arrays the score needs."""

    t: F64
    tick: IntArr
    local: F64
    y: F64
    aux: dict[str, F64] = field(default_factory=dict)


def _empty(k: int, n_local: int) -> Samples:
    return Samples(np.zeros(0), np.zeros(0, dtype=np.int64), np.zeros((0, n_local)),
                   np.zeros((0, k)))


class _Nn(_MultiView, Detector):
    """Shared plumbing: samples, inputs, one shared MLP for all vehicles, traces."""

    family = "bayesian"
    comms = False
    cols: tuple[str, ...] = ()
    overrides: tuple[tuple[str, Any], ...] = ()  # DetectConfig fields for this fault's model

    def samples(self, d: SeedData, sat: str) -> Samples:
        raise NotImplementedError

    def score(self, d: SeedData, per: dict[str, tuple[Samples, F64, F64]]) -> list[Trace]:
        raise NotImplementedError

    def x(self, d: SeedData, sat: str, s: Samples) -> F32:
        if self.signals:
            _, base = self.inputs(d, sat)
        else:
            _, base, _ = context(d, sat, self.cfg)
        return np.concatenate([base[s.tick], s.local], axis=1).astype(np.float32)

    def summarize(self, d: SeedData) -> dict[str, Any]:
        xs, ys = [], []
        for sat in d.sat_ids:
            s = self.samples(d, sat)
            if len(s.t):
                ok = d.fault_free(sat, s.t)
                xs.append(self.x(d, sat, s))
                ys.append(np.where(ok[:, None], s.y, np.nan))
        return {"x": xs, "y": ys, "seed": d.seed}

    def fit(self, summaries: list[dict[str, Any]]) -> dict[str, Any]:
        xs = [x for s in summaries for x in s["x"]]
        ys = [y for s in summaries for y in s["y"]]
        seeds = [s["seed"] for s in summaries for _ in s["x"]]
        cfg = replace(self.cfg, **dict(self.overrides))
        return {ALL: McdModel(cfg, list(self.cols), arch="mlp").fit(xs, ys, seeds)}

    def traces(self, d: SeedData, nominal: dict[str, Any]) -> list[Trace]:
        model: McdModel = nominal[ALL]
        per: dict[str, tuple[Samples, F64, F64]] = {}
        for sat in d.sat_ids:
            s = self.samples(d, sat)
            if len(s.t):
                mean, std = model.predict(self.x(d, sat, s), d.seed)
                per[sat] = (s, mean, std)
        return self.score(d, per)


# ---- F9: upsets while inside the SAA ---------------------------------------------------------
class McdUpsets(_Nn):
    name = "mcdn_f9"
    fault_ids = ("F9",)
    block_key = "f9"
    cols = ("upsets_per_block",)
    overrides = (("mcd_epochs", 60),)

    def signal_block(self, d: SeedData, sat: str) -> F64:
        m = d.measurements(sat)
        recent = trailing_sum(np.nan_to_num(m["upsets"]), self.cfg.mcd_stride)
        return np.concatenate([measured_block(m), recent[:, None]], axis=1)

    def samples(self, d: SeedData, sat: str) -> Samples:
        m = d.measurements(sat)
        n = len(m["t_s"])
        start, end = blocks(n, self.cfg.mcd_stride)
        length = np.add.reduceat(np.ones(n), start)
        counts = np.add.reduceat(np.nan_to_num(m["upsets"]), start)
        inside = np.add.reduceat((m["gate_saa"] > 0.5).astype(np.float64), start)
        field_nt = np.add.reduceat(np.nan_to_num(m["field_nT"]), start) / length
        y = np.where(inside > 0, counts, np.nan)[:, None]
        return Samples(m["t_s"][end], end, np.stack([inside / length, field_nt], axis=1), y,
                       {"inside": inside, "counts": counts})

    def score(self, d: SeedData, per: dict[str, tuple[Samples, F64, F64]]) -> list[Trace]:
        cfg = self.cfg
        out = []
        for sat, (s, mean, std) in per.items():
            sel = s.aux["inside"] > 0
            if not sel.any():
                continue
            dt = float(np.median(np.diff(d.measurements(sat)["t_s"])))
            floor = cfg.f9_lambda_floor_per_s * dt * s.aux["inside"]
            expected = np.maximum(mean[:, 0] + std[:, 0], floor)
            out.append(Trace(sat, "F9", s.t[sel], poisson_glr(s.aux["counts"][sel],
                                                             expected[sel], cfg.mcd_window)))
        return out


# ---- F8: dropped messages --------------------------------------------------------------------
def _incident(cache: dict[str, Any], sat: str, n: int) -> tuple[F64, F64]:
    """Per tick: messages sent on the vehicle's links, and how many of them dropped."""
    sent = np.zeros(n)
    dropped = np.zeros(n)
    for other in cache["feats"]:
        if other == sat:
            continue
        for key in ((sat, other), (other, sat)):
            sent += cache["sent"][key]
            dropped += cache["sent"][key] & ~cache["delivered"][key]
    return sent, dropped


class McdDrops(_Nn):
    name = "mcdn_f8"
    fault_ids = ("F8",)
    block_key = "f8"
    cols = ("drop_fraction",)
    overrides = (("mcd_epochs", 60),)

    def signal_block(self, d: SeedData, sat: str) -> F64:
        cache = d.__dict__["_mcdm_cache"]  # built before the blocks are
        sent, dropped = _incident(cache, sat, len(cache["t"]))
        window = self.cfg.mcd_stride
        recent = trailing_sum(dropped, window) / np.maximum(trailing_sum(sent, window), 1.0)
        return np.concatenate([measured_block(d.measurements(sat)), recent[:, None]], axis=1)

    def samples(self, d: SeedData, sat: str) -> Samples:
        cache = self._seed(d)
        t = cache["t"]
        start, end = blocks(len(t), self.cfg.mcd_stride)
        sent_t, dropped_t = _incident(cache, sat, len(t))
        sent = np.add.reduceat(sent_t, start)
        dropped = np.add.reduceat(dropped_t, start)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(sent > 0, dropped / sent, np.nan)[:, None]
        return Samples(t[end], end, (sent / self.cfg.mcd_stride)[:, None], y,
                       {"sent": sent, "dropped": dropped})

    def score(self, d: SeedData, per: dict[str, tuple[Samples, F64, F64]]) -> list[Trace]:
        cfg = self.cfg
        score: dict[str, F64] = {}
        for sat, (s, mean, std) in per.items():
            seen = s.aux["sent"] > 0
            p0 = float(np.clip(np.median((mean[:, 0] + std[:, 0])[seen]) if seen.any() else 0.0,
                               cfg.f8_p0_floor, 1.0))
            score[sat] = binomial_glr(s.aux["dropped"], s.aux["sent"], p0, cfg.mcd_window)
        best = np.max(np.stack(list(score.values())), axis=0)  # as the physics detector: a
        return [Trace(sat, "F8", per[sat][0].t, np.where(v >= best, v, 0.0))  # drop counts at both
                for sat, v in score.items()]                                 # ends of a link


# ---- F10: dark current at the take -----------------------------------------------------------
class McdDark(_Nn):
    name = "mcdn_f10"
    fault_ids = ("F10",)
    block_key = "f10"
    cols = ("log2_dark_current",)
    overrides = (("mcd_epochs", 300), ("mcd_batch", 32))

    def samples(self, d: SeedData, sat: str) -> Samples:
        c = d.sat_columns("takes", sat, ["t_take_s", "payload_K", "dark_current_e_s_px"])
        t_take = np.asarray(c["t_take_s"], dtype=np.float64)
        if not t_take.size:
            return _empty(1, 1)
        grid = d.measurements(sat)["t_s"]
        tick = np.maximum(np.searchsorted(grid, t_take, side="right") - 1, 0)
        y = np.log2(np.asarray(c["dark_current_e_s_px"], dtype=np.float64))[:, None]
        return Samples(t_take, tick.astype(np.int64),
                       np.asarray(c["payload_K"], dtype=np.float64)[:, None], y)

    def score(self, d: SeedData, per: dict[str, tuple[Samples, F64, F64]]) -> list[Trace]:
        return [Trace(sat, "F10", s.t, gaussian_glr((s.y[:, 0] - mean[:, 0]) / std[:, 0],
                                                   0.0, 1.0, 1))
                for sat, (s, mean, std) in per.items()]


# ---- F3, F4: wheel momentum steps ------------------------------------------------------------
class McdWheels(_Nn):
    name = "mcdn_wheels"
    fault_ids = ("F3", "F4")
    block_key = "wheels"
    cols = tuple(f"wheel_{j + 1}_dh_per_s" for j in range(N_WHEELS))
    overrides = (("mcd_epochs", 100), ("mcd_sigma_floor", 1e-12))

    def samples(self, d: SeedData, sat: str) -> Samples:
        names = ["slew", "t_s", *[f"h_{j + 1}_Nms" for j in range(N_WHEELS)],
                 *[f"u_{j + 1}_Nm" for j in range(N_WHEELS)]]
        c = d.sat_columns("wheel_steps", sat, names)
        rows: list[tuple[F64, F64, F64, F64]] = []
        for slew in np.unique(c["slew"]):
            sel = np.flatnonzero(c["slew"] == slew)
            sel = sel[np.argsort(np.asarray(c["t_s"], dtype=np.float64)[sel], kind="stable")]
            if sel.size < 2:
                continue
            t = np.asarray(c["t_s"], dtype=np.float64)[sel]
            h = np.stack([np.asarray(c[f"h_{j + 1}_Nms"], dtype=np.float64)[sel]
                          for j in range(N_WHEELS)], axis=1)
            u = np.stack([np.asarray(c[f"u_{j + 1}_Nm"], dtype=np.float64)[sel]
                          for j in range(N_WHEELS)], axis=1)
            dt = np.diff(t)[:, None]
            rows.append((t[1:], (h[:-1] - h[1:]) / dt, u[1:], h[:-1]))
        if not rows:
            return _empty(N_WHEELS, 2 * N_WHEELS)
        rows.sort(key=lambda r: r[0][0])
        t, y, u, h_prev = (np.concatenate([r[i] for r in rows]) for i in range(4))
        grid = d.measurements(sat)["t_s"]
        tick = np.maximum(np.searchsorted(grid, t, side="right") - 1, 0).astype(np.int64)
        return Samples(t, tick, np.concatenate([u, h_prev], axis=1), y,
                       {"u": u, "h_prev": h_prev})

    def score(self, d: SeedData, per: dict[str, tuple[Samples, F64, F64]]) -> list[Trace]:
        out = []
        for sat, (s, mean, std) in per.items():
            z = np.nan_to_num((s.y - mean) / std)
            for fid, key in (("F4", "u"), ("F3", "h_prev")):
                per_wheel = np.stack([slope_glr(z[:, j], s.aux[key][:, j], 1.0,
                                                self.cfg.nn_wheel_window)
                                      for j in range(N_WHEELS)], axis=1)
                detail = np.array([f"wheel_{int(j) + 1}" for j in per_wheel.argmax(axis=1)],
                                  dtype="U16")
                out.append(Trace(sat, fid, s.t, per_wheel.max(axis=1), detail))
        return out


# ---- F5, F6: thrust shortfall in the semi-major axis -----------------------------------------
def gnss_elements(m: dict[str, F64], mu_km3_s2: float) -> tuple[F64, F64]:
    """Semi-major axis [km] and argument of latitude [rad] from the measured fixes, NaN on ticks
    without a valid fix. The argument of latitude is measured from the ascending node of the
    fix's own orbit plane."""
    r = np.stack([m[f"gnss_r_{ax}_km_meas"] for ax in "xyz"], axis=1)
    v = np.stack([m[f"gnss_v_{ax}_km_s_meas"] for ax in "xyz"], axis=1)
    h = np.cross(r, v)
    node = np.stack([-h[:, 1], h[:, 0], np.zeros(len(h))], axis=1)
    node /= np.linalg.norm(node, axis=1, keepdims=True)
    a = 1.0 / (2.0 / np.linalg.norm(r, axis=1) - np.sum(v * v, axis=1) / mu_km3_s2)
    u = np.arctan2(np.sum(np.cross(node, r) * h, axis=1) / np.linalg.norm(h, axis=1),
                   np.sum(node * r, axis=1))
    valid = m["gnss_valid"] > 0.5
    return np.where(valid, a, np.nan), np.where(valid, u, np.nan)


class McdBurn(_Nn):
    """F5 and F6: the semi-major axis from the GNSS fixes against what the ordered delta-v so far
    should have made of it.

    A burn raises the semi-major axis by 2 a dv / v, about 1.8 km for the 1 m/s burn, while the
    short-period J2 oscillation of the osculating axis is several km. The network learns that
    oscillation from the argument of latitude and the growth from the cumulative ordered delta-v,
    on nominal data, and a delivered thrust 30 % short appears as a negative miss that grows
    with the burn. Both faults cut the delivered delta-v alike and differ only in the propellant
    flow, which the tank gauge's noise (0.1 kg against 0.04 kg drained in a burn) hides, so one
    score is reported under both ids and the two cannot be told apart from these signals.
    """

    name = "mcdn_burn"
    fault_ids = ("F5", "F6")
    block_key = "burn"
    cols = ("semi_major_axis_km",)
    overrides = (("mcd_epochs", 150), ("mcd_hidden", 64), ("mcd_sigma_floor", 1e-4))

    def _mu(self) -> float:
        assert self.params_dir is not None
        return load_params(self.params_dir).scenario.constants.mu_km3_s2

    @staticmethod
    def ordered_dv(d: SeedData, sat: str, grid: F64) -> F64:
        """Delta-v [m/s] ordered up to each tick of `grid`, from the burn table's commands."""
        c = d.sat_columns("burns", sat, ["t_s", "a_cmd_x_m_s2", "a_cmd_y_m_s2", "a_cmd_z_m_s2"])
        t = np.asarray(c["t_s"], dtype=np.float64)
        if not t.size:
            return np.zeros(len(grid))
        a = np.linalg.norm(np.stack([np.asarray(c[k], dtype=np.float64) for k in
                                     ("a_cmd_x_m_s2", "a_cmd_y_m_s2", "a_cmd_z_m_s2")], axis=1),
                           axis=1)
        order = np.argsort(t, kind="stable")
        done = np.concatenate([[0.0], np.cumsum(a[order] * float(np.median(np.diff(grid))))])
        return done[np.searchsorted(t[order], grid, side="right")]

    def signal_block(self, d: SeedData, sat: str) -> F64:
        m = d.measurements(sat)
        a, _ = gnss_elements(m, self._mu())
        idx, _ = hold(np.isfinite(a), 1.0)
        held = np.where(idx >= 0, a[np.maximum(idx, 0)], np.nan)
        return np.concatenate([measured_block(m), held[:, None],
                               self.ordered_dv(d, sat, m["t_s"])[:, None]], axis=1)

    def samples(self, d: SeedData, sat: str) -> Samples:
        m = d.measurements(sat)
        a, u = gnss_elements(m, self._mu())
        pick = np.flatnonzero(np.isfinite(a))[:: max(self.cfg.mcd_stride // 2, 1)]
        dv = self.ordered_dv(d, sat, m["t_s"])[pick]
        local = np.stack([np.sin(u[pick]), np.cos(u[pick]), np.sin(2 * u[pick]),
                          np.cos(2 * u[pick]), dv], axis=1)
        return Samples(m["t_s"][pick], pick.astype(np.int64), local, a[pick, None])

    def score(self, d: SeedData, per: dict[str, tuple[Samples, F64, F64]]) -> list[Trace]:
        out = []
        for sat, (s, mean, std) in per.items():
            z = np.nan_to_num((s.y[:, 0] - mean[:, 0]) / std[:, 0])
            score = gaussian_glr(z, 0.0, 1.0, self.cfg.nn_burn_window, side="down")
            out += [Trace(sat, fid, s.t, score) for fid in self.fault_ids]
        return out


# ---- gnss_jump, gnss_noise: the fixes against the vehicle's navigation solution -------------
class McdGnss(_Nn):
    """The vehicle's fix minus its own navigation solution, 60 s apart: three position axes in m
    and three velocity axes in m/s. The solution is the output of an EKF on the fixes with J2 and
    drag in its dynamics (`vehicle/nav.py`), so a healthy receiver leaves a residual of about the
    fix noise. A step in the position shows as a transient while the filter takes the offset in,
    a ramp mostly not at all, and noisier fixes widen both residuals. The shift of the position
    residual scores gnss_jump and the spread of the velocity residual gnss_noise (a position
    offset does not touch velocity).

    Alone, the network sees the vehicle's own context only. In `_sig`, and `_sig_comms` through
    the links, it also sees every other vehicle's raw fix, navigation solution and their
    difference (its own are the target). All six vehicles meet the same atmosphere, so a residual
    from drag that the filter's nominal density misses is common to them and a peer's residual
    predicts it.
    """

    name = "mcdn_gnss"
    fault_ids = ("gnss_jump", "gnss_noise")
    block_key = "gnss"
    cols: tuple[str, ...] = ("fix_nav_x_m", "fix_nav_y_m", "fix_nav_z_m",
                             "fix_nav_vx_m_s", "fix_nav_vy_m_s", "fix_nav_vz_m_s")
    overrides = (("mcd_epochs", 100), ("mcd_hidden", 32), ("mcd_sigma_floor", 1e-3))

    @staticmethod
    def _states(m: dict[str, F64]) -> tuple[F64, F64, F64, F64, NDArray[np.bool_]]:
        def cols(stem: str, unit: str) -> F64:
            return np.stack([m[f"{stem}_{ax}_{unit}_meas"] for ax in "xyz"], axis=1)

        return (cols("gnss_r", "km"), cols("gnss_v", "km_s"), cols("nav_r", "km"),
                cols("nav_v", "km_s"), m["gnss_valid"] > 0.5)

    def signal_block(self, d: SeedData, sat: str) -> F64:
        """Per vehicle, held between fixes: the fix, the navigation solution and the fix minus
        the solution (position in m, velocity in m/s)."""
        r, v, nr, nv, valid = self._states(d.measurements(sat))
        idx, _ = hold(valid, 1.0)
        got = (idx >= 0)[:, None]
        block = np.concatenate([r, v, nr, nv, (r - nr) * M_PER_KM, (v - nv) * M_PER_KM], axis=1)
        return np.where(got, block[np.maximum(idx, 0)], np.nan)

    def samples(self, d: SeedData, sat: str) -> Samples:
        m = d.measurements(sat)
        r, v, nr, nv, valid = self._states(m)
        pick = np.flatnonzero(valid)[self.cfg.nn_gnss_window:]  # the filter has settled
        if not pick.size:
            return _empty(6, 6)
        unit = r[pick] / np.linalg.norm(r[pick], axis=1, keepdims=True)
        vu = v[pick] / np.linalg.norm(v[pick], axis=1, keepdims=True)
        miss = np.concatenate([r[pick] - nr[pick], v[pick] - nv[pick]], axis=1) * M_PER_KM
        return Samples(m["t_s"][pick], pick.astype(np.int64),
                       np.concatenate([unit, vu], axis=1), miss)

    def score(self, d: SeedData, per: dict[str, tuple[Samples, F64, F64]]) -> list[Trace]:
        out = []
        for sat, (s, mean, std) in per.items():
            z = np.nan_to_num((s.y - mean) / std)
            shift = sum(gaussian_glr(z[:, j], 0.0, 1.0, self.cfg.nn_gnss_window)
                        for j in range(3))
            spread = variance_glr(np.mean(z[:, 3:] ** 2, axis=1), self.cfg.nn_gnss_window, 3)
            out += [Trace(sat, "gnss_jump", s.t, np.asarray(shift)),
                    Trace(sat, "gnss_noise", s.t, spread)]
        return out


# ---- F2: string power, an LSTM like the thermal detectors ------------------------------------
class McdStrings(_MultiView, McdThermal):
    """F2: each string's power while sunlit, scored for a drop."""

    name = "mcd_f2"
    fault_ids = ("F2",)
    comms = False

    def _buses(self) -> dict[str, str]:
        return _MultiView._buses(self) if self.signals else McdThermal._buses(self)

    def summarize(self, d: SeedData) -> dict[str, Any]:
        # one shared model with the other vehicles, one per bus alone, as the thermal detectors
        return _MultiView.summarize(self, d) if self.signals else McdThermal.summarize(self, d)

    def series(self, d: SeedData, sat: str) -> tuple[F64, F32, dict[str, F64], F64]:
        step = self.cfg.mcd_stride
        if self.signals:
            t, x = self.inputs(d, sat)
        else:
            t, x, _ = context(d, sat, self.cfg)
        m = d.measurements(sat)
        sun = sunlit(m, self.cfg) > 0
        y = {c: np.where(sun, m[c], np.nan)[::step] for c in
             (f"string_p_{i + 1}_W_meas" for i in range(N_STRINGS)) if not np.all(np.isnan(m[c]))}
        return t[::step], x[::step].astype(np.float32), y, np.zeros(len(t[::step]))

    def traces(self, d: SeedData, nominal: dict[str, Any]) -> list[Trace]:
        buses = self._buses()
        out = []
        for sat in d.sat_ids:
            model: McdModel = nominal[buses[sat]]
            t, x, y, _ = self.series(d, sat)
            mean, std = model.predict(x, d.seed)
            for k, c in enumerate(model.columns):
                if c not in y:
                    continue
                r = (y[c] - mean[:, k]) / std[:, k]
                sel = np.isfinite(r)
                score = gaussian_glr(r[sel], 0.0, 1.0, self.cfg.mcd_window, side="down")
                out.append(Trace(sat, "F2", t[sel], score,
                                 np.full(int(sel.sum()), f"string_{c.split('_')[2]}", dtype="U16")))
        return out


def _variants(base: Any) -> dict[str, type[Detector]]:
    """The detector alone, with the other vehicles' signals at once, and through the links."""
    sig: Any = type(f"{base.__name__}Sig", (base,), {"signals": True, "name": f"{base.name}_sig"})
    comms: Any = type(f"{base.__name__}SigComms", (sig,),
                 {"comms": True, "name": f"{base.name}_sig_comms"})
    return {c.name: c for c in (base, sig, comms)}


MCDN: dict[str, type[Detector]] = {
    k: v for base in (McdStrings, McdDrops, McdUpsets, McdDark, McdWheels, McdBurn)
    for k, v in _variants(base).items()}
MCDN |= _variants(McdGnss)
