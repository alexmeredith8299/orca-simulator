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
- mag_bias, mag_noise, mag_coupling (extended faults): the magnetometer against the vehicle's
  onboard field model, with the other vehicles' residuals in the multi-satellite forms.
- tx_fade (extended fault): the partners' received signal strength of the sender, which cannot
  sense its own transmit power (`McdTxFade`, `_sig` and `_sig_comms` only).
- gnss_jump, gnss_noise (extended faults): the GNSS fix against the vehicle's own navigation
  solution (`McdGnss`), with the other vehicles' fixes and solutions in the multi-satellite forms.
  `McdGnssRaw` is a learned detector on the raw fixes alone (position change less the velocity
  trapezoid, no dynamics law), with the other vehicles' raw fixes in the multi-satellite forms.
  `McdGnssRel` takes the cluster mates' relative fixes only, as one orbit's change in the
  relative vector.

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
from expand3.detect.comms import hold, relay_source
from expand3.detect.data import SeedData
from expand3.detect.features import context, sunlit
from expand3.detect.glr import binomial_glr, gaussian_glr, poisson_glr, slope_glr, variance_glr
from expand3.detect.mcdmulti import ALL, _MultiView, measured_block, slot_order
from expand3.detect.mcdrop import McdModel, McdThermal
from expand3.params import load_params

F64 = NDArray[np.float64]
F32 = NDArray[np.float32]
IntArr = NDArray[np.int64]
N_WHEELS = 4
N_STRINGS = 4
GYRO_UNIT_RAD_S = 1.0e-5  # gyro residuals are scored in units of this, about a fifth of its noise
GYRO_RATE_SCALE = 10.0  # the rate input and regressor in these units, about its typical size
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


class McdGnssRaw(_Nn):
    """A learned detector on the raw fixes alone: no navigation solution and no dynamics law. The
    sample is the change in position between the vehicle's consecutive fixes less the trapezoid
    of its two velocity fixes over the interval (three axes, m). That is quadrature of two raw
    signals, not a model of the orbit: what is left is the fix noise plus about 150 m of orbit
    curvature that the network learns from the vehicle's own raw signals (the direction of its
    fix and velocity, and the previous residual). A noisier receiver widens the miss. A held
    offset cancels in a change between two fixes, so a step shows in the one change that spans it
    and a slow ramp hardly at all.

    In `_sig`, and `_sig_comms` through the links, the inputs add every other vehicle's raw fix
    and its latest residual (the vehicle's own are masked). Receiver noise is independent between
    vehicles, so a peer can predict only what the vehicles share.
    """

    name = "mcdn_gnssraw"
    fault_ids = ("gnss_jump", "gnss_noise")
    block_key = "gnssraw"
    cols: tuple[str, ...] = ("miss_x_m", "miss_y_m", "miss_z_m")
    overrides = (("mcd_epochs", 100), ("mcd_hidden", 32), ("mcd_sigma_floor", 1e-3))

    @staticmethod
    def _changes(m: dict[str, F64]) -> tuple[F64, F64, F64, IntArr]:
        """Per consecutive pair of valid fixes, without pairs across an outage: the later fix
        (position, velocity), the change in position less the trapezoid of the velocity fixes
        [m], and the later tick."""
        r = np.stack([m[f"gnss_r_{ax}_km_meas"] for ax in "xyz"], axis=1)
        v = np.stack([m[f"gnss_v_{ax}_km_s_meas"] for ax in "xyz"], axis=1)
        idx = np.flatnonzero(m["gnss_valid"] > 0.5)
        if idx.size < 2:
            return r[:0], v[:0], r[:0], idx[:0].astype(np.int64)
        i0, i1 = idx[:-1], idx[1:]
        dt = m["t_s"][i1] - m["t_s"][i0]
        ok = dt <= 1.5 * float(np.median(dt))  # a pair across an outage is not a pair
        miss = (r[i1] - r[i0] - 0.5 * (v[i0] + v[i1]) * dt[:, None]) * M_PER_KM
        return r[i1][ok], v[i1][ok], miss[ok], i1[ok].astype(np.int64)

    def signal_block(self, d: SeedData, sat: str) -> F64:
        """Per vehicle, held between fixes: the fix and the latest residual."""
        m = d.measurements(sat)
        r, v, miss, tick = self._changes(m)
        full = np.full((len(m["t_s"]), 9), np.nan)
        full[tick] = np.concatenate([r, v, miss], axis=1)
        idx, _ = hold(np.isfinite(full[:, 0]), 1.0)
        return np.where((idx >= 0)[:, None], full[np.maximum(idx, 0)], np.nan)

    def samples(self, d: SeedData, sat: str) -> Samples:
        m = d.measurements(sat)
        r, v, miss, tick = self._changes(m)
        if len(tick) < 2:
            return _empty(3, 9)
        prev = np.concatenate([np.zeros((1, 3)), miss[:-1]], axis=0)
        local = np.concatenate([r / np.linalg.norm(r, axis=1, keepdims=True),
                                v / np.linalg.norm(v, axis=1, keepdims=True), prev], axis=1)
        return Samples(m["t_s"][tick][1:], tick[1:], local[1:], miss[1:])  # the first has no prev

    def score(self, d: SeedData, per: dict[str, tuple[Samples, F64, F64]]) -> list[Trace]:
        out = []
        for sat, (s, mean, std) in per.items():
            z = np.nan_to_num((s.y - mean) / std)
            shift = sum(gaussian_glr(z[:, j], 0.0, 1.0, self.cfg.nn_gnss_window)
                        for j in range(z.shape[1]))
            spread = variance_glr(np.mean(z**2, axis=1), self.cfg.nn_gnss_window, z.shape[1])
            out += [Trace(sat, "gnss_jump", s.t, np.asarray(shift)),
                    Trace(sat, "gnss_noise", s.t, spread)]
        return out


class McdGnssPeer(McdGnssRaw):
    """`McdGnssRaw` with a lean multi-satellite input and one alarming vehicle per tick. The
    inputs are the vehicle's own context and raw-fix inputs, and for each other vehicle only its
    raw fix, its latest residual and how old they are: not the 29 sun, gate and heater features
    of every slot, which say nothing about a receiver. A fault is on one vehicle, but its fix is
    an input to every other vehicle's network, so a healthy vehicle's prediction moves too. Only
    the vehicle with the largest score at a tick keeps it, as F8 does for a link's two ends.
    Both forms are multi-satellite: `_sig` (at once) and `_sig_comms` (through the links).
    """

    name = "mcdn_gnsspeer"
    block_key = "gnssraw"  # the same peer block as McdGnssRaw, so the cache is shared
    comms = False

    def x(self, d: SeedData, sat: str, s: Samples) -> F32:
        seed = self._seed(d)
        sc = self._scenario()
        t = seed["t"]
        dt = float(np.median(np.diff(t)))
        ticks = np.arange(len(t), dtype=np.int64)
        cluster_of = {v: k for k, c in enumerate(sc.clusters) for v in c}
        parts = [context(d, sat, self.cfg)[1][s.tick], s.local]
        for peer in slot_order(sat, sc.clusters)[1:]:
            if peer not in seed["feats"]:
                continue
            if not self.comms:
                idx, age = ticks, np.zeros(len(t))
            elif cluster_of[peer] == cluster_of[sat]:
                idx, age = hold(seed["delivered"][(peer, sat)], dt)
            else:
                idx = relay_source(seed["pass"][cluster_of[peer]], seed["pass"][cluster_of[sat]])
                age = np.where(idx >= 0, (ticks - idx) * dt, np.inf)
            block = seed["meas"][self.block_key][peer]
            held = np.where((idx >= 0)[:, None], block[np.maximum(idx, 0)], np.nan)
            cap = self.cfg.mcdm_age_cap_s
            parts += [held[s.tick], (np.minimum(age, cap) / cap)[s.tick, None]]
        return np.concatenate(parts, axis=1).astype(np.float32)

    def score(self, d: SeedData, per: dict[str, tuple[Samples, F64, F64]]) -> list[Trace]:
        n = len(self._seed(d)["t"])
        traces = {t.sat_id + t.fault_id: t for t in super().score(d, per)}
        out = []
        for fault in self.fault_ids:
            held = {}
            for sat, (s, _, _) in per.items():
                score = traces[sat + fault].score
                full = np.zeros(n)
                full[s.tick] = score
                idx, _ = hold(np.isin(np.arange(n), s.tick), 1.0)
                held[sat] = np.where(idx >= 0, full[np.maximum(idx, 0)], 0.0)
            best = np.max(np.stack(list(held.values())), axis=0)
            for sat, (s, _, _) in per.items():
                score = traces[sat + fault].score
                out.append(Trace(sat, fault, s.t, np.where(score >= best[s.tick], score, 0.0)))
        return out


class McdGnssPeerComms(McdGnssPeer):
    name = "mcdn_gnsspeer_comms"
    comms = True


class McdGnssRel(_Nn):
    """Relative raw fixes only: the vehicle's fix minus its cluster mates' fixes, from the same
    tick, with nothing absolute as an input. A held offset on one receiver moves every relative
    vector it is part of, and no mate's moves with it, which is the one way raw GPS can show it.

    The sample is the mean over the mates present of (own fix - mate fix), as one minus the same
    vector an orbit earlier (the relative orbit repeats with the orbital period, so the change
    over an orbit is small and the network learns it to about the fix noise). The inputs are
    relative too: that earlier vector, its change over the orbit before, and its slope. A step
    shows for one orbit, until the vector an orbit earlier holds it too, and a ramp for as long as
    it grows. A vehicle whose fixes carry a larger noise widens every relative vector it is in.

    Instantaneous: every mate with a valid fix at the tick. `_comms`: only mates whose message
    was delivered at that tick (F8 drops make gaps). The other cluster is out of range, so it
    contributes nothing; a vehicle with no mate has no sample.
    """

    name = "mcdn_gnssrel"
    fault_ids = ("gnss_jump", "gnss_noise")
    block_key = "gnssrel"
    cols: tuple[str, ...] = ("rel_x_m", "rel_y_m", "rel_z_m")
    overrides = (("mcd_epochs", 100), ("mcd_hidden", 32), ("mcd_sigma_floor", 1e-3))
    delivered_only = False

    def signal_block(self, d: SeedData, sat: str) -> F64:
        return np.zeros((len(d.measurements(sat)["t_s"]), 1))  # unused: the inputs are relative

    def _period_ticks(self, dt_s: float, step: int) -> int:
        """One orbit in ticks, to a whole number of fix intervals."""
        assert self.params_dir is not None
        p = load_params(self.params_dir).scenario
        a = p.orbit.a_km
        period = 2.0 * np.pi * np.sqrt(a**3 / p.constants.mu_km3_s2)
        return int(max(step, step * round(period / (step * dt_s))))

    def samples(self, d: SeedData, sat: str) -> Samples:
        seed = self._seed(d)
        t = seed["t"]
        mates = [s for c in self._scenario().clusters if sat in c for s in c
                 if s != sat and s in seed["feats"]]
        m_own = d.measurements(sat)
        r_own = np.stack([m_own[f"gnss_r_{ax}_km_meas"] for ax in "xyz"], axis=1)
        have = m_own["gnss_valid"] > 0.5
        if not mates or have.sum() < 3:
            return _empty(3, 10)
        step = int(np.median(np.diff(np.flatnonzero(have))))
        rel = np.full((len(t), len(mates), 3), np.nan)
        for j, s in enumerate(mates):
            m = d.measurements(s)
            both = have & (m["gnss_valid"] > 0.5)
            if self.delivered_only:
                both &= seed["delivered"][(s, sat)]
            r = np.stack([m[f"gnss_r_{ax}_km_meas"] for ax in "xyz"], axis=1)
            rel[both, j] = (r_own[both] - r[both]) * M_PER_KM
        here = np.isfinite(rel[:, :, 0])  # (n, mates): the mates present at each tick
        k = here.sum(axis=1)
        period = self._period_ticks(float(np.median(np.diff(t))), step)

        def at(lag: int) -> tuple[F64, NDArray[np.bool_]]:
            """Mean relative vector over the mates present now, `lag` ticks earlier; ok where
            every one of them was present then."""
            sh = np.full_like(rel, np.nan)
            sh[lag:] = rel[:-lag] if lag else rel
            ok = np.all(~here | np.isfinite(sh[:, :, 0]), axis=1) & (k > 0)
            mean = np.where(here[:, :, None], np.nan_to_num(sh), 0.0).sum(axis=1)
            return mean / np.maximum(k, 1)[:, None], ok

        now, ok0 = at(0)
        y1, ok1 = at(period)
        y2, ok2 = at(2 * period)
        ya, oka = at(period - step)
        yb, okb = at(period + step)
        pick = np.flatnonzero(ok0 & ok1 & ok2 & oka & okb)
        if not pick.size:
            return _empty(3, 10)
        local = np.concatenate([y1[pick], y1[pick] - y2[pick], ya[pick] - yb[pick],
                                k[pick, None].astype(np.float64)], axis=1)
        return Samples(t[pick], pick.astype(np.int64), local, (now - y1)[pick])

    def x(self, d: SeedData, sat: str, s: Samples) -> F32:
        return s.local.astype(np.float32)

    def score(self, d: SeedData, per: dict[str, tuple[Samples, F64, F64]]) -> list[Trace]:
        out = []
        for sat, (s, mean, std) in per.items():
            z = np.nan_to_num((s.y - mean) / std)
            shift = sum(gaussian_glr(z[:, j], 0.0, 1.0, self.cfg.nn_gnss_window)
                        for j in range(z.shape[1]))
            spread = variance_glr(np.mean(z**2, axis=1), self.cfg.nn_gnss_window, z.shape[1])
            out += [Trace(sat, "gnss_jump", s.t, np.asarray(shift)),
                    Trace(sat, "gnss_noise", s.t, spread)]
        return out


class McdGnssRelComms(McdGnssRel):
    name = "mcdn_gnssrel_comms"
    delivered_only = True


# ---- tx_fade: the partners received signal strength of a sender --------------------------
class McdTxFade(_Nn):
    """A vehicle cannot sense its own transmit power, so a fade shows only in the partners'
    received signal strength of its messages. One sample per delivered message from the sender:
    the reading, predicted from the range of the two vehicles' navigation solutions (free-space
    loss, omni antennas), so the network learns the level at each range. The sample's scores of
    one tick are combined across the partners (their misses divided by the root of their number)
    and a GLR scores a fall.

    `mcdn_txfade_sig` takes every reading when it is made. `_comms` lets a reading reach the
    sender only on the next tick a message from that partner is delivered to it, the crosslink
    path in `comms.py`; a reading with no later delivery never arrives. Alone, a vehicle has no
    reading of its own transmission, so there is no single-vehicle form.
    """

    name = "mcdn_txfade_sig"
    fault_ids = ("tx_fade",)
    block_key = "txfade"
    cols: tuple[str, ...] = ("rssi_dBm",)
    overrides = (("mcd_epochs", 100), ("mcd_hidden", 32), ("mcd_sigma_floor", 0.05))
    signals = True

    def x(self, d: SeedData, sat: str, s: Samples) -> F32:
        return s.local.astype(np.float32)

    @staticmethod
    def _nav_position(d: SeedData, sat: str, ticks: IntArr, dt_s: float) -> F64:
        """The vehicle's navigation position at each tick, the latest solution carried forward
        with its velocity (NaN before the first solution)."""
        m = d.measurements(sat)
        r = np.stack([m[f"nav_r_{a}_km_meas"] for a in "xyz"], axis=1)
        v = np.stack([m[f"nav_v_{a}_km_s_meas"] for a in "xyz"], axis=1)
        idx, _ = hold(m["gnss_valid"] > 0.5, dt_s)
        i = idx[ticks]
        pos = r[np.maximum(i, 0)] + v[np.maximum(i, 0)] * ((ticks - i) * dt_s)[:, None]
        return np.where((i >= 0)[:, None], pos, np.nan)

    def samples(self, d: SeedData, sat: str) -> Samples:
        grid = d.measurements(sat)["t_s"]
        dt = float(np.median(np.diff(grid)))
        link = d.table("crosslink", ["t_s", "sender", "receiver", "delivered", "rssi_dBm"])
        t_l = np.asarray(link.column("t_s"), dtype=np.float64)
        sender, receiver = (np.asarray(link.column(c)) for c in ("sender", "receiver"))
        ok = np.asarray(link.column("delivered"), dtype=bool)
        rssi = np.asarray(link.column("rssi_dBm"), dtype=np.float64)
        rows: list[tuple[F64, IntArr, F64, F64]] = []
        for rx in sorted(set(receiver[sender == sat])):
            sel = (sender == sat) & (receiver == rx) & ok
            tick = np.searchsorted(grid, t_l[sel]).astype(np.int64)
            back = np.zeros(len(grid), dtype=bool)  # a message from the receiver reaches the sender
            back[np.searchsorted(grid, t_l[(sender == rx) & (receiver == sat) & ok])] = True
            avail = tick
            if self.comms:
                later = np.flatnonzero(back)
                at = np.searchsorted(later, tick)
                known = at < len(later)
                tick, avail = tick[known], later[at[known]].astype(np.int64)
                sel_idx = np.flatnonzero(sel)[known]
            else:
                sel_idx = np.flatnonzero(sel)
            rng_km = np.linalg.norm(self._nav_position(d, sat, tick, dt)
                                    - self._nav_position(d, rx, tick, dt), axis=1)
            keep = np.isfinite(rng_km) & (rng_km > 0.0) & np.isfinite(rssi[sel_idx])
            rows.append((grid[avail][keep], avail[keep], rng_km[keep], rssi[sel_idx][keep]))
        if not rows or not sum(len(r[0]) for r in rows):
            return _empty(1, 2)
        t = np.concatenate([r[0] for r in rows])
        order = np.argsort(t, kind="stable")
        km = np.concatenate([r[2] for r in rows])[order]
        local = np.stack([np.log(km), 20.0 * np.log10(km) / 100.0], axis=1)
        return Samples(t[order], np.concatenate([r[1] for r in rows])[order], local,
                       np.concatenate([r[3] for r in rows])[order, None])

    def score(self, d: SeedData, per: dict[str, tuple[Samples, F64, F64]]) -> list[Trace]:
        out = []
        for sat, (s, mean, std) in per.items():
            z = np.nan_to_num((s.y[:, 0] - mean[:, 0]) / std[:, 0])
            t_u, inv = np.unique(s.t, return_inverse=True)
            total = np.bincount(inv, weights=z)
            count = np.bincount(inv)
            score = gaussian_glr(total / np.sqrt(count), 0.0, 1.0, self.cfg.nn_txfade_window,
                                 side="down")
            out.append(Trace(sat, "tx_fade", t_u, np.asarray(score)))
        return out


class McdTxFadeComms(McdTxFade):
    name = "mcdn_txfade_sig_comms"
    comms = True


# ---- mag_bias, mag_noise, mag_coupling: the field against the onboard model ---------------
class McdMag(_Nn):
    """The measured field minus the vehicle's onboard IGRF model: three body-axis residuals (only
    where a star-tracker fix gave the model an attitude) and the magnitude residual (always). The
    true field carries a disturbance the model lacks, so a healthy residual is a few hundred nT of
    drift that one vehicle cannot tell from a bias. In `_sig`, and `_sig_comms` through the links,
    the other vehicles' residuals are inputs: cluster mates cross nearly the same disturbance, so
    the network predicts it and what is left is the sensor.

    Three scores from the standardised residual: the shift (mag_bias), the slope on the vehicle's
    own pack current (mag_coupling), and the variance of its step-to-step change (mag_noise),
    which the smooth disturbance feeds little. The step is scaled by the sensor's own noise, so a
    healthy vehicle scores near 1 and a noisier one above it.
    """

    name = "mcdn_mag"
    fault_ids = ("mag_bias", "mag_coupling", "mag_noise")
    block_key = "mag"
    cols: tuple[str, ...] = ("res_x_nT", "res_y_nT", "res_z_nT", "res_abs_nT")
    overrides = (("mcd_epochs", 100), ("mcd_hidden", 32), ("mcd_sigma_floor", 1.0))

    def _noise_sigma(self) -> float:
        """The magnetometer's noise (1 sigma, per axis) from the parameters."""
        assert self.params_dir is not None
        extra = load_params(self.params_dir).links.signals["magnetometer"].model_extra
        assert extra is not None
        return float(extra["sigma_nT"])

    @staticmethod
    def _residual(m: dict[str, F64]) -> F64:
        """(n, 4): body-axis residuals (NaN without a model) and magnitude residual, NaN off the
        magnetometer's own sampling grid."""
        meas = np.stack([m[f"mag_{a}_nT_meas"] for a in "xyz"], axis=1)
        model = np.stack([m[f"mag_model_{a}_nT_meas"] for a in "xyz"], axis=1)
        mag = np.linalg.norm(meas, axis=1) - m["mag_model_abs_nT_meas"]
        return np.concatenate([meas - model, mag[:, None]], axis=1)

    def signal_block(self, d: SeedData, sat: str) -> F64:
        """The residuals held between the magnetometer's samples."""
        res = self._residual(d.measurements(sat))
        idx, _ = hold(np.isfinite(res[:, 3]), 1.0)
        return np.where((idx >= 0)[:, None], res[np.maximum(idx, 0)], np.nan)

    def samples(self, d: SeedData, sat: str) -> Samples:
        m = d.measurements(sat)
        res = self._residual(m)
        pick = np.flatnonzero(np.isfinite(res[:, 3]))[self.cfg.nn_gnss_window:]
        if not pick.size:
            return _empty(4, 2)
        amps = np.nan_to_num(m["pack_i_A_meas"][pick])
        local = np.stack([m["mag_model_abs_nT_meas"][pick] / 1.0e4, amps], axis=1)
        return Samples(m["t_s"][pick], pick.astype(np.int64), local, res[pick],
                       {"amps": amps})

    def score(self, d: SeedData, per: dict[str, tuple[Samples, F64, F64]]) -> list[Trace]:
        out = []
        w = self.cfg.nn_mag_window
        for sat, (s, mean, std) in per.items():
            z = np.nan_to_num((s.y - mean) / std)
            amps = s.aux["amps"]
            shift = sum(gaussian_glr(z[:, j], 0.0, 1.0, w) for j in range(4))
            slope = sum(slope_glr(z[:, j], amps, 1.0, w) for j in range(4))
            miss = s.y[:, 3] - mean[:, 3]  # [nT], what the network leaves
            step = np.diff(miss, prepend=miss[0]) / (np.sqrt(2.0) * self._noise_sigma())
            spread = variance_glr(step**2, w, 1)
            out += [Trace(sat, "mag_bias", s.t, np.asarray(shift)),
                    Trace(sat, "mag_coupling", s.t, np.asarray(slope)),
                    Trace(sat, "mag_noise", s.t, np.asarray(spread))]
        return out


class McdGyro(_Nn):
    """The gyro against the star tracker (extended faults gyro_bias, gyro_scale, gyro_noise and
    gyro_stuck). Two fixes one tick apart give the body rate, 2 vec(q_k^-1 q_k+1) / dt, to a few
    1e-6 rad/s, a tenth of the gyro's noise. A sample is the gyro reading minus that rate on the
    three axes, wherever both fixes exist (about a quarter of the ticks). Its inputs are the context
    features and the rate itself, so the network learns what a healthy gyro leaves, which is its
    noise and the finite difference during a slew.

    Four scores from the standardised residual: the shift (gyro_bias), the slope on the rate
    (gyro_scale), the variance (gyro_noise), and the run of exactly repeated readings (gyro_stuck),
    which a noisy sensor never produces. A reading that tracks the rate with noise leaves none of
    the four.

    In `_sig`, and `_sig_comms` through the crosslink and the ground relay (store and forward), the
    other vehicles' gyro readings and star-tracker rates are inputs. No other vehicle shares this
    vehicle's rate, so a peer can only say what the cluster does at the same time (a common
    attitude law, a burn or a slew in step), and the test is whether that shrinks the healthy
    residual.
    """

    name = "mcdn_gyro"
    fault_ids = ("gyro_bias", "gyro_scale", "gyro_noise", "gyro_stuck")
    block_key = "gyro"
    cols: tuple[str, ...] = ("res_x", "res_y", "res_z")
    overrides = (("mcd_epochs", 100), ("mcd_hidden", 32), ("mcd_sigma_floor", 0.5))

    def signal_block(self, d: SeedData, sat: str) -> F64:
        """(n, 6): the gyro reading and the star-tracker rate in the residual's units, held
        between the fixes (NaN before the first pair)."""
        m = d.measurements(sat)
        gyro = np.stack([m[f"imu_omega_{a}_rad_s_meas"] for a in "xyz"], axis=1)
        rate = self._rate(m)
        idx, _ = hold(np.all(np.isfinite(rate), axis=1), 1.0)
        held = np.where((idx >= 0)[:, None], rate[np.maximum(idx, 0)], np.nan)
        return np.concatenate([gyro, held], axis=1) / GYRO_UNIT_RAD_S / GYRO_RATE_SCALE

    @staticmethod
    def _rate(m: dict[str, F64]) -> F64:
        """(n, 3): body rate [rad/s] from the fixes at k and k + 1, stored at k + 1 (when it is
        known); NaN where either fix is missing."""
        q = np.stack([m[f"st_q_{c}_meas"] for c in "wxyz"], axis=1)
        out = np.full((len(q), 3), np.nan)
        a, b = q[:-1], q[1:]
        d = np.stack([
            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] - a[:, 2] * b[:, 3] + a[:, 3] * b[:, 2],
            a[:, 0] * b[:, 2] + a[:, 1] * b[:, 3] - a[:, 2] * b[:, 0] - a[:, 3] * b[:, 1],
            a[:, 0] * b[:, 3] - a[:, 1] * b[:, 2] + a[:, 2] * b[:, 1] - a[:, 3] * b[:, 0],
        ], axis=1)  # vector part of conj(a) b
        scalar = a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2] + a[:, 3] * b[:, 3]
        dt = np.diff(m["t_s"])[:, None]
        out[1:] = 2.0 * np.where(scalar[:, None] < 0, -d, d) / dt
        return out

    def samples(self, d: SeedData, sat: str) -> Samples:
        m = d.measurements(sat)
        gyro = np.stack([m[f"imu_omega_{a}_rad_s_meas"] for a in "xyz"], axis=1)
        rate = self._rate(m)
        same = np.all(gyro[1:] == gyro[:-1], axis=1)
        repeats = np.zeros(len(gyro))
        for n in range(1, len(gyro)):
            repeats[n] = repeats[n - 1] + 1.0 if same[n - 1] else 0.0
        pick = np.flatnonzero(np.all(np.isfinite(rate), axis=1))
        pick = pick[pick >= 1]
        if not pick.size:
            return _empty(3, 3)
        res = (gyro[pick - 1] - rate[pick]) / GYRO_UNIT_RAD_S  # the gyro read at k - 1
        local = rate[pick] / GYRO_UNIT_RAD_S / GYRO_RATE_SCALE
        return Samples(m["t_s"][pick], pick.astype(np.int64), local, res,
                       {"rate": rate[pick] / GYRO_UNIT_RAD_S, "repeats": repeats[pick - 1]})

    def score(self, d: SeedData, per: dict[str, tuple[Samples, F64, F64]]) -> list[Trace]:
        out = []
        w = self.cfg.nn_gyro_window
        for sat, (s, mean, std) in per.items():
            z = np.nan_to_num((s.y - mean) / std)
            rate = s.aux["rate"] / GYRO_RATE_SCALE
            shift = sum(gaussian_glr(z[:, j], 0.0, 1.0, w) for j in range(3))
            slope = sum(slope_glr(z[:, j], rate[:, j], 1.0, w) for j in range(3))
            spread = variance_glr(np.mean(z**2, axis=1), w, 3)
            out += [Trace(sat, "gyro_bias", s.t, np.asarray(shift)),
                    Trace(sat, "gyro_scale", s.t, np.asarray(slope)),
                    Trace(sat, "gyro_noise", s.t, np.asarray(spread)),
                    Trace(sat, "gyro_stuck", s.t, s.aux["repeats"].astype(np.float64))]
        return out


# ---- Step 2 sensor faults: the star tracker and the GNSS receiver ----------------------------
ARCSEC_PER_RAD = 206264.80624709636  # definitional
ARCSEC_PER_DEG = 3600.0  # definitional
ST_UNIT_ARCSEC = 100.0  # relative attitudes are scored in units of this, a twentieth of 0.5 deg


def _quats(m: dict[str, F64]) -> F64:
    return np.stack([m[f"st_q_{c}_meas"] for c in "wxyz"], axis=1)


def _rotvec(a: F64, b: F64) -> F64:
    """(n, 3): rotation vector of conj(a) b in rad, taken as the shorter way round (the same
    vector part as `McdGyro._rate`)."""
    d = np.stack([
        a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] - a[:, 2] * b[:, 3] + a[:, 3] * b[:, 2],
        a[:, 0] * b[:, 2] + a[:, 1] * b[:, 3] - a[:, 2] * b[:, 0] - a[:, 3] * b[:, 1],
        a[:, 0] * b[:, 3] - a[:, 1] * b[:, 2] + a[:, 2] * b[:, 1] - a[:, 3] * b[:, 0],
    ], axis=1)
    scalar = a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2] + a[:, 3] * b[:, 3]
    return np.asarray(2.0 * np.where(scalar[:, None] < 0, -d, d), dtype=np.float64)


def _nav_unit(m: dict[str, F64]) -> F64:
    """(n, 6): the unit vectors of the navigation position and velocity, held between solutions
    (zero before the first)."""
    out = []
    for name, axes in (("nav_r_{}_km_meas", "xyz"), ("nav_v_{}_km_s_meas", "xyz")):
        v = np.stack([m[name.format(a)] for a in axes], axis=1)
        idx, _ = hold(np.isfinite(v[:, 0]), 1.0)
        held = np.where((idx >= 0)[:, None], v[np.maximum(idx, 0)], 0.0)
        out.append(held / np.maximum(np.linalg.norm(held, axis=1, keepdims=True), 1e-12))
    return np.concatenate(out, axis=1)


class _Fraction(_Nn):
    """A signal that is present or missing, scored as the fraction missing in blocks of ticks:
    the network predicts the fraction from the vehicle's context and its navigation unit
    vectors, and a shift up of the standardized miss is the fault. In `_sig`, and `_sig_comms`
    through the links, the other vehicles' latest state of the same signal is an input too."""

    block_key = "fraction"
    fault_ids: tuple[str, ...] = ()
    cols: tuple[str, ...] = ("missing_fraction",)
    overrides = (("mcd_epochs", 60), ("mcd_hidden", 32), ("mcd_sigma_floor", 0.05))

    def present(self, d: SeedData, sat: str) -> tuple[F64, F64]:
        """(present, expected): 1 where the signal is there and 1 where it should be sampled."""
        raise NotImplementedError

    def signal_block(self, d: SeedData, sat: str) -> F64:
        present, expected = self.present(d, sat)
        idx, _ = hold(expected > 0.5, 1.0)  # the latest state, held between samples
        return np.where(idx >= 0, present[np.maximum(idx, 0)], 1.0)[:, None]

    def samples(self, d: SeedData, sat: str) -> Samples:
        m = d.measurements(sat)
        present, expected = self.present(d, sat)
        start, end = blocks(len(m["t_s"]), self.cfg.mcd_stride)
        trials = np.add.reduceat(expected, start)
        missing = np.add.reduceat(expected * (1.0 - present), start)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(trials > 0, missing / trials, np.nan)[:, None]
        return Samples(m["t_s"][end], end, _nav_unit(m)[end], y,
                       {"missing": missing, "trials": trials})

    def score(self, d: SeedData, per: dict[str, tuple[Samples, F64, F64]]) -> list[Trace]:
        out = []
        for sat, (s, mean, std) in per.items():
            z = np.nan_to_num((s.y[:, 0] - mean[:, 0]) / std[:, 0])
            out.append(Trace(sat, self.fault_ids[0], s.t,
                             gaussian_glr(z, 0.0, 1.0, self.cfg.mcd_window, side="up")))
        return out


class McdStAvail(_Fraction):
    """st_dropout: whether the star tracker gives a fix. Nominally it gives one about half of the
    time, the Earth and the Sun taking the other half, in a pattern that repeats every orbit. The
    network predicts the fraction of missing fixes from the orbit (the navigation unit vectors).
    A cluster mate flies the same attitude through nearly the same place, so its state is a
    strong input in the multi-satellite forms."""

    name = "mcdn_stavail"
    fault_ids = ("st_dropout",)
    block_key = "stavail"

    def present(self, d: SeedData, sat: str) -> tuple[F64, F64]:
        m = d.measurements(sat)
        return m["st_available"].astype(np.float64), np.ones(len(m["t_s"]))


class McdGnssAvail(_Fraction):
    """gnss_outage: whether the receiver gives a fix, on the ticks where it should. Nominally it
    always does, so the network predicts a fraction of zero, and any shift up is the fault."""

    name = "mcdn_gnssavail"
    fault_ids = ("gnss_outage",)
    block_key = "gnssavail"

    def present(self, d: SeedData, sat: str) -> tuple[F64, F64]:
        m = d.measurements(sat)
        assert self.params_dir is not None
        rate = float(load_params(self.params_dir).links.signals["gnss"].rate_s)
        expected = np.isclose(np.mod(m["t_s"], rate), 0.0).astype(np.float64)
        return m["gnss_valid"].astype(np.float64), expected


class McdStAtt(_Nn):
    """The star tracker's attitude against what the vehicle already knows (extended faults
    st_false_star and st_stuck). Between two fixes a few ticks apart the attitude changes by what
    the gyro read, whatever the Sun-hold law, so the jump in the fix minus the gyro's rotation is
    noise, scaled by the gyro's noise and the square root of the gap. A false star shows as one
    jump of 0.5 deg, 6 standard deviations of one gap, at the tick it starts, and the score is the
    variance of the last few jumps. A stuck tracker repeats a fix exactly, which a noisy sensor
    never does, so its score is the run of repeats, as for the stuck gyro.

    In `_sig`, and `_sig_comms` through the links, the network also predicts the vehicle's fix
    relative to the latest fix of the cluster mate closest to it in attitude. All fly the same
    Sun-hold attitude, so the relative attitude is near zero between slews, and a false star shows
    as a shift in it that stays. A healthy vehicle differs from only the one mate that has the
    false star, and the closest mate is another one. The score adds the shift of these three
    standardized components. Between slews two attitudes agree to tens of arcsec; while either
    vehicle slews, takes or burns they differ by degrees, and the network could not fit those
    samples, so a pair further apart than `nn_st_rel_gate_deg` is left out, in training and in
    testing.
    """

    name = "mcdn_statt"
    fault_ids = ("st_false_star", "st_stuck")
    block_key = "statt"
    overrides = (("mcd_epochs", 100), ("mcd_hidden", 32), ("mcd_sigma_floor", 0.5))

    @property
    def cols(self) -> tuple[str, ...]:  # type: ignore[override]
        jump = ("jump_x", "jump_y", "jump_z")
        return (*jump, "rel_x", "rel_y", "rel_z") if self.signals else jump

    def signal_block(self, d: SeedData, sat: str) -> F64:
        """(n, 4): the latest fix, held between fixes (NaN before the first)."""
        q = _quats(d.measurements(sat))
        idx, _ = hold(np.isfinite(q[:, 0]), 1.0)
        return np.where((idx >= 0)[:, None], q[np.maximum(idx, 0)], np.nan)

    def _mates(self, d: SeedData, sat: str) -> list[F64]:
        """For each cluster mate, (n, 4): its latest fix as the vehicle holds it at each tick,
        through the crosslink (`comms`) or at once, NaN where it holds none."""
        seed = self._seed(d)
        sc = self._scenario()
        n = len(seed["t"])
        dt = float(np.median(np.diff(seed["t"])))
        cluster = next(c for c in sc.clusters if sat in c)
        out = []
        for mate in [s for s in slot_order(sat, sc.clusters)[1:]
                     if s in seed["feats"] and s in cluster]:
            block = seed["meas"][self.block_key][mate]
            if self.comms:
                idx, _ = hold(seed["delivered"][(mate, sat)], dt)
            else:
                idx = np.arange(n)
            out.append(np.where((idx >= 0)[:, None], block[np.maximum(idx, 0)], np.nan))
        return out

    def samples(self, d: SeedData, sat: str) -> Samples:
        m = d.measurements(sat)
        t = m["t_s"]
        dt = float(np.median(np.diff(t)))
        q = _quats(m)
        fix = np.flatnonzero(np.isfinite(q[:, 0]))
        k = len(self.cols)
        if len(fix) < 2:
            return _empty(k, 4)
        gyro = np.stack([m[f"imu_omega_{a}_rad_s_meas"] for a in "xyz"], axis=1)
        csum = np.concatenate([np.zeros((1, 3)), np.cumsum(gyro, axis=0)])
        a, b = fix[:-1], fix[1:]
        near = (b - a) <= self.cfg.nn_st_gap_ticks
        a, b = a[near], b[near]
        if not a.size:
            return _empty(k, 4)
        gap = (b - a).astype(np.float64)
        jump = _rotvec(q[a], q[b])  # what the two fixes say
        expected = (csum[b] - csum[a]) * dt  # what the gyro read over the same ticks
        assert self.params_dir is not None
        sigma = load_params(self.params_dir).links.signals["imu"].model_extra
        assert sigma is not None
        unit = float(sigma["sigma_rad_s"]) * dt * np.sqrt(gap)[:, None]
        y = (jump - expected) / unit
        repeats = np.zeros(len(fix))
        same = np.all(q[fix[1:]] == q[fix[:-1]], axis=1)
        for n in range(1, len(fix)):
            repeats[n] = repeats[n - 1] + 1.0 if same[n - 1] else 0.0
        run = np.zeros(len(t))
        run[fix] = repeats
        local = np.concatenate([gap[:, None] / self.cfg.nn_st_gap_ticks,
                                jump * ARCSEC_PER_RAD / ST_UNIT_ARCSEC / 10.0], axis=1)
        if self.signals:
            gate = self.cfg.nn_st_rel_gate_deg * ARCSEC_PER_DEG
            rels = []
            for mate in self._mates(d, sat):
                r = _rotvec(np.nan_to_num(mate[b]), q[b]) * ARCSEC_PER_RAD
                ok = np.isfinite(mate[b][:, 0]) & (np.linalg.norm(r, axis=1) < gate)
                rels.append(np.where(ok[:, None], r / ST_UNIT_ARCSEC, np.nan))
            if rels:  # the mate closest in attitude: one with the false star is not that one
                stack = np.stack(rels)
                size = np.where(np.isfinite(stack[..., 0]), np.linalg.norm(stack, axis=2), np.inf)
                pick = np.argmin(size, axis=0)
                rel = np.take_along_axis(stack, pick[None, :, None], axis=0)[0]
            else:
                rel = np.full((len(b), 3), np.nan)
            y = np.concatenate([y, rel], axis=1)
        return Samples(t[b], b.astype(np.int64), local, y, {"repeats": run[b]})

    def score(self, d: SeedData, per: dict[str, tuple[Samples, F64, F64]]) -> list[Trace]:
        out = []
        for sat, (s, mean, std) in per.items():
            z = np.nan_to_num((s.y - mean) / std)
            burst = variance_glr(np.mean(z[:, :3] ** 2, axis=1), self.cfg.nn_st_burst, 3)
            if z.shape[1] > 3:
                burst = burst + sum(gaussian_glr(z[:, 3 + j], 0.0, 1.0, self.cfg.nn_st_window)
                                    for j in range(3))
            out += [Trace(sat, "st_false_star", s.t, np.asarray(burst)),
                    Trace(sat, "st_stuck", s.t, s.aux["repeats"].astype(np.float64))]
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
MCDN |= _variants(McdGnssRaw)
MCDN |= {c.name: c for c in (McdGnssRel, McdGnssRelComms)}
MCDN |= {c.name: c for c in (McdGnssPeer, McdGnssPeerComms)}
MCDN |= _variants(McdMag)
MCDN |= {c.name: c for c in (McdTxFade, McdTxFadeComms)}
MCDN |= _variants(McdGyro)
MCDN |= _variants(McdStAvail)
MCDN |= _variants(McdGnssAvail)
MCDN |= _variants(McdStAtt)
