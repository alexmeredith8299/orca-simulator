"""Monte Carlo dropout sequence forecasters (extra `detect-dl`).

After Sadr, Zhu and Hu, "An Anomaly Detection Method for Satellites Using Monte Carlo Dropout"
(2022): an LSTM with dropout inside its layers and before the output is kept stochastic at test
time, and the spread of repeated forward passes is the model's uncertainty. Here the network
predicts a measured signal from a window of exogenous inputs, and the standardized miss goes to
the same GLR as the other learned residuals.

The paper forecasts a signal from its own history. That adapts to a slow drift, which is the
form of the faults tested here (F7 cooling, F1 resistance ramp), so the window holds exogenous
inputs only, as in `learned.py`.

The predictive variance is the passes' variance (epistemic) plus an aleatoric term fitted on
tune seeds the network did not train on, which is what the passes' variance misses: sensor noise
and model error. Training is on nominal data only, with separate models per bus.

- `mcd_thermal` (F7): every node's temperature from the context window.
- `mcd_eps` (F1): pack voltage from the window of measured current, SoC estimate and battery
  temperature plus context; GLR on the slope of the standardized miss on the current.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from expand3.detect.base import Detector, Trace
from expand3.detect.config import DetectConfig
from expand3.detect.data import SeedData
from expand3.detect.features import bus_of, context, node_columns
from expand3.detect.glr import gaussian_glr, slope_glr
from expand3.detect.learned import _voltage_inputs

F64 = NDArray[np.float64]
F32 = NDArray[np.float32]
PREDICT_BATCH = 4096


def _torch() -> Any:
    try:
        import torch
    except ImportError as err:  # pragma: no cover
        raise ImportError("the mcd detectors need torch: pip install -e '.[detect-dl]'") from err
    return torch


def pick_device(cfg: DetectConfig) -> Any:
    torch = _torch()
    if cfg.mcd_device != "auto":
        return torch.device(cfg.mcd_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _network(torch: Any, n_in: int, n_out: int, cfg: DetectConfig) -> Any:
    nn = torch.nn

    class Net(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(n_in, cfg.mcd_hidden, cfg.mcd_layers, batch_first=True,
                                dropout=cfg.mcd_dropout)
            self.drop = nn.Dropout(cfg.mcd_dropout)
            self.head = nn.Linear(cfg.mcd_hidden, n_out)

        def forward(self, x: Any) -> Any:
            out, _ = self.lstm(x)
            return self.head(self.drop(out[:, -1]))

    return Net()


def _windows(x: F32, rows: NDArray[np.int_], start: NDArray[np.int_], lags: int) -> F32:
    """(len(rows), lags, F) windows ending at `rows`, oldest first, clamped at the series start."""
    idx = np.maximum(rows[:, None] - np.arange(lags - 1, -1, -1)[None, :], start[rows][:, None])
    return np.asarray(x[idx], dtype=np.float32)


class McdModel:
    """A dropout LSTM and its calibration. `predict` returns the mean and the predictive std."""

    def __init__(self, cfg: DetectConfig, columns: list[str]) -> None:
        self.cfg = cfg
        self.columns = columns
        self.net: Any = None
        self.n_in = 0
        self.state: dict[str, Any] | None = None
        self.x_mu: F32 = np.zeros(0, dtype=np.float32)
        self.x_sd: F32 = np.ones(0, dtype=np.float32)
        self.y_mu: F64 = np.zeros(0)
        self.y_sd: F64 = np.ones(0)
        self.sigma_ale: F64 = np.zeros(0)
        self.coverage: float = float("nan")

    def __getstate__(self) -> dict[str, Any]:
        """The network class is local to `_network`, so pickle its weights and rebuild it."""
        state = dict(self.__dict__)
        if self.net is not None:
            state["state"] = {k: v.cpu() for k, v in self.net.state_dict().items()}
        state["net"] = None
        return state

    def _network(self, n_out: int) -> Any:
        if self.net is None:
            assert self.state is not None
            self.net = _network(_torch(), self.n_in, n_out, self.cfg)
            self.net.load_state_dict(self.state)
        return self.net

    def _standardize(self, x: F32) -> F32:
        """Standardized inputs; a missing input is zero, the training mean."""
        z = ((x - self.x_mu) / self.x_sd).astype(np.float32)
        return np.nan_to_num(z, nan=0.0) if np.isnan(z).any() else z

    # ---- training -------------------------------------------------------------------------
    def fit(self, series: list[F32], targets: list[F64], seeds: list[int]) -> McdModel:
        """`series[i]` is (n_i, F) inputs, `targets[i]` (n_i, K) with NaN where not usable."""
        torch = _torch()
        cfg = self.cfg
        torch.set_num_threads(cfg.mcd_threads)
        device = pick_device(cfg)
        x = np.concatenate(series)
        y = np.concatenate(targets)
        g = np.concatenate([np.full(len(s), sd) for s, sd in zip(series, seeds, strict=True)])
        start = np.concatenate([np.full(len(s), off) for s, off in
                                zip(series, np.cumsum([0, *[len(s) for s in series[:-1]]]),
                                    strict=True)]).astype(np.int64)
        if np.isnan(x).any():  # inputs not yet received; filled with the training mean below
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self.x_mu = np.nan_to_num(np.nanmean(x, axis=0)).astype(np.float32)
                sd = np.nan_to_num(np.nanstd(x, axis=0))
        else:
            self.x_mu, sd = x.mean(axis=0), x.std(axis=0)
        self.x_sd = np.where(sd > 0, sd, 1.0).astype(np.float32)
        self.y_mu = np.nanmean(y, axis=0)
        self.y_sd = np.maximum(np.nanstd(y, axis=0), cfg.mcd_sigma_floor)
        xs = self._standardize(x)
        ys = ((y - self.y_mu) / self.y_sd).astype(np.float32)
        usable = np.any(np.isfinite(ys), axis=1)
        val = (g % cfg.mcd_val_mod == 0) & usable
        train_rows = np.flatnonzero(usable & ~val)
        val_rows = np.flatnonzero(val)
        if len(val_rows) == 0 or len(train_rows) == 0:
            train_rows = val_rows = np.flatnonzero(usable)
        torch.manual_seed(cfg.random_seed)
        rng = np.random.default_rng(cfg.random_seed)
        self.n_in = xs.shape[1]
        net = _network(torch, self.n_in, ys.shape[1], cfg).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=cfg.mcd_lr, weight_decay=cfg.mcd_weight_decay)
        y_t = torch.from_numpy(np.nan_to_num(ys))
        ok_t = torch.from_numpy(np.isfinite(ys).astype(np.float32))
        for _ in range(cfg.mcd_epochs):
            net.train()
            order = rng.permutation(train_rows)[: cfg.mcd_max_rows]
            for b in range(0, len(order), cfg.mcd_batch):
                rows = order[b: b + cfg.mcd_batch]
                xb = torch.from_numpy(_windows(xs, rows, start, cfg.mcd_lags)).to(device)
                err = (net(xb) - y_t[rows].to(device)) * ok_t[rows].to(device)
                loss = (err ** 2).sum() / ok_t[rows].sum().clamp(min=1.0).to(device)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net = net.to("cpu")
        # aleatoric term and coverage from the seeds the network did not train on
        mean, var_e = self._mc(xs, val_rows, start)
        resid = (ys[val_rows] - mean)
        ok = np.isfinite(resid)
        total = np.array([np.var(resid[ok[:, k], k]) if ok[:, k].any() else 1.0
                          for k in range(ys.shape[1])])
        epi = np.array([np.mean(var_e[ok[:, k], k]) if ok[:, k].any() else 0.0
                        for k in range(ys.shape[1])])
        floor = (cfg.mcd_sigma_floor / self.y_sd) ** 2
        self.sigma_ale = np.sqrt(np.maximum(total - epi, floor))
        z = resid / np.sqrt(self.sigma_ale ** 2 + var_e)
        self.coverage = float(np.mean(np.abs(z[ok]) < 1.96)) if ok.any() else float("nan")
        return self

    # ---- inference ------------------------------------------------------------------------
    def _mc(self, xs: F32, rows: NDArray[np.int_], start: NDArray[np.int_]
            ) -> tuple[F64, F64]:
        """Mean and variance (standardized units) over the stochastic passes."""
        torch = _torch()
        cfg = self.cfg
        device = pick_device(cfg)
        net = self._network(len(self.columns)).to(device)
        net.train()  # dropout stays on
        mean = np.zeros((len(rows), len(self.columns)))
        var = np.zeros_like(mean)
        with torch.no_grad():
            for b in range(0, len(rows), PREDICT_BATCH):
                sel = rows[b: b + PREDICT_BATCH]
                xb = torch.from_numpy(_windows(xs, sel, start, cfg.mcd_lags)).to(device)
                passes = torch.stack([net(xb) for _ in range(cfg.mcd_samples)]).cpu().numpy()
                mean[b: b + len(sel)] = passes.mean(axis=0)
                var[b: b + len(sel)] = passes.var(axis=0)
        self.net = net.to("cpu")
        return mean, var

    def predict(self, x: F32, seed: int) -> tuple[F64, F64]:
        """(mean, std) in signal units for every row of one series, `seed` fixing the passes."""
        torch = _torch()
        cfg = self.cfg
        torch.set_num_threads(1 if pick_device(cfg).type == "cpu" else cfg.mcd_threads)
        self._network(len(self.columns))  # build before seeding: its init draws random numbers
        torch.manual_seed(cfg.random_seed + seed)
        xs = self._standardize(x)
        rows = np.arange(len(xs))
        mean, var_e = self._mc(xs, rows, np.zeros(len(xs), dtype=np.int64))
        std = np.sqrt(self.sigma_ale ** 2 + var_e) * self.y_sd
        return mean * self.y_sd + self.y_mu, std


class _Mcd(Detector):
    """Shared plumbing: stride-subsampled series per vehicle, one model per bus."""

    family = "bayesian"
    target_names: str = ""

    def _buses(self) -> dict[str, str]:
        assert self.params_dir is not None
        return bus_of(self.params_dir)

    def series(self, d: SeedData, sat: str) -> tuple[F64, F32, dict[str, F64], F64]:
        """(t_s, inputs, targets by column, current) at every `mcd_stride`-th tick."""
        raise NotImplementedError

    def summarize(self, d: SeedData) -> dict[str, Any]:
        buses = self._buses()
        out: dict[str, Any] = {}
        for sat in d.sat_ids:
            t, x, y, _ = self.series(d, sat)
            ok = d.fault_free(sat, t)
            cols = sorted(y)
            targets = np.stack([np.where(ok, y[c], np.nan) for c in cols], axis=1)
            part = out.setdefault(buses[sat], {"cols": cols, "x": [], "y": [], "seed": d.seed})
            part["x"].append(x)
            part["y"].append(targets)
        return out

    def fit(self, summaries: list[dict[str, Any]]) -> dict[str, Any]:
        nominal: dict[str, Any] = {}
        for bus in sorted({b for s in summaries for b in s}):
            parts = [(s[bus], s[bus]["seed"]) for s in summaries if bus in s]
            cols = sorted({c for p, _ in parts for c in p["cols"]})
            series, targets, seeds = [], [], []
            for p, seed in parts:
                for x, y in zip(p["x"], p["y"], strict=True):
                    wide = np.full((len(y), len(cols)), np.nan)
                    for j, c in enumerate(p["cols"]):
                        wide[:, cols.index(c)] = y[:, j]
                    series.append(x)
                    targets.append(wide)
                    seeds.append(seed)
            have = [k for k in range(len(cols)) if any(np.isfinite(t[:, k]).any()
                                                       for t in targets)]
            cols = [cols[k] for k in have]
            targets = [t[:, have] for t in targets]
            nominal[bus] = McdModel(self.cfg, cols).fit(series, targets, seeds)
        return nominal


class McdThermal(_Mcd):
    """F7: per-node temperature misses, scored by the worst node."""

    name = "mcd_thermal"
    fault_ids = ("F7",)

    def series(self, d: SeedData, sat: str) -> tuple[F64, F32, dict[str, F64], F64]:
        step = self.cfg.mcd_stride
        t, x, _ = context(d, sat, self.cfg)
        m = d.measurements(sat)
        y = {c: m[c][::step] for c in node_columns(m)}
        return t[::step], x[::step].astype(np.float32), y, np.zeros(len(t[::step]))

    def traces(self, d: SeedData, nominal: dict[str, Any]) -> list[Trace]:
        buses = self._buses()
        out = []
        for sat in d.sat_ids:
            model: McdModel = nominal[buses[sat]]
            t, x, y, _ = self.series(d, sat)
            mean, std = model.predict(x, d.seed)
            scores = []
            for k, c in enumerate(model.columns):
                r = (y.get(c, np.full(len(t), np.nan)) - mean[:, k]) / std[:, k]
                scores.append(gaussian_glr(np.nan_to_num(r), 0.0, 1.0, self.cfg.mcd_window))
            stack = np.stack(scores)
            worst = np.array([c.removeprefix("T_").removesuffix("_K_meas")
                              for c in model.columns], dtype="U32")[stack.argmax(axis=0)]
            out.append(Trace(sat, "F7", t, stack.max(axis=0), worst))
        return out


class McdEps(_Mcd):
    """F1: the pack-voltage miss is tested for a slope on the measured current."""

    name = "mcd_eps"
    fault_ids = ("F1",)

    def series(self, d: SeedData, sat: str) -> tuple[F64, F32, dict[str, F64], F64]:
        step = self.cfg.mcd_stride
        t, x, _ = context(d, sat, self.cfg)
        m = d.measurements(sat)
        inputs = np.concatenate([_voltage_inputs(m), x], axis=1)[::step].astype(np.float32)
        return (t[::step], np.nan_to_num(inputs), {"pack_v_V_meas": m["pack_v_V_meas"][::step]},
                m["pack_i_A_meas"][::step])

    def traces(self, d: SeedData, nominal: dict[str, Any]) -> list[Trace]:
        buses = self._buses()
        out = []
        for sat in d.sat_ids:
            model: McdModel = nominal[buses[sat]]
            t, x, y, current = self.series(d, sat)
            mean, std = model.predict(x, d.seed)
            z = (y["pack_v_V_meas"] - mean[:, 0]) / std[:, 0]
            out.append(Trace(sat, "F1", t, slope_glr(np.nan_to_num(z), np.nan_to_num(current),
                                                      1.0, self.cfg.mcd_window)))
        return out


MCD: dict[str, type[Detector]] = {"mcd_thermal": McdThermal, "mcd_eps": McdEps}
