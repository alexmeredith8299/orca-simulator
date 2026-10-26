"""Unsupervised baselines: PCA (Hotelling T^2 and SPE) and an isolation forest, per subsystem.

Measured signals are cut into non-overlapping 10-minute windows (mean, spread or sum per
signal), standardized with nominal means and spreads, and scored per subsystem and bus. Models
are fitted on nominal windows only. The scores go through the same threshold tuning as the
others, so every method meets the same false-alarm target. They say that something is off in a
subsystem, not which fault: alarms are `detection_only`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from expand3.detect.base import Detector, Trace
from expand3.detect.data import SeedData
from expand3.detect.features import bus_of, node_columns, sunlit

F64 = NDArray[np.float64]
N_STRINGS = 4
REDUCERS: dict[str, Callable[..., Any]] = {"mean": np.nanmean, "std": np.nanstd,
                                           "sum": np.nansum}
# subsystem -> (signal, statistic) with statistic in {"mean", "std", "sum"}
SUBSYSTEMS = ("thermal", "eps", "cdh", "attitude")


def signals(sub: str, m: dict[str, F64]) -> list[tuple[str, str]]:
    if sub == "thermal":
        return [(c, "mean") for c in node_columns(m)] + [("sunlit", "mean"),
                                                         ("gate_pass", "mean"),
                                                         ("gate_burn", "mean")]
    if sub == "eps":
        strings = [f"string_p_{i + 1}_W_meas" for i in range(N_STRINGS)]
        strings = [c for c in strings if not np.all(np.isnan(m[c]))]
        return ([(c, "mean") for c in ("pack_v_V_meas", "pack_i_A_meas", "soc_est_meas",
                                       "T_battery_K_meas", *strings)]
                + [("pack_v_V_meas", "std"), ("pack_i_A_meas", "std"), ("sunlit", "mean")])
    if sub == "cdh":
        return [("upsets", "sum"), ("resets", "sum"), ("gate_saa", "mean"), ("field_nT", "mean")]
    return ([(f"imu_omega_{a}_rad_s_meas", s) for a in "xyz" for s in ("mean", "std")]
            + [("gate_slew", "mean"), ("gate_burn", "mean"), ("st_available", "mean")])


def windows(d: SeedData, sat: str, sub: str, w: int, cfg: Any
            ) -> tuple[F64, F64, list[tuple[str, str]]]:
    """(window end times, features (n_windows, n_features), the feature names)."""
    m = dict(d.measurements(sat))
    m["sunlit"] = sunlit(m, cfg)
    spec = signals(sub, m)
    n = len(m["t_s"]) // w
    cols = []
    for name, stat in spec:
        x = m[name][: n * w].reshape(n, w)
        with np.errstate(all="ignore"):
            cols.append(np.asarray(REDUCERS[stat](x, axis=1), dtype=np.float64))
    return m["t_s"][: n * w].reshape(n, w)[:, -1], np.stack(cols, axis=1), spec


class _Unsup(Detector):
    family = "unsupervised"
    detection_only = True
    h_floor = 0.0  # scores are not GLR statistics; the false-alarm target alone sets the level

    def _buses(self) -> dict[str, str]:
        assert self.params_dir is not None
        return bus_of(self.params_dir)

    def summarize(self, d: SeedData) -> dict[str, Any]:
        buses = self._buses()
        w = self.cfg.unsup_window_ticks
        out: dict[str, Any] = {}
        for sat in d.sat_ids:
            for sub in SUBSYSTEMS:
                t_end, x, _ = windows(d, sat, sub, w, self.cfg)
                ok = d.fault_free(sat, t_end - w * 30.0) & d.fault_free(sat, t_end)
                out.setdefault(f"{buses[sat]}|{sub}", []).append(x[ok])
        return {k: np.concatenate(v) for k, v in out.items()}

    def fit(self, summaries: list[dict[str, Any]]) -> dict[str, Any]:
        nominal = {}
        for key in {k for s in summaries for k in s}:
            x = np.concatenate([s[key] for s in summaries if key in s])
            mu = np.nanmean(x, axis=0)
            sd = np.where(np.nanstd(x, axis=0) > 0, np.nanstd(x, axis=0), 1.0)
            nominal[key] = (mu, sd, self._fit_model(np.nan_to_num((x - mu) / sd)))
        return nominal

    def _fit_model(self, z: F64) -> Any:
        raise NotImplementedError

    def _score(self, model: Any, z: F64) -> dict[str, F64]:
        raise NotImplementedError

    def traces(self, d: SeedData, nominal: dict[str, Any]) -> list[Trace]:
        buses = self._buses()
        out = []
        for sat in d.sat_ids:
            for sub in SUBSYSTEMS:
                t_end, x, _ = windows(d, sat, sub, self.cfg.unsup_window_ticks, self.cfg)
                mu, sd, model = nominal[f"{buses[sat]}|{sub}"]
                for stat, score in self._score(model, np.nan_to_num((x - mu) / sd)).items():
                    out.append(Trace(sat, f"{stat}:{sub}", t_end, score))
        return out


class PcaDetector(_Unsup):
    """Hotelling T^2 on the retained components and the squared prediction error of the rest."""

    name = "unsup_pca"
    fault_ids = tuple(f"{k}:{s}" for s in SUBSYSTEMS for k in ("T2", "SPE"))

    def _fit_model(self, z: F64) -> Any:
        from sklearn.decomposition import PCA
        return PCA(n_components=self.cfg.unsup_pca_variance, svd_solver="full").fit(z)

    def _score(self, model: Any, z: F64) -> dict[str, F64]:
        scores = model.transform(z)
        t2 = np.sum(scores**2 / model.explained_variance_, axis=1)
        spe = np.sum((z - model.inverse_transform(scores)) ** 2, axis=1)
        return {"T2": np.asarray(t2, dtype=np.float64), "SPE": np.asarray(spe, dtype=np.float64)}


class IsolationForestDetector(_Unsup):
    name = "unsup_iforest"
    fault_ids = tuple(f"IF:{s}" for s in SUBSYSTEMS)

    def _fit_model(self, z: F64) -> Any:
        from sklearn.ensemble import IsolationForest
        return IsolationForest(n_estimators=self.cfg.unsup_trees,
                               random_state=self.cfg.random_seed).fit(z)

    def _score(self, model: Any, z: F64) -> dict[str, F64]:
        return {"IF": -np.asarray(model.score_samples(z), dtype=np.float64)}


UNSUP: dict[str, type[Detector]] = {
    "unsup_pca": PcaDetector,
    "unsup_iforest": IsolationForestDetector,
}
