"""Learned residual generators: predict a measured signal from context, run the GLR on the miss.

Training data is nominal only: unfaulted vehicles plus each faulted vehicle before arm, from the
tune seeds, with separate models per bus. A model is ridge on filtered context or gradient-boosted
trees (`kind`). The residual spread comes from seeds held out of the fit.

- Thermal (F7): every node's temperature from exogenous context only (see features.py).
- EPS (F1): pack voltage from measured current, SoC estimate and battery temperature. A
  resistance change adds a voltage miss proportional to the current, so the GLR tests the slope of
  the residual on the current. (F2): each string's power while sunlit from context, GLR on a drop.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from expand3.detect.base import Detector, Trace
from expand3.detect.data import SeedData
from expand3.detect.features import bus_of, context, node_columns, sunlit
from expand3.detect.glr import gaussian_glr, slope_glr
from expand3.detect.models import heldout_sigma, make_regressor

F64 = NDArray[np.float64]
N_STRINGS = 4


def _rows(n: int, stride: int) -> NDArray[np.bool_]:
    on = np.zeros(n, dtype=bool)
    on[::stride] = True
    return on


def _pool(summaries: list[dict[str, Any]], max_rows: int, seed: int
          ) -> dict[str, dict[str, Any]]:
    """Concatenate per-bus training arrays across seeds, thinned to at most max_rows."""
    out: dict[str, dict[str, Any]] = {}
    for s in summaries:
        for bus, part in s.items():
            acc = out.setdefault(bus, {"X": [], "g": [], "Y": {}})
            acc["X"].append(part["X"])
            acc["g"].append(part["g"])
            for k, v in part["Y"].items():
                acc["Y"].setdefault(k, []).append(v)
    rng = np.random.default_rng(seed)
    for acc in out.values():
        acc["X"] = np.concatenate(acc["X"])
        acc["g"] = np.concatenate(acc["g"])
        acc["Y"] = {k: np.concatenate(v) for k, v in acc["Y"].items()}
        if len(acc["g"]) > max_rows:
            keep = np.sort(rng.choice(len(acc["g"]), max_rows, replace=False))
            acc["X"], acc["g"] = acc["X"][keep], acc["g"][keep]
            acc["Y"] = {k: v[keep] for k, v in acc["Y"].items()}
    return out


class _Learned(Detector):
    family = "learned"
    kind = "ridge"

    def _buses(self) -> dict[str, str]:
        assert self.params_dir is not None
        return bus_of(self.params_dir)


class ThermalResidual(_Learned):
    """F7: per-node temperature residuals, scored by the worst node."""

    fault_ids = ("F7",)

    def summarize(self, d: SeedData) -> dict[str, Any]:
        buses = self._buses()
        out: dict[str, dict[str, Any]] = {}
        for sat in d.sat_ids:
            _, x, _ = context(d, sat, self.cfg)
            m = d.measurements(sat)
            use = d.fault_free(sat, m["t_s"]) & _rows(len(x), self.cfg.learn_stride)
            part = out.setdefault(buses[sat], {"X": [], "g": [], "Y": {}})
            part["X"].append(x[use])
            part["g"].append(np.full(int(use.sum()), d.seed))
            for c in node_columns(m):
                part["Y"].setdefault(c, []).append(m[c][use])
        return {bus: {"X": np.concatenate(p["X"]), "g": np.concatenate(p["g"]),
                      "Y": {k: np.concatenate(v) for k, v in p["Y"].items()}}
                for bus, p in out.items()}

    def fit(self, summaries: list[dict[str, Any]]) -> dict[str, Any]:
        pooled = _pool(summaries, self.cfg.learn_max_rows, self.cfg.random_seed)
        nominal: dict[str, Any] = {}
        for bus, acc in pooled.items():
            nominal[bus] = {}
            for col, y in acc["Y"].items():
                ok = np.isfinite(y)
                sigma = heldout_sigma(self.kind, self.cfg, acc["X"][ok], y[ok], acc["g"][ok])
                model = make_regressor(self.kind, self.cfg).fit(acc["X"][ok], y[ok])
                nominal[bus][col] = (model, sigma)
        return nominal

    def traces(self, d: SeedData, nominal: dict[str, Any]) -> list[Trace]:
        buses = self._buses()
        out = []
        for sat in d.sat_ids:
            t, x, _ = context(d, sat, self.cfg)
            m = d.measurements(sat)
            names = list(nominal[buses[sat]])
            scores = []
            for col in names:
                model, sigma = nominal[buses[sat]][col]
                r = (m[col] - model.predict(x)) / sigma
                scores.append(gaussian_glr(np.nan_to_num(r), 0.0, 1.0, self.cfg.learn_window))
            stack = np.stack(scores)
            worst = np.array([n.removeprefix("T_").removesuffix("_K_meas") for n in names],
                             dtype="U32")[stack.argmax(axis=0)]
            out.append(Trace(sat, "F7", t, stack.max(axis=0), worst))
        return out


def _voltage_inputs(m: dict[str, F64]) -> F64:
    return np.stack([m["pack_i_A_meas"], m["soc_est_meas"], m["T_battery_K_meas"]], axis=1)


class EpsResidual(_Learned):
    """F1 from the pack voltage residual and F2 from each string's power residual."""

    fault_ids = ("F1", "F2")

    def summarize(self, d: SeedData) -> dict[str, Any]:
        buses = self._buses()
        out: dict[str, dict[str, Any]] = {}
        for sat in d.sat_ids:
            _, x, _ = context(d, sat, self.cfg)
            m = d.measurements(sat)
            use = d.fault_free(sat, m["t_s"]) & _rows(len(x), self.cfg.learn_stride)
            sun = sunlit(m, self.cfg) > 0
            part = out.setdefault(buses[sat], {"v": {"X": [], "y": [], "g": []},
                                               "s": {"X": [], "g": [], "Y": {}}})
            part["v"]["X"].append(_voltage_inputs(m)[use])
            part["v"]["y"].append(m["pack_v_V_meas"][use])
            part["v"]["g"].append(np.full(int(use.sum()), d.seed))
            use_s = use & sun
            part["s"]["X"].append(x[use_s])
            part["s"]["g"].append(np.full(int(use_s.sum()), d.seed))
            for i in range(N_STRINGS):
                c = f"string_p_{i + 1}_W_meas"
                if not np.all(np.isnan(m[c])):
                    part["s"]["Y"].setdefault(c, []).append(m[c][use_s])
        return out

    def fit(self, summaries: list[dict[str, Any]]) -> dict[str, Any]:
        cfg = self.cfg
        buses = {b for s in summaries for b in s}
        nominal: dict[str, Any] = {}
        for bus in buses:
            parts = [s[bus] for s in summaries if bus in s]
            xv = np.concatenate([np.concatenate(p["v"]["X"]) for p in parts])
            yv = np.concatenate([np.concatenate(p["v"]["y"]) for p in parts])
            gv = np.concatenate([np.concatenate(p["v"]["g"]) for p in parts])
            if len(yv) > cfg.learn_max_rows:
                keep = np.sort(np.random.default_rng(cfg.random_seed).choice(
                    len(yv), cfg.learn_max_rows, replace=False))
                xv, yv, gv = xv[keep], yv[keep], gv[keep]
            degree = cfg.learn_poly_degree if self.kind == "ridge" else 1
            sigma_v = heldout_sigma(self.kind, cfg, xv, yv, gv, degree)
            entry: dict[str, Any] = {"v": (make_regressor(self.kind, cfg, degree).fit(xv, yv),
                                           sigma_v, degree), "s": {}}
            sums = [{bus: {"X": np.concatenate(p["s"]["X"]), "g": np.concatenate(p["s"]["g"]),
                           "Y": {k: np.concatenate(v) for k, v in p["s"]["Y"].items()}}}
                    for p in parts]
            pooled = _pool(sums, cfg.learn_max_rows, cfg.random_seed)[bus]
            for col, y in pooled["Y"].items():
                sigma = heldout_sigma(self.kind, cfg, pooled["X"], y, pooled["g"])
                entry["s"][col] = (make_regressor(self.kind, cfg).fit(pooled["X"], y), sigma)
            nominal[bus] = entry
        return nominal

    def traces(self, d: SeedData, nominal: dict[str, Any]) -> list[Trace]:
        buses = self._buses()
        out = []
        for sat in d.sat_ids:
            nom = nominal[buses[sat]]
            t, x, _ = context(d, sat, self.cfg)
            m = d.measurements(sat)
            model, sigma, _ = nom["v"]
            resid = m["pack_v_V_meas"] - model.predict(_voltage_inputs(m))
            out.append(Trace(sat, "F1", t, slope_glr(np.nan_to_num(resid), m["pack_i_A_meas"],
                                                      sigma, self.cfg.learn_window)))
            sun = sunlit(m, self.cfg) > 0
            for col, (smodel, ssigma) in nom["s"].items():
                r = (m[col][sun] - smodel.predict(x[sun])) / ssigma
                score = gaussian_glr(np.nan_to_num(r), 0.0, 1.0, self.cfg.f2_window, side="down")
                k = col.split("_")[2]
                out.append(Trace(sat, "F2", t[sun], score,
                                 np.full(int(sun.sum()), f"string_{k}", dtype="U16")))
        return out


def _variant(base: type[_Learned], kind: str, name: str) -> type[_Learned]:
    return type(f"{base.__name__}_{kind}", (base,), {"kind": kind, "name": name})


LEARNED: dict[str, type[Detector]] = {
    "lrn_thermal_ridge": _variant(ThermalResidual, "ridge", "lrn_thermal_ridge"),
    "lrn_thermal_hgb": _variant(ThermalResidual, "hgb", "lrn_thermal_hgb"),
    "lrn_eps_ridge": _variant(EpsResidual, "ridge", "lrn_eps_ridge"),
    "lrn_eps_hgb": _variant(EpsResidual, "hgb", "lrn_eps_hgb"),
}
