"""Regressors for the learned residual generators: ridge in numpy and gradient-boosted trees."""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from expand3.detect.config import DetectConfig

F64 = NDArray[np.float64]


class Regressor(Protocol):
    def fit(self, x: F64, y: F64) -> Regressor: ...

    def predict(self, x: F64) -> F64: ...


def poly_terms(x: F64, degree: int) -> F64:
    """All monomials of the columns of x up to `degree`, without the constant."""
    cols = [np.prod(x[:, list(c)], axis=1)
            for deg in range(1, degree + 1)
            for c in combinations_with_replacement(range(x.shape[1]), deg)]
    return np.stack(cols, axis=1)


class Ridge:
    """Ridge regression on standardized (optionally polynomial) features, closed form."""

    def __init__(self, alpha: float, degree: int = 1) -> None:
        self.alpha, self.degree = alpha, degree
        self.mu: F64 = np.zeros(0)
        self.sd: F64 = np.ones(0)
        self.w: F64 = np.zeros(0)
        self.b = 0.0

    def _expand(self, x: F64) -> F64:
        return x if self.degree == 1 else poly_terms(x, self.degree)

    def fit(self, x: F64, y: F64) -> Ridge:
        z = self._expand(np.asarray(x, dtype=np.float64))
        self.mu = z.mean(axis=0)
        self.sd = np.where(z.std(axis=0) > 0, z.std(axis=0), 1.0)
        zs = (z - self.mu) / self.sd
        self.b = float(y.mean())
        gram = zs.T @ zs + self.alpha * np.eye(zs.shape[1])
        self.w = np.linalg.solve(gram, zs.T @ (y - self.b))
        return self

    def predict(self, x: F64) -> F64:
        z = (self._expand(np.asarray(x, dtype=np.float64)) - self.mu) / self.sd
        out: F64 = z @ self.w + self.b
        return out


def make_regressor(kind: str, cfg: DetectConfig, degree: int = 1) -> Regressor:
    """`ridge` or `hgb` (scikit-learn HistGradientBoostingRegressor, extra `detect-ml`)."""
    if kind == "ridge":
        return Ridge(cfg.learn_alpha, degree)
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        model: Any = HistGradientBoostingRegressor(
            max_iter=cfg.hgb_max_iter, learning_rate=cfg.hgb_learning_rate,
            max_leaf_nodes=cfg.hgb_max_leaf_nodes, random_state=cfg.random_seed)
        return model  # type: ignore[no-any-return]
    raise ValueError(f"unknown regressor {kind!r}")


def heldout_sigma(kind: str, cfg: DetectConfig, x: F64, y: F64, groups: NDArray[np.int_],
                  degree: int = 1) -> float:
    """Residual spread on seeds the model did not see: fit on the even seeds and score the odd,
    and the reverse. An in-sample spread would flatter a flexible model."""
    res = []
    for parity in (0, 1):
        train = groups % 2 == parity
        if train.all() or not train.any():
            continue
        m = make_regressor(kind, cfg, degree).fit(x[train], y[train])
        res.append(y[~train] - m.predict(x[~train]))
    resid = np.concatenate(res) if res else np.zeros(1)
    return max(float(resid.std()), cfg.learn_sigma_floor)
