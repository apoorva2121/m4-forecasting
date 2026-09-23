"""Global LightGBM: one model per horizon step, trained across all series on
normalised lag windows (cross-learning).

Each series is scaled by the mean of its last `n_lags` observations so the model
learns shape, not level. Predictions are rescaled back per series.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


def _windows(y: np.ndarray, n_lags: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """All (lags -> next h) windows from one series, each scaled by its lag mean."""
    X, Y = [], []
    for end in range(n_lags, len(y) - h + 1):
        lags = y[end - n_lags:end]
        scale = np.mean(np.abs(lags)) or 1.0
        X.append(lags / scale)
        Y.append(y[end:end + h] / scale)
    return np.asarray(X), np.asarray(Y)


def build_training_set(train: pd.DataFrame, n_lags: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    Xs, Ys = [], []
    for _, g in train.groupby("unique_id", sort=False):
        y = g["y"].to_numpy(dtype=float)
        if len(y) >= n_lags + h:
            X, Y = _windows(y, n_lags, h)
            Xs.append(X); Ys.append(Y)
    return np.vstack(Xs), np.vstack(Ys)


def _features(X: np.ndarray) -> np.ndarray:
    """Lags plus a few shape descriptors (slope, last-vs-mean, seasonal diffs)."""
    slope = (X[:, -1] - X[:, 0]) / (X.shape[1] - 1)
    last_minus_mean = X[:, -1] - X.mean(axis=1)
    sd = X.std(axis=1)
    s4 = X[:, -1] - X[:, -5] if X.shape[1] >= 5 else np.zeros(len(X))
    return np.column_stack([X, slope, last_minus_mean, sd, s4])


class GlobalLGBM:
    def __init__(self, cfg: dict[str, Any]) -> None:
        g = cfg["global_model"]
        self.n_lags, self.h = g["n_lags"], cfg["data"]["horizon"]
        self.params = g["params"]
        self.models: list[LGBMRegressor] = []

    def fit(self, train: pd.DataFrame, seed: int = 42) -> "GlobalLGBM":
        X, Y = build_training_set(train, self.n_lags, self.h)
        F = _features(X)
        self.models = []
        for k in range(self.h):
            m = LGBMRegressor(random_state=seed + k, **self.params)
            m.fit(F, Y[:, k])
            self.models.append(m)
        return self

    def forecast(self, train: pd.DataFrame, name: str = "LightGBM") -> pd.DataFrame:
        rows = []
        for uid, g in train.groupby("unique_id", sort=False):
            y = g["y"].to_numpy(dtype=float)
            last = int(g["ds"].max())
            if len(y) < self.n_lags:
                pad = np.repeat(y[0], self.n_lags - len(y))
                y = np.concatenate([pad, y])
            lags = y[-self.n_lags:]
            scale = np.mean(np.abs(lags)) or 1.0
            F = _features((lags / scale)[None, :])
            f = np.array([m.predict(F)[0] for m in self.models]) * scale
            rows.extend({"unique_id": uid, "ds": last + 1 + k, name: f[k]} for k in range(self.h))
        return pd.DataFrame(rows)
