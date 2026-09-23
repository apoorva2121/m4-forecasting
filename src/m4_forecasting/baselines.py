"""Statistical baselines via statsforecast, plus the M4 Naive2 reference method.

Naive2: if a 90%-significance seasonality test on the autocorrelation at lag m
passes, classically decompose (multiplicative), forecast the seasonally-adjusted
series with Naive, and re-apply the seasonal indices. Otherwise plain Naive.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, Naive, SeasonalNaive, Theta


def _seasonality_test(y: np.ndarray, m: int) -> bool:
    n = len(y)
    if n < 3 * m:
        return False
    y = y - y.mean()
    denom = np.sum(y * y)
    if denom == 0:
        return False
    acf = np.array([np.sum(y[k:] * y[:-k]) / denom for k in range(1, m + 1)])
    limit = 1.645 * np.sqrt((1 + 2 * np.sum(acf[:-1] ** 2)) / n)
    return abs(acf[-1]) > limit


def _seasonal_indices(y: np.ndarray, m: int) -> np.ndarray:
    """Classical multiplicative decomposition seasonal indices (length m)."""
    n = len(y)
    # centred moving average of order m (even m -> 2xm MA)
    if m % 2 == 0:
        ma = pd.Series(y).rolling(m, center=True).mean().rolling(2, center=True).mean().to_numpy()
    else:
        ma = pd.Series(y).rolling(m, center=True).mean().to_numpy()
    ratio = y / ma
    idx = np.array([np.nanmean(ratio[i::m]) for i in range(m)])
    idx = idx / np.nanmean(idx)
    return idx


def naive2_forecast(y: np.ndarray, h: int, m: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if _seasonality_test(y, m):
        si = _seasonal_indices(y, m)
        si_full = np.resize(si, n)  # aligned to positions 0..n-1
        adj = y / si_full
        level = adj[-1]
        future_si = np.array([si[(n + k) % m] for k in range(h)])
        return level * future_si
    return np.repeat(y[-1], h)


def fit_baselines(train: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Return forecasts frame: unique_id, ds, one column per baseline."""
    h, m = cfg["data"]["horizon"], cfg["data"]["season_length"]
    wanted = cfg["baselines"]["models"]
    sf_models = []
    if "Naive" in wanted:
        sf_models.append(Naive())
    if "SeasonalNaive" in wanted:
        sf_models.append(SeasonalNaive(season_length=m))
    if "Theta" in wanted:
        sf_models.append(Theta(season_length=m))
    if "AutoETS" in wanted:
        sf_models.append(AutoETS(season_length=m))

    out = None
    if sf_models:
        sf = StatsForecast(models=sf_models, freq=1, n_jobs=cfg["baselines"]["n_jobs"])
        out = sf.forecast(df=train[["unique_id", "ds", "y"]], h=h).reset_index(drop=True)
        if "index" in out.columns:
            out = out.drop(columns=["index"])

    if "Naive2" in wanted:
        rows = []
        for uid, g in train.groupby("unique_id", sort=False):
            y = g["y"].to_numpy()
            last = int(g["ds"].max())
            f = naive2_forecast(y, h, m)
            rows.extend({"unique_id": uid, "ds": last + 1 + k, "Naive2": f[k]} for k in range(h))
        n2 = pd.DataFrame(rows)
        out = n2 if out is None else out.merge(n2, on=["unique_id", "ds"], how="left")
    return out
