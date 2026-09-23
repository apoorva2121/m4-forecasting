"""Official M4 metrics: sMAPE, MASE, and OWA relative to Naive2.

Definitions follow the M4 Competitor's Guide. MASE scales by the in-sample
seasonal-naive error (lag = season_length) of the *training* series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def smape(y: np.ndarray, f: np.ndarray) -> float:
    denom = np.abs(y) + np.abs(f)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denom == 0, 0.0, 2.0 * np.abs(y - f) / denom)
    return float(100.0 * np.mean(r))


def mase(y: np.ndarray, f: np.ndarray, insample: np.ndarray, m: int) -> float:
    scale = np.mean(np.abs(insample[m:] - insample[:-m])) if len(insample) > m else np.mean(np.abs(np.diff(insample)))
    if scale == 0 or np.isnan(scale):
        return np.nan
    return float(np.mean(np.abs(y - f)) / scale)


def evaluate(forecasts: pd.DataFrame, test: pd.DataFrame, train: pd.DataFrame,
             model_cols: list[str], m: int) -> pd.DataFrame:
    """Per-model sMAPE / MASE averaged over series, plus OWA vs Naive2.

    `forecasts` has columns unique_id, ds, <model...>; `test` has unique_id, ds, y.
    """
    merged = test.merge(forecasts, on=["unique_id", "ds"], how="left")
    insample = {uid: g["y"].to_numpy() for uid, g in train.groupby("unique_id")}
    rows = {}
    for col in model_cols:
        s_list, m_list = [], []
        for uid, g in merged.groupby("unique_id"):
            y, f = g["y"].to_numpy(), g[col].to_numpy()
            s_list.append(smape(y, f))
            m_list.append(mase(y, f, insample[uid], m))
        rows[col] = {"sMAPE": float(np.mean(s_list)), "MASE": float(np.nanmean(m_list))}
    out = pd.DataFrame(rows).T
    if "Naive2" in out.index:
        n2 = out.loc["Naive2"]
        out["OWA"] = 0.5 * (out["sMAPE"] / n2["sMAPE"] + out["MASE"] / n2["MASE"])
    return out.sort_values("OWA" if "OWA" in out else "sMAPE")
