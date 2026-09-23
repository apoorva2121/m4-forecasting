"""Load the M4 wide CSVs into long format and split train / official test."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _wide_to_long(path: str | Path) -> pd.DataFrame:
    """M4 CSV: first column is the series id, remaining columns are values (NaN-padded)."""
    wide = pd.read_csv(path)
    wide = wide.rename(columns={wide.columns[0]: "unique_id"})
    long = wide.melt(id_vars="unique_id", var_name="step", value_name="y").dropna(subset=["y"])
    long["step"] = long["step"].str.lstrip("V").astype(int) - 2  # V2 -> 0
    long = long.sort_values(["unique_id", "step"]).reset_index(drop=True)
    return long


def load_m4(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_long, test_long). Test holds exactly `horizon` points per series.

    Both frames carry an integer `ds` index continuous across the split, which is
    all statsforecast needs for non-calendar M4 series.
    """
    train = _wide_to_long(cfg["paths"]["train"])
    test = _wide_to_long(cfg["paths"]["test"])

    n = cfg["data"]["n_series"]
    if n:
        rng = np.random.default_rng(cfg["data"]["seed"])
        keep = rng.choice(train["unique_id"].unique(), size=n, replace=False)
        train = train[train["unique_id"].isin(keep)]
        test = test[test["unique_id"].isin(keep)]

    train = train.rename(columns={"step": "ds"})
    lengths = train.groupby("unique_id")["ds"].max() + 1
    test = test.merge(lengths.rename("offset"), left_on="unique_id", right_index=True)
    test["ds"] = test["step"] + test["offset"]
    test = test.drop(columns=["step", "offset"])
    h = cfg["data"]["horizon"]
    assert (test.groupby("unique_id").size() == h).all(), "every test series must have exactly h points"
    return train.reset_index(drop=True), test.reset_index(drop=True)


def to_wide(long: pd.DataFrame, value: str = "y") -> pd.DataFrame:
    return long.pivot(index="unique_id", columns="ds", values=value)
