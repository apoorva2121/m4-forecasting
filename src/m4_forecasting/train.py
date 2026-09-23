"""Fit baselines + global LightGBM on M4 Quarterly train, evaluate on the official test set.

Usage:
    python -m m4_forecasting.train [--config config.yaml] [--n-series 2000]

The M4 test set is public but is a true holdout: nothing here is fit on it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .baselines import fit_baselines
from .data import load_config, load_m4
from .global_model import GlobalLGBM
from .metrics import evaluate

# Published M4 Quarterly results (Makridakis, Spiliotis & Assimakopoulos, IJF 2020, Table 3).
# Verify against the paper before quoting; listed here as the reference points.
PUBLISHED = {
    "Naive2 (paper)": {"sMAPE": 11.012, "MASE": 1.371, "OWA": 1.000},
    "Theta (paper)": {"sMAPE": 10.311, "MASE": 1.231, "OWA": 0.917},
    "Comb S-H-D (paper benchmark)": {"sMAPE": 10.186, "MASE": 1.196, "OWA": 0.899},
    "ES-RNN, Smyl (winner)": {"sMAPE": 9.679, "MASE": 1.111, "OWA": 0.847},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n-series", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.n_series:
        cfg["data"]["n_series"] = args.n_series
    rep = Path(cfg["paths"]["reports"]); rep.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    train, test = load_m4(cfg)
    m = cfg["data"]["season_length"]
    print(f"series: {train['unique_id'].nunique():,}  train points: {len(train):,}  "
          f"test points: {len(test):,}  horizon: {cfg['data']['horizon']}")

    t = time.time()
    fc = fit_baselines(train, cfg)
    print(f"baselines fitted in {time.time() - t:.0f}s")

    t = time.time()
    lgb = GlobalLGBM(cfg).fit(train, seed=cfg["data"]["seed"])
    fc = fc.merge(lgb.forecast(train), on=["unique_id", "ds"], how="left")
    print(f"global LightGBM fitted in {time.time() - t:.0f}s")

    members = [c for c in cfg["ensemble"]["members"] if c in fc.columns]
    fc["Ensemble"] = fc[members].median(axis=1)

    model_cols = [c for c in fc.columns if c not in ("unique_id", "ds")]
    scores = evaluate(fc, test, train, model_cols, m)
    print("\nOfficial M4 Quarterly test set:\n", scores.round(4).to_string())
    print("\nPublished reference points:\n", pd.DataFrame(PUBLISHED).T.to_string())

    fc.to_parquet(rep / "forecasts_quarterly.parquet", index=False)
    report = {
        "n_series": int(train["unique_id"].nunique()),
        "horizon": cfg["data"]["horizon"],
        "scores_on_official_test": scores.round(5).to_dict(orient="index"),
        "published_reference": PUBLISHED,
        "ensemble_members": members,
        "runtime_seconds": round(time.time() - t0, 1),
        "config": cfg,
    }
    (rep / "results.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport -> {rep / 'results.json'}   runtime {report['runtime_seconds']}s")


if __name__ == "__main__":
    main()
