from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from m4_forecasting.baselines import naive2_forecast
from m4_forecasting.data import load_config, load_m4
from m4_forecasting.global_model import build_training_set
from m4_forecasting.metrics import mase, smape

# The M4 CSVs are not committed - they belong to the M4 team and are 10MB+.
# Tests that need them skip rather than fail, so `pytest` is green on a fresh
# clone and the one test that genuinely needs data says so.
M4_DATA = Path(__file__).resolve().parents[1] / "data" / "Quarterly-train.csv"
needs_m4_data = pytest.mark.skipif(
    not M4_DATA.exists(),
    reason="M4 Quarterly data not present; see README > Reproduce to download it",
)


def test_smape_and_mase_basics():
    y = np.array([100.0, 100.0]); f = np.array([100.0, 100.0])
    assert smape(y, f) == 0.0
    ins = np.array([1, 2, 3, 4, 2, 3, 4, 5], dtype=float)
    assert mase(np.array([6.0]), np.array([6.0]), ins, m=4) == 0.0


def test_naive2_seasonal_series_reseasonalises():
    y = np.tile([100, 200, 300, 400], 10).astype(float) * np.linspace(1, 1.5, 40)
    f = naive2_forecast(y, h=4, m=4)
    # forecast should carry the seasonal shape (not a flat line)
    assert f.std() > 10


@needs_m4_data
def test_load_split_alignment():
    cfg = load_config(); cfg["data"]["n_series"] = 50
    train, test = load_m4(cfg)
    last = train.groupby("unique_id")["ds"].max()
    first_test = test.groupby("unique_id")["ds"].min()
    assert ((first_test - last) == 1).all()


def test_windows_shape():
    df = pd.DataFrame({"unique_id": "a", "ds": range(40), "y": np.arange(40, dtype=float)})
    X, Y = build_training_set(df, n_lags=20, h=8)
    assert X.shape == (13, 20) and Y.shape == (13, 8)
