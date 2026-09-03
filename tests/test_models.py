"""
Automated unit tests for baseline forecasting models (B0, B1, B2, B3).
"""
import numpy as np
import pandas as pd
import pytest

from shift_ami.models.seasonal_naive import SeasonalNaiveModel
from shift_ami.models.ridge import RidgeModel
from shift_ami.models.gradient_boosting import HistGBRModel
from shift_ami.models.quantile import QuantileGBRModel


def test_seasonal_naive_model():
    X = pd.DataFrame({"load_lag_48": [10.0, 20.0, 30.0], "load_lag_336": [15.0, 25.0, 35.0]})
    y = pd.Series([12.0, 22.0, 32.0])

    m48 = SeasonalNaiveModel(season_lag=48).fit(X, y)
    preds48 = m48.predict(X)
    np.testing.assert_array_equal(preds48, np.array([10.0, 20.0, 30.0]))


def test_ridge_and_histgbr_models():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(0, 1, (100, 5)), columns=[f"f_{i}" for i in range(5)])
    y = pd.Series(2.0 * X["f_0"] + rng.normal(0, 0.1, 100))

    ridge = RidgeModel(alpha=1.0).fit(X, y)
    preds_r = ridge.predict(X)
    assert len(preds_r) == 100

    hgbr = HistGBRModel(max_iter=20, random_state=42).fit(X, y)
    preds_h = hgbr.predict(X)
    assert len(preds_h) == 100


def test_quantile_gbr_model():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(0, 1, (100, 3)), columns=[f"f_{i}" for i in range(3)])
    y = pd.Series(5.0 + rng.normal(0, 1, 100))

    q_model = QuantileGBRModel(quantiles=[0.05, 0.50, 0.95], max_iter=20, random_state=42).fit(X, y)
    q05 = q_model.predict_quantile(X, 0.05)
    q50 = q_model.predict_quantile(X, 0.50)
    q95 = q_model.predict_quantile(X, 0.95)

    # Monotonicity check on average: q05 <= q50 <= q95
    assert np.mean(q05) <= np.mean(q50) <= np.mean(q95)
