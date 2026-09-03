"""
Automated unit tests for Static Split Conformal Prediction (P1) and CQR (P2).
"""
import numpy as np
import pytest

from shift_ami.conformal.static import StaticSplitConformal, compute_conformal_quantile
from shift_ami.conformal.cqr import ConformalizedQuantileRegression


def test_conformal_quantile_formula():
    # Calibration scores: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] (n=10)
    # alpha = 0.20 (80% coverage) -> ceil((10+1)*0.80)/10 = ceil(8.8)/10 = 9/10 = 0.90 quantile -> 9.1 or 9th element
    scores = np.arange(1.0, 11.0)
    q = compute_conformal_quantile(scores, alpha=0.20)
    assert q >= 8.5


def test_static_split_conformal_calibration():
    y_true_cal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred_cal = np.array([11.0, 19.0, 32.0, 38.0, 53.0])  # residuals: [1, 1, 2, 2, 3]

    model = StaticSplitConformal(alpha=0.20)
    model.calibrate(y_true_cal, y_pred_cal)

    assert model.is_calibrated
    assert model.conformal_quantile >= 2.0

    y_test_pred = np.array([25.0])
    lower, upper = model.predict_interval(y_test_pred)
    assert lower[0] <= 25.0
    assert upper[0] >= 25.0
    assert np.isclose(upper[0] - 25.0, model.conformal_quantile)


def test_cqr_calibration():
    y_true_cal = np.array([10.0, 20.0, 30.0])
    q_lo_cal = np.array([8.0, 19.0, 32.0])   # item 3 undercovers by 2
    q_hi_cal = np.array([12.0, 22.0, 35.0])

    cqr = ConformalizedQuantileRegression(alpha=0.10)
    cqr.calibrate(y_true_cal, q_lo_cal, q_hi_cal)

    assert cqr.is_calibrated
    lower, upper = cqr.predict_interval(np.array([10.0]), np.array([20.0]))
    assert lower[0] <= 10.0
    assert upper[0] >= 20.0
