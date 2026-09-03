"""
Automated unit tests for point and probabilistic evaluation metrics with hand-computable ground truths.
"""
import numpy as np
import pytest

from shift_ami.evaluation.point_metrics import calculate_point_metrics
from shift_ami.evaluation.probabilistic_metrics import (
    calculate_probabilistic_metrics,
    calculate_interval_score,
    calculate_pinball_loss
)


def test_point_metrics_hand_computed():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 30.0, 44.0])

    # Absolute errors: [2, 2, 0, 4] -> MAE = 8 / 4 = 2.0
    # Squared errors: [4, 4, 0, 16] -> Mean = 24 / 4 = 6.0 -> RMSE = sqrt(6.0) = 2.4495
    # Mean true = 25.0 -> NMAE = 2.0 / 25.0 = 0.08
    m = calculate_point_metrics(y_true, y_pred)

    assert np.isclose(m["mae"], 2.0, atol=1e-3)
    assert np.isclose(m["rmse"], np.sqrt(6.0), atol=1e-3)
    assert np.isclose(m["nmae"], 0.08, atol=1e-3)


def test_probabilistic_metrics_hand_computed():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    lower = np.array([8.0, 15.0, 25.0, 45.0])   # 4th item undercovered (y=40 < 45)
    upper = np.array([12.0, 25.0, 35.0, 50.0])

    # Covered: [1, 1, 1, 0] -> 3/4 = 0.75 coverage
    # Nominal alpha = 0.10 (Target = 0.90) -> ACE = |0.75 - 0.90| = 0.15
    # Widths: [4, 10, 10, 5] -> MPIW = 29 / 4 = 7.25
    m = calculate_probabilistic_metrics(y_true, lower, upper, nominal_alpha=0.10)

    assert np.isclose(m["empirical_coverage"], 0.75, atol=1e-3)
    assert np.isclose(m["target_coverage"], 0.90, atol=1e-3)
    assert np.isclose(m["ace"], 0.15, atol=1e-3)
    assert np.isclose(m["mpiw"], 7.25, atol=1e-3)


def test_winkler_interval_score():
    # IS = (upper - lower) + 2/alpha * (lower - y)*I(y < lower) + 2/alpha * (y - upper)*I(y > upper)
    # Item 1: y=10, [8, 12] -> covered -> IS = 4 + 0 = 4
    # Item 2: y=5,  [8, 12] -> below -> IS = 4 + (2/0.10)*(8 - 5) = 4 + 20*3 = 64
    # Item 3: y=15, [8, 12] -> above -> IS = 4 + (2/0.10)*(15 - 12) = 4 + 20*3 = 64
    y = np.array([10.0, 5.0, 15.0])
    l = np.array([8.0, 8.0, 8.0])
    u = np.array([12.0, 12.0, 12.0])

    score = calculate_interval_score(y, l, u, alpha=0.10)
    expected = (4.0 + 64.0 + 64.0) / 3.0
    assert np.isclose(score, expected, atol=1e-3)


def test_pinball_loss():
    # Quantile tau = 0.90
    # y = 10, q = 8 (under): loss = 0.90 * (10 - 8) = 1.8
    # y = 10, q = 12 (over): loss = (0.90 - 1) * (10 - 12) = -0.10 * -2 = 0.2
    assert np.isclose(calculate_pinball_loss(np.array([10.0]), np.array([8.0]), 0.90), 1.8, atol=1e-3)
    assert np.isclose(calculate_pinball_loss(np.array([10.0]), np.array([12.0]), 0.90), 0.2, atol=1e-3)
