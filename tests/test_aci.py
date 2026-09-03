"""
Automated unit tests for Adaptive Conformal Inference (P4).
"""
import numpy as np
import pytest

from shift_ami.conformal.aci import AdaptiveConformalInference


def test_aci_adaptation_direction():
    # Calibration scores
    cal_scores = np.linspace(0.5, 5.0, 100)
    aci = AdaptiveConformalInference(alpha=0.10, gamma=0.02, alpha_min=0.01, alpha_max=0.50)
    aci.calibrate(cal_scores)

    initial_alpha = aci.alpha_t  # 0.10

    # 1. Simulate miscoverage (y outside [lower, upper]) -> err_t = 1.0
    # alpha_{t+1} = alpha_t + gamma * (0.10 - 1.0) = alpha_t - 0.90 * gamma < alpha_t
    aci.step(y_true=100.0, lower=10.0, upper=20.0)
    assert aci.alpha_t < initial_alpha

    # 2. Simulate correct coverage -> err_t = 0.0
    # alpha_{t+1} = alpha_t + gamma * (0.10 - 0.0) = alpha_t + 0.10 * gamma > alpha_t
    alpha_after_miss = aci.alpha_t
    aci.step(y_true=15.0, lower=10.0, upper=20.0)
    assert aci.alpha_t > alpha_after_miss


def test_aci_clipping_bounds():
    cal_scores = np.linspace(0.5, 5.0, 100)
    aci = AdaptiveConformalInference(alpha=0.10, gamma=0.10, alpha_min=0.02, alpha_max=0.40)
    aci.calibrate(cal_scores)

    # Trigger repeated failures to force alpha downward to alpha_min
    for _ in range(50):
        aci.step(y_true=100.0, lower=10.0, upper=20.0)
    assert aci.alpha_t == 0.02  # Clipped at alpha_min

    # Trigger repeated successes to force alpha upward to alpha_max
    for _ in range(100):
        aci.step(y_true=15.0, lower=10.0, upper=20.0)
    assert aci.alpha_t == 0.40  # Clipped at alpha_max
