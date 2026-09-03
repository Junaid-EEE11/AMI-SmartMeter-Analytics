"""
Automated unit tests for Proposed Shift-Aware Adaptive Conformal Prediction (P5: SA-ACP).
"""
import numpy as np
import pytest

from shift_ami.conformal.sa_acp import ShiftAwareAdaptiveConformal


def test_sa_acp_responsiveness_modulation():
    rng = np.random.default_rng(42)
    cal_scores = rng.normal(10.0, 2.0, 500)

    sa = ShiftAwareAdaptiveConformal(
        alpha=0.10,
        gamma_slow=0.005,
        gamma_fast=0.035,
        threshold=1.25,
        window_length=50
    )
    sa.calibrate(cal_scores)

    # In stationary regime, gamma must remain gamma_slow
    state_stat = sa.step(y_true=10.0, lower=8.0, upper=12.0, y_pred=10.0)
    assert state_stat["gamma_t"] == 0.005

    # Feed a series of massive outliers to trigger shift
    for _ in range(50):
        sa.step(y_true=50.0, lower=8.0, upper=12.0, y_pred=10.0)

    # Now shift flag should be True and gamma escalated to gamma_fast
    assert sa.history[-1]["shift_flag"] is True
    assert sa.history[-1]["gamma_t"] == 0.035


def test_sa_acp_ablation_disabled_detector_reduces_to_fixed_gamma():
    rng = np.random.default_rng(42)
    cal_scores = rng.normal(10.0, 2.0, 200)

    # Disabled detector (Ablation A4)
    sa_dis = ShiftAwareAdaptiveConformal(
        alpha=0.10,
        gamma_slow=0.01,
        gamma_fast=0.05,
        enable_detector=False
    )
    sa_dis.calibrate(cal_scores)

    # Even with massive outliers, gamma must remain fixed at gamma_slow and shift_flag must be False
    for _ in range(30):
        state = sa_dis.step(y_true=100.0, lower=8.0, upper=12.0, y_pred=10.0)
        assert state["shift_flag"] is False
        assert state["gamma_t"] == 0.01
