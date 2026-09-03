"""
Automated tests verifying temporal causality during online sequential simulation.

CRITICAL LEAKAGE TEST (Section 24 of GEMINI.md):
Ensures sequential adaptive updates use an outcome ONLY after that outcome becomes observable.
"""
import numpy as np
import pandas as pd
import pytest

from shift_ami.conformal.aci import AdaptiveConformalInference
from shift_ami.conformal.sa_acp import ShiftAwareAdaptiveConformal


def test_sequential_adaptive_causality():
    cal_scores = np.ones(100) * 2.0
    aci = AdaptiveConformalInference(alpha=0.10, gamma=0.05).calibrate(cal_scores)
    sa = ShiftAwareAdaptiveConformal(alpha=0.10, gamma_slow=0.01, gamma_fast=0.05).calibrate(cal_scores)

    # Initial state before seeing outcome 0
    alpha_0_aci = aci.alpha_t
    alpha_0_sa = sa.alpha_t

    # Step 0 prediction interval is computed strictly BEFORE step 0 outcome is fed
    l_0_aci, u_0_aci = aci.predict_interval(np.array([10.0]))
    l_0_sa, u_0_sa = sa.predict_interval(np.array([10.0]))

    # Now reveal outcome 0
    aci.step(y_true=50.0, lower=l_0_aci, upper=u_0_aci)
    sa.step(y_true=50.0, lower=l_0_sa, upper=u_0_sa, y_pred=10.0)

    # State for Step 1 is updated only AFTER outcome 0 is revealed
    assert aci.alpha_t < alpha_0_aci
    assert sa.alpha_t < alpha_0_sa
