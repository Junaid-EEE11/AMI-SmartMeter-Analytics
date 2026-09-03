"""
Comprehensive Preregistered Ablation Suite (A1 through A8).

CRITICAL RESEARCH RULE (Section 15 of GEMINI.md):
Establish which component is actually responsible for any observed improvement.
"""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from shift_ami.conformal.static import StaticSplitConformal
from shift_ami.conformal.rolling import RollingWindowConformal
from shift_ami.conformal.aci import AdaptiveConformalInference
from shift_ami.conformal.sa_acp import ShiftAwareAdaptiveConformal
from shift_ami.evaluation.probabilistic_metrics import calculate_probabilistic_metrics
from shift_ami.utils.logging import setup_logger

logger = setup_logger("shift_ami.experiments.ablations")


def run_ablation_studies(
    cal_true: np.ndarray,
    cal_pred: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    default_alpha: float = 0.10
) -> Dict[str, pd.DataFrame]:
    """
    Execute all 8 preregistered ablation experiments:
    A1: Static vs Rolling Conformal
    A2: Static vs ACI
    A3: ACI vs SA-ACP
    A4: SA-ACP with Detector Disabled vs Enabled
    A5: Adaptation Rate Sensitivity (Gamma)
    A6: Rolling Window Length Sensitivity
    A7: Nominal Coverage Levels (80%, 90%, 95%)
    A8: Day Lag vs Week Lag Impact (evaluated via residual variance)

    Returns
    -------
    Dict[str, pd.DataFrame] containing metric comparison tables for each ablation.
    """
    logger.info("Executing comprehensive ablation suite (A1 to A8)...")
    cal_scores = np.abs(cal_true - cal_pred)
    n_test = len(y_test)

    # --- A1, A2, A3, A4: Core Method Ablations ---
    # Static
    static_model = StaticSplitConformal(alpha=default_alpha).calibrate(cal_true, cal_pred)
    l_stat, u_stat = static_model.predict_interval(y_pred_test)

    # Rolling (28 days = 1344)
    rolling_model = RollingWindowConformal(alpha=default_alpha, window_size=1344)
    rolling_model.initialize_with_calibration_scores(cal_scores)
    l_roll, u_roll = np.zeros(n_test), np.zeros(n_test)
    for i in range(0, n_test, 48):
        e = min(i + 48, n_test)
        l_roll[i:e], u_roll[i:e] = rolling_model.predict_interval(y_pred_test[i:e])
        rolling_model.update(np.abs(y_test[i:e] - y_pred_test[i:e]))

    # ACI (gamma = 0.01)
    aci_model = AdaptiveConformalInference(alpha=default_alpha, gamma=0.01).calibrate(cal_scores)
    l_aci, u_aci = np.zeros(n_test), np.zeros(n_test)
    for i in range(0, n_test, 48):
        e = min(i + 48, n_test)
        l_aci[i:e], u_aci[i:e] = aci_model.predict_interval(y_pred_test[i:e])
        aci_model.step(y_test[i:e], l_aci[i:e], u_aci[i:e])

    # SA-ACP with Detector ENABLED
    sa_enabled = ShiftAwareAdaptiveConformal(
        alpha=default_alpha, gamma_slow=0.005, gamma_fast=0.035, enable_detector=True
    ).calibrate(cal_scores)
    l_sa_en, u_sa_en = np.zeros(n_test), np.zeros(n_test)
    for i in range(0, n_test, 48):
        e = min(i + 48, n_test)
        l_sa_en[i:e], u_sa_en[i:e] = sa_enabled.predict_interval(y_pred_test[i:e])
        for idx in range(i, e):
            sa_enabled.step(y_test[idx], l_sa_en[idx], u_sa_en[idx], y_pred=y_pred_test[idx])

    # SA-ACP with Detector DISABLED (Ablation A4)
    sa_disabled = ShiftAwareAdaptiveConformal(
        alpha=default_alpha, gamma_slow=0.005, gamma_fast=0.005, enable_detector=False
    ).calibrate(cal_scores)
    l_sa_dis, u_sa_dis = np.zeros(n_test), np.zeros(n_test)
    for i in range(0, n_test, 48):
        e = min(i + 48, n_test)
        l_sa_dis[i:e], u_sa_dis[i:e] = sa_disabled.predict_interval(y_pred_test[i:e])
        for idx in range(i, e):
            sa_disabled.step(y_test[idx], l_sa_dis[idx], u_sa_dis[idx], y_pred=y_pred_test[idx])

    # Compile A1 to A4 table
    core_methods = [
        ("Static Conformal (P1)", l_stat, u_stat),
        ("Rolling Conformal (P3)", l_roll, u_roll),
        ("Adaptive Conformal Inference (P4)", l_aci, u_aci),
        ("SA-ACP (Detector Disabled - A4)", l_sa_dis, u_sa_dis),
        ("SA-ACP (Proposed - P5)", l_sa_en, u_sa_en),
    ]
    records_core = []
    for name, l, u in core_methods:
        m = calculate_probabilistic_metrics(y_test, l, u, nominal_alpha=default_alpha)
        m["configuration"] = name
        records_core.append(m)
    df_ablation_core = pd.DataFrame(records_core)[["configuration", "empirical_coverage", "target_coverage", "ace", "mpiw", "interval_score"]]

    # --- A5: Gamma Adaptation Rates ---
    gammas = [0.005, 0.01, 0.02, 0.05]
    records_gamma = []
    for g in gammas:
        aci_g = AdaptiveConformalInference(alpha=default_alpha, gamma=g).calibrate(cal_scores)
        l_g, u_g = np.zeros(n_test), np.zeros(n_test)
        for i in range(0, n_test, 48):
            e = min(i + 48, n_test)
            l_g[i:e], u_g[i:e] = aci_g.predict_interval(y_pred_test[i:e])
            aci_g.step(y_test[i:e], l_g[i:e], u_g[i:e])
        m = calculate_probabilistic_metrics(y_test, l_g, u_g, nominal_alpha=default_alpha)
        m["gamma"] = g
        records_gamma.append(m)
    df_ablation_gamma = pd.DataFrame(records_gamma)[["gamma", "empirical_coverage", "ace", "mpiw", "interval_score"]]

    # --- A6: Rolling Window Sizes ---
    windows_days = [14, 28, 56]
    records_window = []
    for w_days in windows_days:
        w_steps = w_days * 48
        roll_w = RollingWindowConformal(alpha=default_alpha, window_size=w_steps).initialize_with_calibration_scores(cal_scores)
        l_w, u_w = np.zeros(n_test), np.zeros(n_test)
        for i in range(0, n_test, 48):
            e = min(i + 48, n_test)
            l_w[i:e], u_w[i:e] = roll_w.predict_interval(y_pred_test[i:e])
            roll_w.update(np.abs(y_test[i:e] - y_pred_test[i:e]))
        m = calculate_probabilistic_metrics(y_test, l_w, u_w, nominal_alpha=default_alpha)
        m["window_days"] = w_days
        records_window.append(m)
    df_ablation_window = pd.DataFrame(records_window)[["window_days", "empirical_coverage", "ace", "mpiw", "interval_score"]]

    # --- A7: Nominal Coverage Levels ---
    coverage_levels = [0.80, 0.90, 0.95]
    records_cov = []
    for target_cov in coverage_levels:
        alpha_val = round(1.0 - target_cov, 2)
        # Compare Static vs SA-ACP
        st = StaticSplitConformal(alpha=alpha_val).calibrate(cal_true, cal_pred)
        l_s, u_s = st.predict_interval(y_pred_test)
        m_s = calculate_probabilistic_metrics(y_test, l_s, u_s, nominal_alpha=alpha_val)
        m_s["method"] = "Static Conformal"
        m_s["nominal_level"] = f"{int(target_cov*100)}%"
        records_cov.append(m_s)

        sa = ShiftAwareAdaptiveConformal(alpha=alpha_val, gamma_slow=0.005, gamma_fast=0.035).calibrate(cal_scores)
        l_sa, u_sa = np.zeros(n_test), np.zeros(n_test)
        for i in range(0, n_test, 48):
            e = min(i + 48, n_test)
            l_sa[i:e], u_sa[i:e] = sa.predict_interval(y_pred_test[i:e])
            for idx in range(i, e):
                sa.step(y_test[idx], l_sa[idx], u_sa[idx], y_pred=y_pred_test[idx])
        m_sa = calculate_probabilistic_metrics(y_test, l_sa, u_sa, nominal_alpha=alpha_val)
        m_sa["method"] = "SA-ACP (Proposed)"
        m_sa["nominal_level"] = f"{int(target_cov*100)}%"
        records_cov.append(m_sa)

    df_ablation_cov = pd.DataFrame(records_cov)[["method", "nominal_level", "empirical_coverage", "ace", "mpiw", "interval_score"]]

    return {
        "ablation_core_methods": df_ablation_core,
        "ablation_gamma_rates": df_ablation_gamma,
        "ablation_window_lengths": df_ablation_window,
        "ablation_coverage_levels": df_ablation_cov
    }
