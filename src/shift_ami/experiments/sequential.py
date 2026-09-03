"""
Sequential Online Evaluation Engine for Day-Ahead Conformal Prediction.

CRITICAL RESEARCH RULE (Section 11 & 24 of GEMINI.md):
Simulate forecasting sequentially through the test period.
At each forecast origin:
1. Generate day-ahead predictions (h=1..48);
2. Wait conceptually for outcomes to become observable;
3. Update adaptive calibration state using strictly already-observed outcomes;
4. Advance to the next origin.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from shift_ami.conformal.static import StaticSplitConformal
from shift_ami.conformal.cqr import ConformalizedQuantileRegression
from shift_ami.conformal.rolling import RollingWindowConformal
from shift_ami.conformal.aci import AdaptiveConformalInference
from shift_ami.conformal.sa_acp import ShiftAwareAdaptiveConformal
from shift_ami.utils.logging import setup_logger

logger = setup_logger("shift_ami.experiments.sequential")


def run_sequential_evaluation(
    df_test_features: pd.DataFrame,
    y_test_true: pd.Series,
    test_timestamps: pd.Series,
    point_predictions: Dict[str, np.ndarray],
    quantile_predictions: Dict[float, np.ndarray],
    cal_true: np.ndarray,
    cal_pred_point: np.ndarray,
    cal_q_lo: np.ndarray,
    cal_q_hi: np.ndarray,
    conformal_alpha: float = 0.10,
    aci_gamma: float = 0.01,
    sa_acp_params: Optional[Dict[str, Any]] = None,
    rolling_window_steps: int = 1344,
    step_block_size: int = 48  # Day-ahead origin step size
) -> pd.DataFrame:
    """
    Execute sequential online operational simulation on the test set.

    Initializes:
    - P0: Uncalibrated Quantiles (q_lo, q_hi)
    - P1: Static Split Conformal (fixed calibration quantile)
    - P2: CQR (Conformalized Quantile Regression)
    - P3: Rolling Conformal (sliding FIFO buffer)
    - P4: ACI (Adaptive Conformal Inference)
    - P5: SA-ACP (Shift-Aware Adaptive Conformal Prediction)

    Returns
    -------
    pd.DataFrame
        Detailed sequential test evaluation record containing all bounds, errors, widths,
        and diagnostic states across every half-hour test step.
    """
    logger.info(f"Starting sequential evaluation over {len(y_test_true):,} test time steps...")

    y_test = y_test_true.values
    n_total = len(y_test)
    primary_point_pred = point_predictions.get("hist_gradient_boosting", list(point_predictions.values())[0])

    q_lo_key = min(quantile_predictions.keys())
    q_hi_key = max(quantile_predictions.keys())
    q_lo_test = quantile_predictions[q_lo_key]
    q_hi_test = quantile_predictions[q_hi_key]

    # 1. Initialize P1: Static Split Conformal
    static_model = StaticSplitConformal(alpha=conformal_alpha)
    static_model.calibrate(cal_true, cal_pred_point)

    # 2. Initialize P2: CQR
    cqr_model = ConformalizedQuantileRegression(alpha=conformal_alpha)
    cqr_model.calibrate(cal_true, cal_q_lo, cal_q_hi)

    # 3. Initialize P3: Rolling Conformal
    cal_scores = np.abs(cal_true - cal_pred_point)
    rolling_model = RollingWindowConformal(alpha=conformal_alpha, window_size=rolling_window_steps)
    rolling_model.initialize_with_calibration_scores(cal_scores)

    # 4. Initialize P4: ACI
    aci_model = AdaptiveConformalInference(alpha=conformal_alpha, gamma=aci_gamma)
    aci_model.calibrate(cal_scores)

    # 5. Initialize P5: SA-ACP
    sa_params = sa_acp_params or {}
    sa_acp_model = ShiftAwareAdaptiveConformal(
        alpha=conformal_alpha,
        gamma_slow=sa_params.get("gamma_slow", 0.005),
        gamma_fast=sa_params.get("gamma_fast", 0.035),
        shift_penalty_eta=sa_params.get("shift_penalty_eta", 0.02),
        threshold=sa_params.get("threshold", 1.25),
        window_length=sa_params.get("window_length", 336),
        reference_length=sa_params.get("reference_length", 1344),
        enable_detector=sa_params.get("enable_detector", True),
        detector_type=sa_params.get("detector", "wasserstein_1d")
    )
    sa_acp_model.calibrate(cal_scores)

    # Output arrays
    p0_lower, p0_upper = np.zeros(n_total), np.zeros(n_total)
    p1_lower, p1_upper = np.zeros(n_total), np.zeros(n_total)
    p2_lower, p2_upper = np.zeros(n_total), np.zeros(n_total)
    p3_lower, p3_upper = np.zeros(n_total), np.zeros(n_total)
    p4_lower, p4_upper = np.zeros(n_total), np.zeros(n_total)
    p5_lower, p5_upper = np.zeros(n_total), np.zeros(n_total)

    sa_shift_scores = np.zeros(n_total)
    sa_shift_flags = np.zeros(n_total, dtype=bool)
    sa_alphas = np.zeros(n_total)
    sa_gammas = np.zeros(n_total)
    aci_alphas = np.zeros(n_total)

    # Step through sequentially by daily blocks (step_block_size = 48)
    for start_idx in range(0, n_total, step_block_size):
        end_idx = min(start_idx + step_block_size, n_total)
        block_len = end_idx - start_idx

        # --- A. PREDICTION PHASE (Before outcomes are observed) ---
        # P0: Uncalibrated
        p0_l = np.maximum(0.0, q_lo_test[start_idx:end_idx])
        p0_u = np.maximum(p0_l, q_hi_test[start_idx:end_idx])
        p0_lower[start_idx:end_idx] = p0_l
        p0_upper[start_idx:end_idx] = p0_u

        # P1: Static Conformal
        p1_l, p1_u = static_model.predict_interval(primary_point_pred[start_idx:end_idx])
        p1_lower[start_idx:end_idx] = p1_l
        p1_upper[start_idx:end_idx] = p1_u

        # P2: CQR
        p2_l, p2_u = cqr_model.predict_interval(q_lo_test[start_idx:end_idx], q_hi_test[start_idx:end_idx])
        p2_lower[start_idx:end_idx] = p2_l
        p2_upper[start_idx:end_idx] = p2_u

        # P3: Rolling Conformal
        p3_l, p3_u = rolling_model.predict_interval(primary_point_pred[start_idx:end_idx])
        p3_lower[start_idx:end_idx] = p3_l
        p3_upper[start_idx:end_idx] = p3_u

        # P4: ACI
        aci_alphas[start_idx:end_idx] = aci_model.alpha_t
        p4_l, p4_u = aci_model.predict_interval(primary_point_pred[start_idx:end_idx])
        p4_lower[start_idx:end_idx] = p4_l
        p4_upper[start_idx:end_idx] = p4_u

        # P5: SA-ACP
        sa_alphas[start_idx:end_idx] = sa_acp_model.alpha_t
        p5_l, p5_u = sa_acp_model.predict_interval(primary_point_pred[start_idx:end_idx])
        p5_lower[start_idx:end_idx] = p5_l
        p5_upper[start_idx:end_idx] = p5_u

        # --- B. OUTCOME REVELATION & ADAPTATION UPDATE PHASE ---
        y_block = y_test[start_idx:end_idx]
        pred_block = primary_point_pred[start_idx:end_idx]
        scores_block = np.abs(y_block - pred_block)

        # Update P3: Rolling
        rolling_model.update(scores_block)

        # Update P4: ACI
        aci_model.step(y_block, p4_l, p4_u)

        # Update P5: SA-ACP
        for i in range(block_len):
            idx = start_idx + i
            state = sa_acp_model.step(y_test[idx], p5_lower[idx], p5_upper[idx], y_pred=primary_point_pred[idx])
            sa_shift_scores[idx] = state["shift_score"]
            sa_shift_flags[idx] = state["shift_flag"]
            sa_gammas[idx] = state["gamma_t"]

    # Assemble comprehensive results DataFrame
    results_df = pd.DataFrame({
        "timestamp": test_timestamps.values,
        "y_true": y_test,
        "y_point_pred": primary_point_pred,
        # P0
        "uncalibrated_lower": p0_lower,
        "uncalibrated_upper": p0_upper,
        # P1
        "static_lower": p1_lower,
        "static_upper": p1_upper,
        # P2
        "cqr_lower": p2_lower,
        "cqr_upper": p2_upper,
        # P3
        "rolling_lower": p3_lower,
        "rolling_upper": p3_upper,
        # P4
        "aci_lower": p4_lower,
        "aci_upper": p4_upper,
        "aci_alpha": aci_alphas,
        # P5
        "sa_acp_lower": p5_lower,
        "sa_acp_upper": p5_upper,
        "sa_acp_alpha": sa_alphas,
        "sa_acp_gamma": sa_gammas,
        "shift_score": sa_shift_scores,
        "shift_flag": sa_shift_flags
    })

    # Add point forecast baselines if available
    for model_name, preds in point_predictions.items():
        results_df[f"point_pred_{model_name}"] = preds

    logger.info(f"Sequential evaluation finished successfully for {len(results_df):,} test time steps.")
    return results_df
