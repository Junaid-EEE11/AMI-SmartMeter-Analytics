"""
P1: Static Split Conformal Prediction.
"""
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd


def compute_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Compute finite-sample conformal quantile:
        q_val = np.quantile(scores, ceil((n + 1) * (1 - alpha)) / n, method='higher' or linear)
    Ensures exact 1 - alpha coverage under exchangeability.
    """
    scores = np.asarray(scores).flatten()
    n = len(scores)
    if n == 0:
        return 0.0
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, q_level))


class StaticSplitConformal:
    """
    Standard static split conformal prediction calibrator.
    Computes absolute residual nonconformity scores on fixed calibration set:
        s_i = |y_i - \hat{y}_i|
    """
    def __init__(self, alpha: float = 0.10, name: str = "StaticSplitConformal"):
        self.alpha = alpha
        self.name = name
        self.calibration_scores: np.ndarray = np.array([])
        self.conformal_quantile: float = 0.0
        self.is_calibrated: bool = False

    def calibrate(self, y_true_cal: np.ndarray, y_pred_cal: np.ndarray) -> "StaticSplitConformal":
        """Calibrate nonconformity scores on held-out calibration split."""
        y_t = np.asarray(y_true_cal)
        y_p = np.asarray(y_pred_cal)
        self.calibration_scores = np.abs(y_t - y_p)
        self.conformal_quantile = compute_conformal_quantile(self.calibration_scores, self.alpha)
        self.is_calibrated = True
        return self

    def predict_interval(
        self,
        y_pred: np.ndarray,
        alpha_override: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals [lower, upper] for point predictions.
        """
        if not self.is_calibrated:
            raise RuntimeError("StaticSplitConformal must be calibrated before generating intervals.")
        
        q_val = self.conformal_quantile
        if alpha_override is not None:
            q_val = compute_conformal_quantile(self.calibration_scores, alpha_override)

        y_p = np.asarray(y_pred)
        lower = np.maximum(0.0, y_p - q_val)
        upper = y_p + q_val
        return lower, upper
