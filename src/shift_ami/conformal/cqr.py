"""
P2: Conformalized Quantile Regression (CQR) (Romano, Sesia, & Candès, 2019).
"""
from typing import Optional, Tuple, Union
import numpy as np

from shift_ami.conformal.static import compute_conformal_quantile


class ConformalizedQuantileRegression:
    """
    Conformalized Quantile Regression calibrator.
    Computes signed conformity boundary violations on calibration set:
        s_i = max( \hat{q}_{lo}(x_i) - y_i, y_i - \hat{q}_{hi}(x_i) )
    """
    def __init__(self, alpha: float = 0.10, name: str = "CQR"):
        self.alpha = alpha
        self.name = name
        self.calibration_scores: np.ndarray = np.array([])
        self.conformal_quantile: float = 0.0
        self.is_calibrated: bool = False

    def calibrate(
        self,
        y_true_cal: np.ndarray,
        q_lo_cal: np.ndarray,
        q_hi_cal: np.ndarray
    ) -> "ConformalizedQuantileRegression":
        """
        Calibrate CQR nonconformity scores using lower and upper quantile estimates.
        """
        y_t = np.asarray(y_true_cal)
        q_l = np.asarray(q_lo_cal)
        q_h = np.asarray(q_hi_cal)

        self.calibration_scores = np.maximum(q_l - y_t, y_t - q_h)
        self.conformal_quantile = compute_conformal_quantile(self.calibration_scores, self.alpha)
        self.is_calibrated = True
        return self

    def predict_interval(
        self,
        q_lo: np.ndarray,
        q_hi: np.ndarray,
        alpha_override: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate conformalized prediction intervals:
            [max(0, q_lo - Q), q_hi + Q]
        """
        if not self.is_calibrated:
            raise RuntimeError("CQR must be calibrated before generating intervals.")
        
        q_val = self.conformal_quantile
        if alpha_override is not None:
            q_val = compute_conformal_quantile(self.calibration_scores, alpha_override)

        lower = np.maximum(0.0, np.asarray(q_lo) - q_val)
        upper = np.maximum(lower, np.asarray(q_hi) + q_val)
        return lower, upper
