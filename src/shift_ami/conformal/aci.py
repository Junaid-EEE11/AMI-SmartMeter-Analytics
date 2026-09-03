"""
P4: Adaptive Conformal Inference (ACI) (Gibbs & Candès, 2021).
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from shift_ami.conformal.static import compute_conformal_quantile


class AdaptiveConformalInference:
    """
    Adaptive Conformal Inference (ACI) for sequential online uncertainty calibration.
    Maintains adaptive miscoverage parameter alpha_t updated via:
        alpha_{t+1} = clip(alpha_t + gamma * (alpha - err_t), alpha_min, alpha_max)
    """
    def __init__(
        self,
        alpha: float = 0.10,
        gamma: float = 0.01,
        alpha_min: float = 0.01,
        alpha_max: float = 0.50,
        name: str = "ACI"
    ):
        self.alpha_nominal = alpha
        self.alpha_t = alpha
        self.gamma = gamma
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.name = name

        self.calibration_scores: np.ndarray = np.array([])
        self.history: List[Dict[str, float]] = []

    def calibrate(self, cal_scores: np.ndarray) -> "AdaptiveConformalInference":
        """Seed ACI with calibration nonconformity scores."""
        self.calibration_scores = np.asarray(cal_scores).flatten()
        self.alpha_t = self.alpha_nominal
        self.history.clear()
        return self

    def get_current_conformal_quantile(self) -> float:
        """Compute conformal quantile on calibration scores evaluated at alpha_t."""
        return compute_conformal_quantile(self.calibration_scores, self.alpha_t)

    def predict_interval(
        self,
        y_pred: np.ndarray,
        q_lo: Optional[np.ndarray] = None,
        q_hi: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Produce prediction intervals at current step t using alpha_t.
        Supports both point predictions and base quantile intervals.
        """
        q_val = self.get_current_conformal_quantile()
        if q_lo is not None and q_hi is not None:
            lower = np.maximum(0.0, np.asarray(q_lo) - q_val)
            upper = np.maximum(lower, np.asarray(q_hi) + q_val)
        else:
            y_p = np.asarray(y_pred)
            lower = np.maximum(0.0, y_p - q_val)
            upper = y_p + q_val
        return lower, upper

    def step(
        self,
        y_true: Union[float, np.ndarray],
        lower: Union[float, np.ndarray],
        upper: Union[float, np.ndarray]
    ) -> float:
        """
        Observe true outcome(s), compute empirical miscoverage error, and update alpha_{t+1}.

        err_t = 1 if y not in [lower, upper] else 0
        alpha_{t+1} = clip(alpha_t + gamma * (alpha_nominal - err_t), alpha_min, alpha_max)
        """
        y_t = np.asarray(y_true).flatten()
        low = np.asarray(lower).flatten()
        high = np.asarray(upper).flatten()

        # Mean miscoverage over the step
        is_covered = (y_t >= low) & (y_t <= high)
        err_t = 1.0 - float(np.mean(is_covered))

        # Record state prior to update
        self.history.append({
            "alpha_t": self.alpha_t,
            "conformal_quantile": self.get_current_conformal_quantile(),
            "err_t": err_t,
            "covered": float(np.mean(is_covered)),
            "width": float(np.mean(high - low))
        })

        # Update rule
        alpha_next = self.alpha_t + self.gamma * (self.alpha_nominal - err_t)
        self.alpha_t = float(np.clip(alpha_next, self.alpha_min, self.alpha_max))

        return self.alpha_t
