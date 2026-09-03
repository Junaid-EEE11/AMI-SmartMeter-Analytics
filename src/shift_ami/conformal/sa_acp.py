"""
P5: Proposed Shift-Aware Adaptive Conformal Prediction (SA-ACP).

A sequential uncertainty calibrator that detects non-parametric distribution shift
and dynamically modulates adaptation learning rates to restore nominal coverage.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from shift_ami.conformal.static import compute_conformal_quantile
from shift_ami.conformal.shift_detector import WassersteinShiftDetector, StandardizedResidualShiftDetector


class ShiftAwareAdaptiveConformal:
    """
    Shift-Aware Adaptive Conformal Prediction (SA-ACP).

    At each sequential step t:
    1. Generates prediction interval using adaptive miscoverage parameter alpha_t.
    2. Receives ground truth y_t and evaluates coverage error err_t = I(y_t not in [L_t, U_t]).
    3. Updates trailing nonconformity window and computes sequential shift score delta_t.
    4. Evaluates shift_flag_t = (delta_t >= threshold).
    5. Modulates learning rate gamma_t:
          gamma_t = gamma_fast if shift_flag_t else gamma_slow
    6. Updates alpha_{t+1} = clip(alpha_t + gamma_t * (alpha_nominal - err_t) - eta * delta_t * err_t, alpha_min, alpha_max).
    7. Stores transparent diagnostic trace.
    """
    def __init__(
        self,
        alpha: float = 0.10,
        gamma_slow: float = 0.005,
        gamma_fast: float = 0.035,
        shift_penalty_eta: float = 0.02,
        threshold: float = 1.25,
        window_length: int = 336,
        reference_length: int = 1344,
        alpha_min: float = 0.01,
        alpha_max: float = 0.50,
        enable_detector: bool = True,
        detector_type: str = "wasserstein_1d",
        name: str = "SA_ACP"
    ):
        self.alpha_nominal = alpha
        self.alpha_t = alpha
        self.gamma_slow = gamma_slow
        self.gamma_fast = gamma_fast
        self.shift_penalty_eta = shift_penalty_eta
        self.threshold = threshold
        self.window_length = window_length
        self.reference_length = reference_length
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.enable_detector = enable_detector
        self.name = name

        # Instantiate selected shift detector
        if detector_type == "wasserstein_1d":
            self.detector = WassersteinShiftDetector(
                window_length=window_length,
                reference_length=reference_length,
                threshold=threshold
            )
        else:
            self.detector = StandardizedResidualShiftDetector(
                window_length=window_length,
                threshold=threshold
            )

        self.calibration_scores: np.ndarray = np.array([])
        self.history: List[Dict[str, Any]] = []

    def calibrate(self, cal_scores: np.ndarray) -> "ShiftAwareAdaptiveConformal":
        """Seed SA-ACP with baseline calibration scores and set reference distribution."""
        self.calibration_scores = np.asarray(cal_scores).flatten()
        self.detector.set_reference(self.calibration_scores)
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
        Generate prediction interval for current step using adaptive alpha_t.
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
        upper: Union[float, np.ndarray],
        y_pred: Optional[Union[float, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Sequential observation and adaptation step.
        """
        y_t = np.asarray(y_true).flatten()
        low = np.asarray(lower).flatten()
        high = np.asarray(upper).flatten()

        # Step error and coverage
        is_covered = (y_t >= low) & (y_t <= high)
        err_t = 1.0 - float(np.mean(is_covered))

        # Compute nonconformity score for current step
        if y_pred is not None:
            score_t = np.abs(y_t - np.asarray(y_pred).flatten())
        else:
            # Boundary violation score
            score_t = np.maximum(low - y_t, y_t - high)

        # Shift detection
        if self.enable_detector:
            shift_score, shift_flag = self.detector.update_and_detect(score_t)
        else:
            # Detector disabled (Ablation A4 -> reduces to standard ACI)
            shift_score, shift_flag = 0.0, False

        # Select dynamic adaptation rate
        gamma_t = self.gamma_fast if shift_flag else self.gamma_slow

        # Record diagnostic state prior to update
        current_q = self.get_current_conformal_quantile()
        state_record = {
            "alpha_t": self.alpha_t,
            "conformal_quantile": current_q,
            "shift_score": shift_score,
            "shift_flag": shift_flag,
            "gamma_t": gamma_t,
            "err_t": err_t,
            "covered": float(np.mean(is_covered)),
            "interval_width": float(np.mean(high - low)),
            "lower_bound_mean": float(np.mean(low)),
            "upper_bound_mean": float(np.mean(high))
        }
        self.history.append(state_record)

        # Update rule:
        # alpha_{t+1} = clip(alpha_t + gamma_t * (alpha - err_t) - eta * shift_score * err_t, alpha_min, alpha_max)
        shift_penalty = (self.shift_penalty_eta * shift_score * err_t) if shift_flag else 0.0
        alpha_next = self.alpha_t + gamma_t * (self.alpha_nominal - err_t) - shift_penalty
        self.alpha_t = float(np.clip(alpha_next, self.alpha_min, self.alpha_max))

        return state_record
