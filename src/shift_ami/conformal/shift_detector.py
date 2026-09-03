"""
Sequential Distribution Shift Detectors for Conformal Prediction Monitoring.
"""
from collections import deque
from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp


class WassersteinShiftDetector:
    """
    Principal Shift Detector using 1D Wasserstein Distance (Earth Mover's Distance).
    Compares the empirical distribution of trailing nonconformity scores / residuals
    against a stationary reference calibration distribution.
    """
    def __init__(
        self,
        window_length: int = 336,       # Trailing 7 days (336 half-hours)
        reference_length: int = 1344,   # Reference 28 days
        threshold: float = 1.25,        # Validation-tuned threshold
        name: str = "Wasserstein_1D"
    ):
        self.window_length = window_length
        self.reference_length = reference_length
        self.threshold = threshold
        self.name = name

        self.reference_scores: np.ndarray = np.array([])
        self.reference_mean: float = 0.0
        self.reference_std: float = 1.0
        self.trailing_buffer: deque = deque(maxlen=window_length)

    def set_reference(self, reference_scores: np.ndarray) -> "WassersteinShiftDetector":
        """Fit reference baseline distribution from calibration/validation set."""
        ref = np.asarray(reference_scores).flatten()
        if len(ref) > self.reference_length:
            ref = ref[-self.reference_length:]
        self.reference_scores = ref
        self.reference_mean = float(np.mean(ref))
        self.reference_std = max(float(np.std(ref)), 1e-6)
        self.trailing_buffer.clear()
        return self

    def update_and_detect(self, new_score: Union[float, np.ndarray, List[float]]) -> Tuple[float, bool]:
        """
        Append newly observed score, compute 1D Wasserstein distance to reference,
        normalize by reference dispersion, and evaluate binary shift flag.

        Returns
        -------
        Tuple[float, bool]
            (shift_score, shift_flag)
        """
        if isinstance(new_score, (int, float)):
            self.trailing_buffer.append(float(new_score))
        else:
            for s in np.asarray(new_score).flatten():
                self.trailing_buffer.append(float(s))

        # Require at least 25% of window buffer to compute robust distance
        if len(self.trailing_buffer) < max(24, self.window_length // 4) or len(self.reference_scores) == 0:
            return 0.0, False

        trailing_arr = np.array(self.trailing_buffer)
        w1_dist = float(wasserstein_distance(trailing_arr, self.reference_scores))
        
        # Standardized shift score
        shift_score = w1_dist / self.reference_std
        shift_flag = bool(shift_score >= self.threshold)

        return shift_score, shift_flag


class StandardizedResidualShiftDetector:
    """
    Alternative / sensitivity detector based on standardized mean and variance deviation.
    """
    def __init__(
        self,
        window_length: int = 336,
        threshold: float = 2.0,
        name: str = "StdResidual_Detector"
    ):
        self.window_length = window_length
        self.threshold = threshold
        self.name = name
        self.reference_mean: float = 0.0
        self.reference_std: float = 1.0
        self.trailing_buffer: deque = deque(maxlen=window_length)

    def set_reference(self, reference_scores: np.ndarray) -> "StandardizedResidualShiftDetector":
        ref = np.asarray(reference_scores).flatten()
        self.reference_mean = float(np.mean(ref))
        self.reference_std = max(float(np.std(ref)), 1e-6)
        self.trailing_buffer.clear()
        return self

    def update_and_detect(self, new_score: Union[float, np.ndarray, List[float]]) -> Tuple[float, bool]:
        if isinstance(new_score, (int, float)):
            self.trailing_buffer.append(float(new_score))
        else:
            for s in np.asarray(new_score).flatten():
                self.trailing_buffer.append(float(s))

        if len(self.trailing_buffer) < 24:
            return 0.0, False

        trailing_mean = float(np.mean(self.trailing_buffer))
        z_score = abs(trailing_mean - self.reference_mean) / (self.reference_std / np.sqrt(len(self.trailing_buffer)))
        shift_flag = bool(z_score >= self.threshold)
        return z_score, shift_flag
