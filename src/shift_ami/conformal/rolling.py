"""
P3: Rolling-Window Conformal Recalibration.
"""
from collections import deque
from typing import List, Optional, Tuple, Union
import numpy as np

from shift_ami.conformal.static import compute_conformal_quantile


class RollingWindowConformal:
    """
    Rolling-window conformal calibrator maintaining a sliding FIFO window of
    recent nonconformity scores observed during online sequential evaluation.
    """
    def __init__(
        self,
        alpha: float = 0.10,
        window_size: int = 1344,  # 28 days * 48 half-hours
        name: str = "RollingConformal"
    ):
        self.alpha = alpha
        self.window_size = window_size
        self.name = name
        self.buffer: deque = deque(maxlen=window_size)
        self.current_quantile: float = 0.0

    def initialize_with_calibration_scores(self, cal_scores: np.ndarray) -> "RollingWindowConformal":
        """Seed the rolling window with baseline calibration scores."""
        self.buffer.clear()
        scores = np.asarray(cal_scores).flatten()
        # Fill buffer up to window_size
        for s in scores[-self.window_size:]:
            self.buffer.append(float(s))
        self._recompute_quantile()
        return self

    def _recompute_quantile(self) -> float:
        if len(self.buffer) == 0:
            self.current_quantile = 0.0
        else:
            self.current_quantile = compute_conformal_quantile(np.array(self.buffer), self.alpha)
        return self.current_quantile

    def update(self, new_scores: Union[float, np.ndarray, List[float]]) -> float:
        """
        Append newly observed score(s) after outcome is revealed and recompute quantile.
        """
        if isinstance(new_scores, (int, float)):
            self.buffer.append(float(new_scores))
        else:
            for s in np.asarray(new_scores).flatten():
                self.buffer.append(float(s))
        return self._recompute_quantile()

    def predict_interval(
        self,
        y_pred: np.ndarray,
        q_override: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate prediction intervals using current rolling quantile."""
        q_val = q_override if q_override is not None else self.current_quantile
        y_p = np.asarray(y_pred)
        lower = np.maximum(0.0, y_p - q_val)
        upper = y_p + q_val
        return lower, upper
