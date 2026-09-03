"""
Probabilistic forecast evaluation metrics: Empirical Coverage, ACE, MPIW, Winkler Interval Score, and Pinball Loss.
"""
from typing import Dict, Union
import numpy as np


def calculate_interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.10
) -> float:
    """
    Compute standard Winkler / Interval Score (IS):
        IS_alpha = (upper - lower) + (2/alpha)*(lower - y)*I(y < lower) + (2/alpha)*(y - upper)*I(y > upper)
    """
    y_t = np.asarray(y_true, dtype=float).flatten()
    l = np.asarray(lower, dtype=float).flatten()
    u = np.asarray(upper, dtype=float).flatten()

    width = u - l
    under_coverage = (2.0 / alpha) * (l - y_t) * (y_t < l).astype(float)
    over_coverage = (2.0 / alpha) * (y_t - u) * (y_t > u).astype(float)

    scores = width + under_coverage + over_coverage
    return float(np.mean(scores))


def calculate_pinball_loss(
    y_true: np.ndarray,
    y_pred_quantile: np.ndarray,
    quantile: float
) -> float:
    """
    Compute pinball / quantile loss for quantile tau in (0, 1):
        L_tau(y, q) = max(tau * (y - q), (tau - 1) * (y - q))
    """
    y_t = np.asarray(y_true, dtype=float).flatten()
    q_p = np.asarray(y_pred_quantile, dtype=float).flatten()
    errors = y_t - q_p
    loss = np.maximum(quantile * errors, (quantile - 1.0) * errors)
    return float(np.mean(loss))


def calculate_probabilistic_metrics(
    y_true: Union[np.ndarray, list],
    lower: Union[np.ndarray, list],
    upper: Union[np.ndarray, list],
    nominal_alpha: float = 0.10
) -> Dict[str, float]:
    """
    Calculate comprehensive probabilistic prediction interval metrics.

    Returns
    -------
    Dict[str, float] with keys:
        - empirical_coverage: Fraction of true values within [lower, upper]
        - target_coverage: Nominal target (1 - nominal_alpha)
        - ace: Absolute Coverage Error |coverage - (1 - alpha)|
        - mpiw: Mean Prediction Interval Width (mean(upper - lower))
        - interval_score: Winkler Interval Score
    """
    y_t = np.asarray(y_true, dtype=float).flatten()
    l = np.asarray(lower, dtype=float).flatten()
    u = np.asarray(upper, dtype=float).flatten()

    if len(y_t) == 0:
        return {
            "empirical_coverage": np.nan,
            "target_coverage": 1.0 - nominal_alpha,
            "ace": np.nan,
            "mpiw": np.nan,
            "interval_score": np.nan
        }

    is_covered = (y_t >= l) & (y_t <= u)
    emp_cov = float(np.mean(is_covered))
    target_cov = 1.0 - nominal_alpha
    ace = float(abs(emp_cov - target_cov))
    mpiw = float(np.mean(u - l))
    winkler_is = calculate_interval_score(y_t, l, u, alpha=nominal_alpha)

    return {
        "empirical_coverage": round(emp_cov, 4),
        "target_coverage": round(target_cov, 4),
        "ace": round(ace, 4),
        "mpiw": round(mpiw, 4),
        "interval_score": round(winkler_is, 4)
    }
