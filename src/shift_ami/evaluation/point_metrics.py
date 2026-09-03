"""
Point forecast accuracy metrics: MAE, RMSE, NMAE, and sMAPE.
"""
from typing import Dict, Union
import numpy as np


def calculate_point_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list]
) -> Dict[str, float]:
    """
    Compute standard point forecasting metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth realized load.
    y_pred : array-like
        Point forecasts.

    Returns
    -------
    Dict[str, float] with keys:
        - mae: Mean Absolute Error
        - rmse: Root Mean Squared Error
        - nmae: Normalized MAE (MAE / mean(y_true))
        - smape: Symmetric Mean Absolute Percentage Error (%)
    """
    y_t = np.asarray(y_true, dtype=float).flatten()
    y_p = np.asarray(y_pred, dtype=float).flatten()

    if len(y_t) == 0 or len(y_p) == 0:
        return {"mae": np.nan, "rmse": np.nan, "nmae": np.nan, "smape": np.nan}

    errors = y_t - y_p
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    mean_true = float(np.mean(y_t))
    nmae = float(mae / max(mean_true, 1e-6))

    # sMAPE with small epsilon in denominator to avoid 0/0
    denominator = (np.abs(y_t) + np.abs(y_p)) / 2.0 + 1e-6
    smape = float(np.mean(abs_errors / denominator) * 100.0)

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "nmae": round(nmae, 4),
        "smape": round(smape, 4)
    }
