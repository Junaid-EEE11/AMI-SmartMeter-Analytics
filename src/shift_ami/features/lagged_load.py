"""
Leakage-safe lagged load feature construction for day-ahead half-hourly forecasting.

CRITICAL RESEARCH RULE (Section 9 & 24 of GEMINI.md):
At forecast origin t_0, all features for horizons h in {1..48} (target t = t_0 + h)
must strictly use historical load observed at or before t_0.
"""
from typing import List, Optional
import numpy as np
import pandas as pd


def construct_lagged_load_features(
    series: pd.Series,
    horizon: int = 48,
    lags: Optional[List[int]] = None,
    include_rolling_stats: bool = True
) -> pd.DataFrame:
    """
    Construct strictly leakage-safe lagged features for target horizon h.

    For target timestamp t and horizon h (origin t_0 = t - h):
    - Base lag features: y_{t - lag} where lag >= h.
    - Standard day-ahead lags:
        - lag 48: same half-hour previous day (y_{t - 48})
        - lag 96: same half-hour two days prior (y_{t - 96})
        - lag 144: same half-hour three days prior (y_{t - 144})
        - lag 336: same half-hour previous week (y_{t - 336})
    - Summary stats of load up to origin t_0 = t - h:
        - mean, std, min, max over [t - h - 47, t - h] (past 24h)
        - load ramp at origin: y_{t - h} - y_{t - h - 1}

    Parameters
    ----------
    series : pd.Series
        Continuous half-hourly load series indexed by timestamp.
    horizon : int
        Forecast horizon in half-hours (1 <= horizon <= 48).
    lags : Optional[List[int]]
        List of lag steps relative to target time t. Must all be >= horizon.
    include_rolling_stats : bool
        Whether to include 24h rolling stats relative to forecast origin.

    Returns
    -------
    pd.DataFrame with aligned lagged features.
    """
    if lags is None:
        lags = [48, 96, 144, 336]

    # Validate that all lags are >= horizon to guarantee no future leakage
    for lag in lags:
        if lag < horizon:
            raise ValueError(
                f"Data Leakage Violation: Requested lag {lag} is smaller than horizon {horizon}. "
                f"At forecast origin (t - {horizon}), load at (t - {lag}) has not yet been observed!"
            )

    feats = {}
    for lag in lags:
        feats[f"load_lag_{lag}"] = series.shift(lag)

    # Lags relative to origin t_0 = t - horizon
    origin_load = series.shift(horizon)  # y_{t_0}
    feats[f"origin_load_h{horizon}"] = origin_load
    feats[f"origin_ramp_h{horizon}"] = origin_load - series.shift(horizon + 1)

    if include_rolling_stats:
        # 24-hour window (48 half-hours) ending at forecast origin t_0
        r48 = origin_load.rolling(window=48, min_periods=24)
        feats[f"origin_mean_24h_h{horizon}"] = r48.mean()
        feats[f"origin_std_24h_h{horizon}"] = r48.std()
        feats[f"origin_min_24h_h{horizon}"] = r48.min()
        feats[f"origin_max_24h_h{horizon}"] = r48.max()

        # 7-day window (336 half-hours) ending at forecast origin t_0
        r336 = origin_load.rolling(window=336, min_periods=96)
        feats[f"origin_mean_7d_h{horizon}"] = r336.mean()
        feats[f"origin_std_7d_h{horizon}"] = r336.std()

    return pd.DataFrame(feats, index=series.index)
