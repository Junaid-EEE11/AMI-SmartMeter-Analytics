"""
B0: Seasonal Naive Baseline Models (Day Lag-48 and Week Lag-336).
"""
from typing import Optional, Union
import numpy as np
import pandas as pd


class SeasonalNaiveModel:
    """
    Seasonal Naive baseline forecaster.
    - If season_lag = 48: Predicts same half-hour from previous day.
    - If season_lag = 336: Predicts same half-hour from previous week.
    """
    def __init__(self, season_lag: int = 48, name: Optional[str] = None):
        self.season_lag = season_lag
        self.name = name or (f"SeasonalNaive_Lag{season_lag}")
        self.lag_col_name = f"load_lag_{season_lag}"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalNaiveModel":
        """Seasonal naive is non-parametric / parameter-free."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return the corresponding seasonal lag column from features.
        """
        if self.lag_col_name in X.columns:
            preds = X[self.lag_col_name].values
        elif "load_lag_48" in X.columns:
            # Fallback if specific lag not present
            preds = X["load_lag_48"].values
        else:
            raise ValueError(f"Feature column '{self.lag_col_name}' not found in X.")
        
        # Ensure non-negative demand
        return np.maximum(0.0, np.nan_to_num(preds, nan=0.0))
