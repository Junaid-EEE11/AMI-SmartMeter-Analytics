"""
End-to-end feature pipeline and dataset matrix builder with strict training-only scaling.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from shift_ami.features.calendar import extract_calendar_features
from shift_ami.features.lagged_load import construct_lagged_load_features
from shift_ami.utils.logging import setup_logger

logger = setup_logger("shift_ami.features.build")


@dataclass
class ForecastingDataMatrix:
    X: pd.DataFrame
    y: pd.Series
    timestamps: pd.Series
    horizons: pd.Series
    feature_names: List[str]


class FeaturePipeline:
    """
    Leakage-safe feature transformer and scaler.
    Fits scaler exclusively on the training split.
    """
    def __init__(
        self,
        lags: Optional[List[int]] = None,
        include_holidays: bool = True,
        scale_features: bool = True
    ):
        self.lags = lags or [48, 96, 144, 336]
        self.include_holidays = include_holidays
        self.scale_features = scale_features
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False

    def build_raw_features_for_series(
        self,
        df: pd.DataFrame,
        target_col: str = "load_kwh",
        timestamp_col: str = "timestamp",
        forecast_horizon: int = 48
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Construct aligned feature dataframe and target series for day-ahead forecasting.
        
        To support standard day-ahead forecasting origins (e.g. daily at 00:00 producing h=1..48),
        features use maximum day-ahead lag safety (lag >= 48) so that predictions across all 
        horizons h in {1..48} are simultaneously available at origin t_0 = t - h.
        """
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        timestamps = df[timestamp_col]
        y = df[target_col]

        # 1. Calendar features (known for target timestamp)
        df_cal = extract_calendar_features(timestamps, include_holidays=self.include_holidays)

        # 2. Lagged load features (safe for all horizons h <= 48)
        # Using minimum day-ahead lag of 48 half-hours (previous day same half-hour)
        df_lags = construct_lagged_load_features(
            series=y,
            horizon=forecast_horizon,
            lags=self.lags,
            include_rolling_stats=True
        )

        # Combine
        X_raw = pd.concat([df_cal, df_lags], axis=1)

        # Retain clean non-null rows
        valid_mask = ~X_raw.isna().any(axis=1) & ~y.isna()

        return X_raw[valid_mask].reset_index(drop=True), y[valid_mask].reset_index(drop=True), timestamps[valid_mask].reset_index(drop=True)

    def fit(self, X_train: pd.DataFrame) -> "FeaturePipeline":
        """Fit scaler strictly on training split features."""
        self.feature_names = list(X_train.columns)
        if self.scale_features:
            self.scaler.fit(X_train.values)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted training statistics."""
        if not self.is_fitted:
            raise RuntimeError("FeaturePipeline must be fitted on training data before transforming.")
        
        # Ensure column alignment
        X = X[self.feature_names]
        if self.scale_features:
            scaled_vals = self.scaler.transform(X.values)
            return pd.DataFrame(scaled_vals, columns=self.feature_names, index=X.index)
        return X.copy()

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X_train).transform(X_train)


def build_forecasting_dataset(
    df: pd.DataFrame,
    target_col: str = "load_kwh",
    timestamp_col: str = "timestamp",
    forecast_horizon: int = 48,
    lags: Optional[List[int]] = None,
    include_holidays: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Helper function to generate raw unscaled features and targets."""
    pipeline = FeaturePipeline(lags=lags, include_holidays=include_holidays, scale_features=False)
    return pipeline.build_raw_features_for_series(df, target_col, timestamp_col, forecast_horizon)
