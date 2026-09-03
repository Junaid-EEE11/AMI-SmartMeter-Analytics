"""
B2: Histogram-Based Gradient-Boosted Decision Trees Model (HistGBR).
"""
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


class HistGBRModel:
    """
    CPU-optimized Gradient Boosted Decision Trees point forecaster.
    Uses sklearn's HistGradientBoostingRegressor.
    """
    def __init__(
        self,
        max_iter: int = 150,
        max_depth: Optional[int] = 6,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 20,
        random_state: int = 42,
        name: str = "HistGradientBoosting"
    ):
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.name = name
        self.feature_names = []

        self.model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HistGBRModel":
        self.feature_names = list(X.columns)
        self.model.fit(X.values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X[self.feature_names].values)
        return np.maximum(0.0, preds)
