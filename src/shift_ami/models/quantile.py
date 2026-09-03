"""
B3 & P0: Multi-Quantile Gradient Boosted Regression Models.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


class QuantileGBRModel:
    """
    Simultaneous multi-quantile regression model using HistGradientBoostingRegressor
    with pinball / quantile loss.
    """
    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        max_iter: int = 150,
        max_depth: Optional[int] = 6,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 20,
        random_state: int = 42,
        name: str = "QuantileGBR"
    ):
        self.quantiles = quantiles or [0.05, 0.50, 0.95]
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.name = name
        self.models: Dict[float, HistGradientBoostingRegressor] = {}
        self.feature_names = []

        for q in self.quantiles:
            self.models[q] = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_iter=max_iter,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileGBRModel":
        self.feature_names = list(X.columns)
        X_vals = X.values
        y_vals = y.values
        for q, model in self.models.items():
            model.fit(X_vals, y_vals)
        return self

    def predict_quantile(self, X: pd.DataFrame, quantile: float) -> np.ndarray:
        if quantile not in self.models:
            raise ValueError(f"Quantile {quantile} was not fitted. Available: {list(self.models.keys())}")
        preds = self.models[quantile].predict(X[self.feature_names].values)
        return np.maximum(0.0, preds)

    def predict_all(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        """Predict all configured quantiles."""
        return {q: self.predict_quantile(X, q) for q in self.quantiles}

    def predict_interval(
        self,
        X: pd.DataFrame,
        alpha: float = 0.10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (lower, median, upper) prediction arrays for nominal miscoverage alpha.
        """
        q_lo = alpha / 2.0
        q_hi = 1.0 - alpha / 2.0
        q_med = 0.50

        # Find closest fitted quantiles if exact match not present
        actual_lo = min(self.quantiles, key=lambda q: abs(q - q_lo))
        actual_hi = min(self.quantiles, key=lambda q: abs(q - q_hi))
        actual_med = min(self.quantiles, key=lambda q: abs(q - q_med))

        lower = self.predict_quantile(X, actual_lo)
        upper = self.predict_quantile(X, actual_hi)
        median = self.predict_quantile(X, actual_med)

        # Enforce quantile monotonicity: lower <= upper
        upper = np.maximum(lower, upper)
        return lower, median, upper
