"""
B1: Transparent Linear / Ridge Regression Baseline.
"""
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class RidgeModel:
    """
    Transparent regularized linear baseline model.
    """
    def __init__(self, alpha: float = 1.0, name: str = "Ridge_Linear"):
        self.alpha = alpha
        self.name = name
        self.model = Ridge(alpha=alpha, fit_intercept=True)
        self.feature_names = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeModel":
        self.feature_names = list(X.columns)
        self.model.fit(X.values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X[self.feature_names].values)
        return np.maximum(0.0, preds)
