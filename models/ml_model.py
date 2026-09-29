import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple

class HealthPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False

    def train(self, X_train: pd.DataFrame=None, y_train: pd.Series= None) -> None:
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            # Fallback heuristic if not yet fit
            return np.ones(len(X))
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, 1.0)