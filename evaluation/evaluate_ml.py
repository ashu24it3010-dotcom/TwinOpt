import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from models.ml_model import HealthPredictor

def evaluate_ml_models():
    csv_path = "data/machine_1.csv"
    if not os.path.exists(csv_path):
        from data.generate_data import generate_machine_dataset
        df = generate_machine_dataset("M1", "Cutting", 1500.0, 65.0, 0.25)
    else:
        df = pd.read_csv(csv_path)

    feature_cols = ["temperature", "vibration", "power", "rpm", "operating_hours"]
    X = df[feature_cols]
    y = df["health"]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    rf = HealthPredictor()
    rf.train(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    results = [
        {
            "Model": "Random Forest Regressor",
            "MAE": round(float(mean_absolute_error(y_test, y_pred_rf)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_pred_rf))), 4),
            "R2_Score": round(float(r2_score(y_test, y_pred_rf)), 4)
        },
        {
            "Model": "Linear Regression",
            "MAE": round(float(mean_absolute_error(y_test, y_pred_lr)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_pred_lr))), 4),
            "R2_Score": round(float(r2_score(y_test, y_pred_lr)), 4)
        }
    ]

    return pd.DataFrame(results), y_test.values, y_pred_rf