import os
from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from digital_twin.factory import FactoryLine
from models.ml_model import HealthPredictor
from models.transformer import TransformerForecaster
from rl.agent import DQNAgent
from agent.industrial_agent import IndustrialCopilot
from evaluation.evaluate_ml import evaluate_ml_models
from evaluation.evaluate_rl import evaluate_rl_agent

app = Flask(__name__)

# Ensure data directory exists and sample telemetry CSV is present
if not os.path.exists("data/machine_1.csv"):
    from data.generate_data import generate_machine_dataset
    os.makedirs("data", exist_ok=True)
    df_m1 = generate_machine_dataset("M1", "Cutting", 1500.0, 65.0, 0.25)
    df_m2 = generate_machine_dataset("M2", "Processing", 1400.0, 68.0, 0.35)
    df_m3 = generate_machine_dataset("M3", "Assembly", 1300.0, 60.0, 0.20)
    df_m1.to_csv("data/machine_1.csv", index=False)
    df_m2.to_csv("data/machine_2.csv", index=False)
    df_m3.to_csv("data/machine_3.csv", index=False)

# Global module instances
factory = FactoryLine()

# Initialize and fit ML model on training dataset
ml_model = HealthPredictor()
df_data = pd.read_csv("data/machine_1.csv")
feature_cols = ["temperature", "vibration", "power", "rpm", "operating_hours"]
X_tr, _, y_tr, _ = train_test_split(df_data[feature_cols], df_data["health"], test_size=0.2, random_state=42)
ml_model.train(X_tr, y_tr)

forecaster = TransformerForecaster()
rl_agent = DQNAgent()
copilot = IndustrialCopilot()
telemetry_history = []

def get_factory_state_payload():
    m1, m2, m3 = factory.m1, factory.m2, factory.m3
    total_power = m1.power + m2.power + m3.power
    avg_health = float(np.mean([m1.health, m2.health, m3.health]) * 100)
    
    return {
        "step": factory.time_step,
        "total_production": m3.produced_units,
        "total_power_kw": round(total_power, 2),
        "avg_health": round(avg_health, 1),
        "status": "RUNNING" if avg_health > 50 else "WARNING",
        "machines": {
            "M1": m1.get_state(),
            "M2": m2.get_state(),
            "M3": m3.get_state(),
        },
        "history": telemetry_history[-10:]
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/state", methods=["GET"])
def get_state():
    return jsonify(get_factory_state_payload())

@app.route("/api/step", methods=["POST"])
def run_step():
    m1, m2, m3 = factory.m1, factory.m2, factory.m3
    total_power = m1.power + m2.power + m3.power
    obs = np.array([
        m1.temperature, m1.vibration, m1.health,
        m2.temperature, m2.vibration, m2.health,
        m3.temperature, m3.vibration, m3.health,
        total_power, float(factory.time_step)
    ], dtype=np.float32)
    
    action = rl_agent.select_action(obs)
    speed_map = {0: 0.5, 1: 0.8, 2: 1.0, 3: 1.2, 4: 0.0}
    
    state_info = factory.step({
        "M1_speed": 1.0,
        "M2_speed": speed_map[action],
        "M3_speed": 1.0
    })
    
    telemetry_history.append({
        "step": state_info["step"],
        "m1_temp": state_info["M1"]["temperature"],
        "m2_temp": state_info["M2"]["temperature"],
        "m3_temp": state_info["M3"]["temperature"],
        "power_kw": state_info["total_power_kw"],
        "production": state_info["total_finished_goods"]
    })
    
    return jsonify(get_factory_state_payload())

@app.route("/api/evaluate-ml", methods=["GET"])
def evaluate_ml():
    df_metrics, _, _ = evaluate_ml_models()
    return jsonify(df_metrics.to_dict(orient="records"))

@app.route("/api/evaluate-rl", methods=["POST"])
def evaluate_rl():
    df_benchmark, rewards_history = evaluate_rl_agent(rl_agent)
    return jsonify({
        "benchmarks": df_benchmark.to_dict(orient="records"),
        "rewards_history": rewards_history
    })

@app.route("/api/transformer-predict", methods=["GET"])
def transformer_predict():
    m2 = factory.m2
    seq = np.array([[m2.temperature + i * 0.2, m2.vibration + i * 0.01] for i in range(6)])
    pred_temp, pred_vib = forecaster.predict_next(seq)
    return jsonify({
        "predicted_temp": round(pred_temp, 2),
        "predicted_vibration": round(pred_vib, 3)
    })

@app.route("/api/copilot", methods=["POST"])
def query_copilot():
    data = request.json or {}
    user_query = data.get("query", "Why did the system adjust Machine 2's speed?")
    m2 = factory.m2
    
    current_state = {
        "M2": m2.get_state(),
        "total_finished_goods": factory.m3.produced_units,
        "total_power_kw": factory.m1.power + m2.power + factory.m3.power
    }
    
    response = copilot.query(
        prompt=user_query,
        current_state=current_state,
        predicted_temp=m2.temperature + 2.5,
        last_action=1
    )
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)