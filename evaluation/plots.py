import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

def plot_ml_predictions(y_true, y_pred, title="Actual vs Predicted Machine Health"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred, alpha=0.5, color='royalblue')
    ax.plot([0, 1], [0, 1], 'r--', label="Ideal Line")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_rl_learning_curve(rewards_history):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(rewards_history, marker='o', color='darkgreen', linewidth=2)
    ax.set_title("RL Agent Learning Curve (Reward vs Episodes)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def plot_policy_comparison_plotly(df_benchmark: pd.DataFrame):
    fig = px.bar(
        df_benchmark, 
        x="Policy", 
        y=["Production (Units)", "Energy (kWh)", "Total Reward"], 
        barmode="group",
        title="Factory Control Policy Comparison",
        labels={"value": "Value", "variable": "Metric"}
    )
    return fig