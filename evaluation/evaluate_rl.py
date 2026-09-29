import numpy as np
import pandas as pd
from rl.environment import SmartFactoryEnv
from rl.agent import DQNAgent

def evaluate_rl_agent(agent: DQNAgent = None):
    env = SmartFactoryEnv()
    if agent is None:
        agent = DQNAgent()

    def run_policy(policy_type: str, steps: int = 50):
        obs, _ = env.reset()
        total_prod = 0
        total_energy = 0.0
        total_reward = 0.0

        for _ in range(steps):
            if policy_type == "Random":
                action = env.action_space.sample()
            elif policy_type == "Fixed (1.0x)":
                action = 2  # 1.0x speed
            else:
                action = agent.select_action(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            total_prod = info["total_finished_goods"]
            total_energy += info["total_power_kw"]
            total_reward += reward

            if terminated or truncated:
                break

        return total_prod, round(total_energy, 2), round(total_reward, 2)

    p_rand, e_rand, r_rand = run_policy("Random")
    p_fix, e_fix, r_fix = run_policy("Fixed (1.0x)")
    p_rl, e_rl, r_rl = run_policy("RL Policy")

    benchmarks = [
        {"Policy": "Random Control", "Production (Units)": p_rand, "Energy (kWh)": e_rand, "Cumulative Reward": r_rand},
        {"Policy": "Fixed Rule (1.0x)", "Production (Units)": p_fix, "Energy (kWh)": e_fix, "Cumulative Reward": r_fix},
        {"Policy": "DQN Agent (Ours)", "Production (Units)": p_rl, "Energy (kWh)": e_rl, "Cumulative Reward": r_rl}
    ]

    rewards_history = [r_rand, r_fix, r_rl]
    return pd.DataFrame(benchmarks), rewards_history