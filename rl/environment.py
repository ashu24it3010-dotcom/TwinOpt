import gymnasium as gym
from gymnasium import spaces
import numpy as np
from digital_twin.factory import FactoryLine

class SmartFactoryEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.factory = FactoryLine()

        # Observation: [M1_temp, M1_vib, M1_health, M2_temp, M2_vib, M2_health, M3_temp, M3_vib, M3_health, total_power, step]
        self.observation_space = spaces.Box(low=0.0, high=500.0, shape=(11,), dtype=np.float32)

        # Action: Discrete choices for M2 speed multiplier -> [0: 0.5x, 1: 0.8x, 2: 1.0x, 3: 1.2x, 4: 0.0x]
        self.action_space = spaces.Discrete(5)
        self.speed_map = {0: 0.5, 1: 0.8, 2: 1.0, 3: 1.2, 4: 0.0}

    def _get_obs(self, state_info: dict) -> np.ndarray:
        m1, m2, m3 = state_info["M1"], state_info["M2"], state_info["M3"]
        return np.array([
            m1["temperature"], m1["vibration"], m1["health"],
            m2["temperature"], m2["vibration"], m2["health"],
            m3["temperature"], m3["vibration"], m3["health"],
            state_info["total_power_kw"],
            float(state_info["step"])
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state_info = self.factory.reset()
        obs = self._get_obs(state_info)
        return obs, {}

    def step(self, action: int):
        m2_speed = self.speed_map[action]
        state_info = self.factory.step({"M1_speed": 1.0, "M2_speed": m2_speed, "M3_speed": 1.0})

        obs = self._get_obs(state_info)

        # Reward formulation: Reward production, penalize energy and machine stress
        prod_reward = state_info["total_finished_goods"] * 2.0
        energy_penalty = state_info["total_power_kw"] * 0.5
        stress_penalty = 0.0

        if state_info["M2"]["temperature"] > 85.0:
            stress_penalty += (state_info["M2"]["temperature"] - 85.0) * 1.5
        if state_info["M2"]["vibration"] > 0.5:
            stress_penalty += (state_info["M2"]["vibration"] - 0.5) * 10.0

        reward = prod_reward - energy_penalty - stress_penalty
        terminated = self.factory.time_step >= 100
        truncated = False

        return obs, float(reward), terminated, truncated, state_info