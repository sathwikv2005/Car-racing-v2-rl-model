import numpy as np


class RewardWrapper:
    def __init__(self):
        self.no_progress_steps = 0
        self.offroad_steps = 0
        self.prev_obs = None

    def reset(self):
        self.no_progress_steps = 0
        self.offroad_steps = 0
        self.prev_obs = None

    def custom_reward(self, obs, reward, action, prev_action):
        steer, gas, brake = action

        if isinstance(obs, tuple):
            obs = obs[0]

        # --- ROAD DETECTION ---
        r, g, b = obs[:, :, 0], obs[:, :, 1], obs[:, :, 2]
        grass = (g > 150) & (g > r + 20) & (g > b + 20)
        road = ~grass
        road_ratio = np.mean(road)

        shaped = float(reward)

        # --- OFFROAD ---
        if road_ratio < 0.1:
            self.offroad_steps += 1
            shaped -= min(2.0, self.offroad_steps * 0.1)
        else:
            self.offroad_steps = 0
            shaped += 0.1

        # --- STEERING SMOOTHNESS ---
        shaped -= abs(steer) * 0.03

        # --- FORWARD DRIVING ---
        if gas > 0.3 and abs(steer) < 0.3:
            shaped += 0.1

        # --- BRAKE PENALTY ---
        shaped -= brake * 0.05

        # --- PROGRESS (CLIPPED — VERY IMPORTANT) ---
        if self.prev_obs is not None:
            diff = np.mean(
                np.abs(obs.astype(np.float32) - self.prev_obs.astype(np.float32))
            )
            progress = np.clip(diff * 0.05, 0, 0.2)
            shaped += progress

            if diff < 0.02:
                self.no_progress_steps += 1
            else:
                self.no_progress_steps = 0

        # --- STAGNATION ---
        if self.no_progress_steps > 40:
            shaped -= 0.3

        self.prev_obs = obs

        return shaped, {}