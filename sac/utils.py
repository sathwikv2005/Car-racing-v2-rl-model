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

        # --- LOGGING DICT ---
        info = {
            "action": action,
            "base": float(reward),
            "offroad_penalty": 0.0,
            "onroad_bonus": 0.0,
            "steer_penalty": 0.0,
            "steer_change_penalty": 0.0,
            "forward_bonus": 0.0,
            "brake_penalty": 0.0,
            "progress": 0.0,
            "stagnation_penalty": 0.0,
        }

        # --- ROAD DETECTION ---
        r = obs[:, :, 0] 
        g = obs[:, :, 1] 
        b = obs[:, :, 2] 
        # grass is green-dominant 
        grass_mask = (g > 120) & (g > r + 10) & (g > b + 10) 
        road_mask = ~grass_mask 
        h, w = road_mask.shape 
        # Define a small region around the car (bottom center) 
        roi = road_mask[int(h*0.6):int(h*0.9), int(w*0.4):int(w*0.6)] 
        road_ratio = np.mean(roi)

        shaped = float(reward)
        info["road_ratio"] = road_ratio
        # --- OFFROAD ---
        if road_ratio < 0.3:
            self.offroad_steps += 1
            penalty = min(2.0, self.offroad_steps * 0.1)
            shaped -= penalty
            info["offroad_penalty"] = -penalty
        else:
            self.offroad_steps = 0
            shaped += 0.1
            info["onroad_bonus"] = 0.1

        # --- STEERING SMOOTHNESS ---
        steer_penalty = abs(steer) * 0.03
        shaped -= steer_penalty
        info["steer_penalty"] = -steer_penalty

        # --- STEERING CHANGE PENALTY ---
        if prev_action is not None:
            prev_steer = prev_action[0]
            steer_change_penalty = abs(steer - prev_steer) * 0.003
            shaped -= steer_change_penalty
            info["steer_change_penalty"] = -steer_change_penalty

        # --- FORWARD DRIVING ---
        if gas > 0.3 and abs(steer) < 0.3:
            shaped += 0.05
            info["forward_bonus"] = 0.05

        # --- EXTRA OFFROAD CONTROL ---
        if road_ratio < 0.1:
            extra_penalty = abs(steer) * 0.1
            shaped -= extra_penalty
            info["offroad_penalty"] -= extra_penalty  # accumulate

        # --- BRAKE PENALTY ---
        brake_penalty = brake * 0.05
        shaped -= brake_penalty
        info["brake_penalty"] = -brake_penalty

        # --- PROGRESS ---
        if self.prev_obs is not None:
            diff = np.mean(
                np.abs(obs.astype(np.float32) - self.prev_obs.astype(np.float32))
            )
            progress = np.clip(diff * 0.05, 0, 0.2)
            shaped += progress
            info["progress"] = progress

            if diff < 0.02:
                self.no_progress_steps += 1
            else:
                self.no_progress_steps = 0

        # --- STAGNATION ---
        if self.no_progress_steps > 40:
            shaped -= 0.3
            info["stagnation_penalty"] = -0.3

        self.prev_obs = obs

        info["total"] = shaped

        return shaped, info