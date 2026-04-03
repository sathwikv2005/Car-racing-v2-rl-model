import numpy as np


class RewardWrapper:
    def __init__(self):
        self.prev_position = None
        self.prev_direction = None
        self.no_progress_steps = 0
        self.offroad_steps = 0
        self.prev_steer = 0
        self.prev_obs = None
        

    def reset(self):
        self.prev_position = None
        self.prev_direction = None
        self.no_progress_steps = 0
        self.offroad_steps = 0
        self.prev_steer = 0
        self.prev_obs = None

    def custom_reward(self, obs, reward, action, prev_action):
        steer, gas, brake = action
        prev_steer = prev_action[0]

        if isinstance(obs, tuple):
            obs = obs[0]

        reward_components = {}

        # --- ROAD DETECTION ---
        r = obs[:, :, 0]
        g = obs[:, :, 1]
        b = obs[:, :, 2]

        # grass is green-dominant
        grass_mask = (g > 150) & (g > r + 20) & (g > b + 20)

        road_mask = ~grass_mask
        road_ratio = np.mean(road_mask)

        # --- BASE ---
        shaped_reward = reward * 1.0
        reward_components["base"] = shaped_reward

        # --- OFFROAD ---
        offroad_penalty = 0
        onroad_bonus = 0

        if road_ratio < 0.1:
            self.offroad_steps += 1
            offroad_penalty = -min(3.0, self.offroad_steps * 0.2)
            shaped_reward += offroad_penalty
        else:
            self.offroad_steps = 0
            onroad_bonus = 0.2
            shaped_reward += onroad_bonus

        reward_components["offroad_penalty"] = offroad_penalty
        reward_components["onroad_bonus"] = onroad_bonus

        # --- STEERING ---
        steer_change = abs(steer - prev_steer)

        steer_change_penalty = -steer_change * 0.2
        steer_penalty = -abs(steer) * 0.04

        shaped_reward += steer_change_penalty + steer_penalty

        reward_components["steer_change"] = steer_change_penalty
        reward_components["steer"] = steer_penalty

        # --- FORWARD ---
        forward_reward = 0
        if gas > 0.3 and abs(steer) < 0.2:
            forward_reward = 0.15
            shaped_reward += forward_reward

        reward_components["forward"] = forward_reward

        # --- BRAKE ---
        brake_penalty = 0
        if brake > 0.5 and gas > 0.3:
            brake_penalty = -0.5
        else:
            brake_penalty = -brake * 0.05

        shaped_reward += brake_penalty
        reward_components["brake"] = brake_penalty

        # --- TURN CONTROL ---
        turn_reward = 0
        if abs(steer) > 0.6:
            if gas > 0.6:
                turn_reward = -0.6
            else:
                turn_reward = 0.05

            shaped_reward += turn_reward

        reward_components["turn"] = turn_reward

        # --- PROGRESS ---
        progress_reward = 0

        if self.prev_obs is not None:
            diff = np.mean(
                np.abs(obs.astype(np.float32) - self.prev_obs.astype(np.float32))
            )

            progress_reward = diff * 0.1
            shaped_reward += progress_reward

            if diff < 0.02:
                self.no_progress_steps += 1
            else:
                self.no_progress_steps = 0

        reward_components["progress"] = progress_reward

        # --- STAGNATION ---
        stagnation_penalty = 0
        if self.no_progress_steps > 30:
            stagnation_penalty = -0.5
            shaped_reward += stagnation_penalty

        reward_components["stagnation"] = stagnation_penalty

        self.prev_obs = obs

        reward_components["total"] = shaped_reward

        return shaped_reward, reward_components