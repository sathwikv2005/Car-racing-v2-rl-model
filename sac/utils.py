import numpy as np


class RewardWrapper:
    """
    Custom reward shaping for CarRacing-v2.

    Design principles:
    - Preserve original environment reward (main learning signal)
    - Add dense shaping rewards (road alignment, speed)
    - Strongly reinforce tile progress
    - Apply light control penalties to stabilize behavior
    """

    def __init__(self):
        # Step counter for optional periodic diagnostics
        self.step_count = 0

        # Running statistics (useful for debugging trends)
        self.road_ratios = []
        self.speeds = []
        self.rewards = []

    def reset(self):
        """Reset episode-specific state."""
        self.step_count = 0
        self.road_ratios.clear()
        self.speeds.clear()
        self.rewards.clear()

    def custom_reward(self, obs, reward, action, prev_action, state_info):
        

        self.step_count += 1

        # Unpack action
        steer, gas, brake = action

        # Handle tuple observations (Gymnasium compatibility)
        if isinstance(obs, tuple):
            obs = obs[0]

       
        # Base Reward (preserve original environment signal)
       
        base_reward = float(reward)
        shaped_reward = base_reward

       
        # Road Detection (from pixel colors)
       
        # Separate RGB channels
        r = obs[:, :, 0]
        g = obs[:, :, 1]
        b = obs[:, :, 2]

        # Identify grass vs road
        grass_mask = (g > 120) & (g > r + 10) & (g > b + 10)
        road_mask = ~grass_mask

        # Focus on region ahead of the car
        h, w = road_mask.shape
        roi = road_mask[int(h * 0.6):int(h * 0.95), int(w * 0.3):int(w * 0.7)]

        road_ratio = float(np.mean(roi))

        # Encourage staying on road (dense positive signal)
        road_reward = road_ratio * 0.2
        shaped_reward += road_reward

       
        # Speed Reward (encourage forward motion)
       
        speed = float(state_info["speed"])

        # Smooth reward proportional to speed
        move_reward = speed / 50.0
        shaped_reward += move_reward

       
        # Tile Progress Reward (primary objective)
       
        tile_progress = state_info["tile_progress"]

        if tile_progress > 0:
            tile_reward = tile_progress * 10.0
            shaped_reward += tile_reward
            got_tile = 1
        else:
            tile_reward = 0.0
            got_tile = 0

       
        # Control Penalties (stability constraints)
       
        steer_penalty = abs(steer) * 0.02
        brake_penalty = brake * 0.005

        shaped_reward -= steer_penalty
        shaped_reward -= brake_penalty

       
        # Update Running Statistics (for diagnostics)
       
        self.road_ratios.append(road_ratio)
        self.speeds.append(speed)
        self.rewards.append(shaped_reward)

       
        # Logging Information (structured for analysis)
       
        info = {
            # --- Reward breakdown ---
            "reward/base": base_reward,
            "reward/total": shaped_reward,
            "reward/road": road_reward,
            "reward/move": move_reward,
            "reward/tile": tile_reward,
            "reward/steer_penalty": -steer_penalty,
            "reward/brake_penalty": -brake_penalty,

            # --- State metrics ---
            "state/speed": speed,
            "state/road_ratio": road_ratio,
            "state/tile_progress": tile_progress,

            # --- Behavior indicators ---
            "event/got_tile": got_tile,
            "event/off_road": int(road_ratio < 0.3),
            "event/stopped": int(speed < 0.5),

            # --- Action diagnostics ---
            "action/steer": float(steer),
            "action/gas": float(gas),
            "action/brake": float(brake),
        }

        return shaped_reward, info