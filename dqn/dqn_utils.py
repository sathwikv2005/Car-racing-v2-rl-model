import numpy as np


def custom_reward(obs, reward, action):
    if isinstance(obs, tuple):
        obs = obs[0]

    if obs.ndim == 2:
        obs = np.expand_dims(obs, axis=-1)

    road = obs[:, :, 0] / 255.0

    steer, gas, _ = action

    road_mask = road > 0.4
    road_ratio = np.mean(road_mask)

    # ❌ Off-road penalty
    if road_ratio < 0.05:
        return -5.0

    shaped_reward = 0.0

    # 🚗 Always encourage forward movement
    shaped_reward += 1.0

    # 🛣️ Stay on road
    shaped_reward += road_ratio * 2.0

    # 🧭 Look-ahead steering (top half of image)
    h, w = road.shape
    look = road_mask[:h // 2, :]

    if np.sum(look) > 0:
        x = np.where(look)[1]
        center = np.mean(x) / w
    else:
        center = 0.5

    direction_error = center - 0.5

    # 🔥 Steering alignment
    shaped_reward += -abs(direction_error - steer)

    return shaped_reward