import numpy as np

def custom_reward(obs, reward, action):
    if isinstance(obs, tuple):
        obs = obs[0]

    if obs.ndim == 2:
        obs = np.expand_dims(obs, axis=-1)

    road = obs[:, :, 0]
    steer, gas, brake = action

    road_mask = road > 0.1
    road_ratio = np.mean(road_mask)

    # softer off-road penalty (early training)
    if road_ratio < 0.02:
        return -3.0

    shaped_reward = 0.0

    #  MAIN SIGNAL (DO NOT SCALE DOWN TOO MUCH)
    shaped_reward += reward

    #  weak shaping only
    shaped_reward += (road_ratio - 0.5)

    #  mild steering alignment
    h, w = road.shape
    look = road_mask[:h // 2, :]

    if np.sum(look) > 0:
        x = np.where(look)[1]
        center = np.mean(x) / w
    else:
        center = 0.5

    direction_error = center - 0.5

    shaped_reward += -abs(direction_error - steer)

    #  small encouragement
    shaped_reward += gas * 0.1

    return np.clip(shaped_reward, -5, 5)