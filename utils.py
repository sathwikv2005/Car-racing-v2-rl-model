import numpy as np

def custom_reward(obs, reward, action):
    """
    obs: (96, 96, 3) image
    reward: original env reward
    action: [steer, gas, brake]
    """

    steer, gas, brake = action

    if isinstance(obs, tuple):
        obs = obs[0]

    road_pixels = obs[:, :, 1]  # extracts green channel (road is green)
    road_ratio = np.mean(road_pixels > 100)

    shaped_reward = reward

    if road_ratio < 0.1:
        shaped_reward -= 2.0  
    else:
        shaped_reward += 0.5  

    shaped_reward += gas * 0.5

    shaped_reward -= brake * 0.3

    shaped_reward -= abs(steer) * 0.2

    # Small alive bonus
    shaped_reward += 0.05

    return shaped_reward