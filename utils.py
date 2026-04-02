import numpy as np

def custom_reward(obs, reward, action, prev_action):
    """
    obs: (96, 96, 3) image
    reward: original env reward
    action: [steer, gas, brake]
    prev_action: previous action [steer, gas, brake]
    """

    steer, gas, brake = action
    prev_steer = prev_action[0]

    if isinstance(obs, tuple):
        obs = obs[0]

    road_pixels = obs[:, :, 1]
    road_ratio = np.mean(road_pixels > 100)

    shaped_reward = reward * 1.0   

    if road_ratio < 0.1:
        shaped_reward -= 3.0
    else:
        shaped_reward += 0.3

    shaped_reward -= abs(steer) * 0.15

    steer_change = abs(steer - prev_steer)
    shaped_reward -= steer_change * 0.4

    if gas > 0.3 and abs(steer) < 0.4:
        shaped_reward += 0.3

    if brake > 0.5 and gas > 0.3:
        shaped_reward -= 0.7 
    else:
        shaped_reward -= brake * 0.05 

    
    if abs(steer) > 0.8 and gas > 0.5:
        shaped_reward -= 1.2

    # SMALL ALIVE BONUS
    shaped_reward += 0.01

    return shaped_reward