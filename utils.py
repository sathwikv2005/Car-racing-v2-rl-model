import numpy as np

def custom_reward(obs, reward, action, prev_action):
    """
    obs: (96, 96, 3) image from environment
    reward: original environment reward (progress-based)
    action: [steer, gas, brake]
    prev_action: previous action [steer, gas, brake]
    """

    steer, gas, brake = action
    prev_steer = prev_action[0]

    if isinstance(obs, tuple):
        obs = obs[0]

    # --- ROAD DETECTION ---
    # Green channel roughly corresponds to road pixels
    road_pixels = obs[:, :, 1]

    # Calculate fraction of pixels that look like road
    road_ratio = np.mean(road_pixels > 100)

    # --- BASE REWARD ---
    # Keep original environment reward as main learning signal
    shaped_reward = reward * 1.0   

    # --- ON-ROAD / OFF-ROAD ---
    # Penalize heavily if car is likely off-road
    if road_ratio < 0.1:
        shaped_reward -= 3.0
    else:
        # Small reward for staying on track
        shaped_reward += 0.15

    # --- STEERING PENALTY ---
    # Penalize large steering angles (encourage straight driving)
    shaped_reward -= abs(steer) * 0.08

    # Penalize sudden steering changes (encourage smooth control)
    steer_change = abs(steer - prev_steer)
    shaped_reward -= steer_change * 0.2

    # --- FORWARD MOVEMENT ---
    # Reward moving forward with stable steering
    if gas > 0.3 and abs(steer) < 0.2:
        shaped_reward += 0.15

    # --- BRAKE CONTROL ---
    # Penalize braking while accelerating (bad driving behavior)
    if brake > 0.5 and gas > 0.3:
        shaped_reward -= 0.5 
    else:
        # Small penalty for unnecessary braking
        shaped_reward -= brake * 0.05 

    # --- SHARP TURN PENALTY ---
    # Strong penalty for high-speed sharp turns (causes spin-outs)
    if abs(steer) > 0.8 and gas > 0.5:
        shaped_reward -= 0.7

    # --- ALIVE BONUS ---
    # Small reward each step to encourage staying alive
    #shaped_reward += 0.01

    return shaped_reward