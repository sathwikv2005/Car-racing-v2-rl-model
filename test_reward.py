import gymnasium as gym
from utils import custom_reward

env = gym.make("CarRacing-v2")

obs, _ = env.reset()

for _ in range(20):
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, _ = env.step(action)

    shaped = custom_reward(next_obs, reward, action)

    print(f"Original: {reward:.3f} | Shaped: {shaped:.3f}")

    if terminated or truncated:
        obs, _ = env.reset()