import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CarRacing-v2", render_mode="human")

model = PPO.load("model/ppo_carracing")

obs, _ = env.reset()

while True:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()