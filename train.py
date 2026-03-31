from stable_baselines3 import PPO
from env import get_env

env = get_env()

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/"
)

model.learn(total_timesteps=100_000)

model.save("model/ppo_carracing")