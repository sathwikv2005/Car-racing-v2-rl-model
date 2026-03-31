from stable_baselines3 import PPO
from env import get_env
import torch

def main():
    print("GPU:", torch.cuda.is_available())

    env = get_env(n_envs=4)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        device="cuda",

        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=500_000)
    model.save("model/ppo_carracing")


if __name__ == "__main__":
    main()