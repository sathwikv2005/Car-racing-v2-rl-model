from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from env import get_env
import torch


def main():
    print("GPU:", torch.cuda.is_available())

    env = get_env(n_envs=8)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        device="cuda",

        n_steps=512,
        batch_size=256,
        n_epochs=10,

        learning_rate=2e-4,      
        ent_coef=0.003,          

        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,

        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./models/",
        name_prefix="ppo_carracing"
    )

    model.learn(
        total_timesteps=1_000_000,
        callback=checkpoint_callback
    )

    model.save("model/ppo_carracing_custom_reward")


if __name__ == "__main__":
    main()