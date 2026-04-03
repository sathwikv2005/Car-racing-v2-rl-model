from stable_baselines3 import DQN
from dqn_env import get_env
import torch


def main():
    print("GPU:", torch.cuda.is_available())
    print("TRAINING NEW DQN MODEL...")

    env = get_env()

    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/dqn/",
        device="cuda",

        buffer_size=50_000,
        learning_rate=5e-4,

        learning_starts=5000,
        batch_size=64,

        gamma=0.99,

        train_freq=4,
        gradient_steps=1,

        target_update_interval=1000,

        exploration_fraction=0.6,
        exploration_final_eps=0.1,

        max_grad_norm=10,
    )

    model.learn(total_timesteps=500_000)

    model.save("../model/dqn_carracing")


if __name__ == "__main__":
    main()