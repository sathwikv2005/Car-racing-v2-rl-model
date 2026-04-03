from stable_baselines3 import DQN
from dqn_env import make_env


from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import torch


def main():
    print("GPU:", torch.cuda.is_available())
    print("TRAINING EDGE-DQN MODEL...")

    env = DummyVecEnv([make_env(render_mode=None)])
    env = VecFrameStack(env, n_stack=4)

    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/dqn/",
        device="cuda",

        learning_rate=1e-4,          # ↓ more stable
        buffer_size=50_000,        # ↑ more experience
        batch_size=128,             # ↑ better gradients
        target_update_interval=5000, # ↑ more stable Q updates

        learning_starts=5000,
        

        gamma=0.99,

        train_freq=4,
        gradient_steps=1,


        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        # exploration_final_eps=0.1,

        max_grad_norm=10,
    )

    model.learn(total_timesteps=500_000)

    model.save("../model/dqn_carracing")


if __name__ == "__main__":
    main()