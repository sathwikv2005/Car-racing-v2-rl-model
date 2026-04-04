from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from dqn_env import make_env
import torch
import os

os.makedirs("../model/dqn/", exist_ok=True)
os.makedirs("../model/dqn/best/", exist_ok=True)
os.makedirs("../model/dqn/checkpoints/", exist_ok=True)
print("Saving to:", os.path.abspath("../model/dqn/"))

def main():
    print("GPU:", torch.cuda.is_available())

    env = DummyVecEnv([make_env()])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    eval_env = DummyVecEnv([make_env()])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="../model/dqn/best/",
        log_path="../logs/eval/",
        eval_freq=50_000,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="../model/dqn/checkpoints/",
        name_prefix="dqn_carracing"
    )

    model = DQN(
        "CnnPolicy",
        env,
        device="cuda",
        verbose=1,
        tensorboard_log="./logs/dqn/",

        learning_rate=1e-4,
        buffer_size=30_000,
        batch_size=64,
        learning_starts=10_000,

        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=5000,

        exploration_fraction=0.2,
        exploration_final_eps=0.05,

        policy_kwargs=dict(net_arch=[256, 256]),
    )

    model.learn(
        total_timesteps=1_000_000,
        callback=CallbackList([eval_callback, checkpoint_callback]),
    )

    model.save("../model/dqn/41_dqn_carracing_final")


if __name__ == "__main__":
    main()