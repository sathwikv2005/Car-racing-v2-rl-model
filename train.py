from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
from env import get_env
import torch


def main():
    print("GPU:", torch.cuda.is_available())

    env = get_env(n_envs=8)

    eval_env = get_env(n_envs=1, normalize=False)

    vec_norm = env
    while not isinstance(vec_norm, VecNormalize):
        vec_norm = vec_norm.venv

    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=vec_norm.norm_obs,
        norm_reward=False,
        clip_obs=vec_norm.clip_obs
    )

    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/ppo/",
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
        save_path="./model/ppo/",
        name_prefix="ppo_carracing"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./model/ppo/best/",
        log_path="./logs/ppo/",
        eval_freq=10_000,
        deterministic=True,
        render=False
    )

    model.learn(
        total_timesteps=1_000_000,
        callback=[checkpoint_callback, eval_callback]
    )

    vec_norm = env
    while not isinstance(vec_norm, VecNormalize):
        vec_norm = vec_norm.venv

    vec_norm.save("model/ppo/vecnormalize.pkl")

if __name__ == "__main__":
    main()