import os
from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import TrainFreq
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from reward_callback import RewardLoggingCallback
from env import make_env

MODEL_DIR = "../model/sac"
BEST_MODEL_DIR = "../model/sac/best"
LOG_DIR = "../logs/sac"
CHECKPOINT_DIR = MODEL_DIR + "/checkpoint"


os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

# --- Parallel environments ---
N_ENVS = 4
N_STACK = 4

import re

def get_next_run_id():
    if not os.path.exists(LOG_DIR):
        return 1

    runs = []
    for name in os.listdir(LOG_DIR):
        match = re.search(r"SAC_(\d+)", name) 
        if match:
            runs.append(int(match.group(1)))

    return max(runs, default=0) + 1


def make_env_fn():
    def _init():
        return make_env()
    return _init


def main():

    train_env = SubprocVecEnv([make_env_fn() for _ in range(N_ENVS)])
    train_env = VecFrameStack(train_env, n_stack=N_STACK)  
    train_env = VecTransposeImage(train_env)

    eval_env = DummyVecEnv([make_env_fn()])
    eval_env = VecFrameStack(eval_env, n_stack=N_STACK)
    eval_env = VecTransposeImage(eval_env) 

    policy_kwargs = dict(
        net_arch=[256, 256],  # stronger policy/value nets
    )

    # model = SAC(
    #     "CnnPolicy",
    #     train_env,
    #     policy_kwargs=policy_kwargs,
    #     verbose=1,


    #     buffer_size=100_000,
    #     batch_size=256,
    #     learning_rate=1e-4,

    #     # multi-env
    #     train_freq=(8, "step"),
    #     gradient_steps=8,  

    #     learning_starts=5_000,

    #     gamma=0.99,
    #     tau=0.005,

    #     # Exploration
    #     ent_coef="auto_0.1",

    #     # Stability
    #     target_update_interval=1,

    #     device="cuda",
    #     tensorboard_log=LOG_DIR,
    # )

    # model.set_parameters("../model/sac/best/88_best_model.zip")

    model = SAC.load(
        "../model/sac/best/88_best_model.zip",
        env=train_env,
        device="cuda",
        tensorboard_log=LOG_DIR,
    )
    
    model.learning_rate = 1e-4
    model.lr_schedule = lambda _: 1e-4



    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=20_000,  
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // N_ENVS, 
        save_path=CHECKPOINT_DIR,
        name_prefix="sac_checkpoint",
        save_replay_buffer=False, 
        save_vecnormalize=False,
    )

    reward_callback = RewardLoggingCallback(log_freq=1000)

    run_id = get_next_run_id()
    run_name = f"{run_id}_sac_carracing"

    print(f'saving at {MODEL_DIR}/{run_name}')

    model.learn(
        total_timesteps=300_000,
        callback=CallbackList([eval_callback, checkpoint_callback, reward_callback]),
        progress_bar=True
    )

    
    model.save(f"{MODEL_DIR}/{run_name}")

    print("Training complete!")


if __name__ == "__main__":
    main()