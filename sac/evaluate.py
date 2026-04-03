import time
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from env import make_env
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = "../model/sac/best/best_model.zip"
# MODEL_PATH = "../model/sac/sac_carracing"
MODEL_PATH = "../model/sac/checkpoint/sac_checkpoint_350000_steps"


def main():
    env = DummyVecEnv([lambda: make_env(render_mode="human")])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env) 

    data = np.load("../logs/sac/evaluations.npz")
    print(data["timesteps"])
    print(data["results"])


    # timesteps = data["timesteps"]
    # mean_rewards = data["results"].mean(axis=1)

    # plt.plot(timesteps, mean_rewards)
    # plt.xlabel("Timesteps")
    # plt.ylabel("Mean Reward")
    # plt.title("SAC Training Progress")
    # plt.show()

    model = SAC.load(MODEL_PATH)

    obs = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        if dones[0]:
            obs = env.reset()

            time.sleep(0.03)


if __name__ == "__main__":
    main()