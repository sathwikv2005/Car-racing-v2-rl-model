import gymnasium as gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

def make_env():
    return gym.make("CarRacing-v2", render_mode="human")

def main():
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)

    model = PPO.load("model/ppo_carracing")

    obs = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        time.sleep(0.03)

        if dones[0]: 
            obs = env.reset()

if __name__ == "__main__":
    main()