import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from utils import custom_reward


class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = custom_reward(obs, reward, action)
        return obs, shaped_reward, terminated, truncated, info


def make_env(rank, render_mode=None):
    def _init():
        env = gym.make("CarRacing-v2", render_mode=render_mode)
        env = RewardWrapper(env)
        env = Monitor(env)

        return env

    return _init


def get_env(n_envs=4, render_mode=None):
    env = SubprocVecEnv([make_env(i, render_mode) for i in range(n_envs)])
    env = VecFrameStack(env, n_stack=4)
    return env