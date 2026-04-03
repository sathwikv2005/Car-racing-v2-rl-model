import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage, VecNormalize
from stable_baselines3.common.monitor import Monitor
from utils import custom_reward


class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_action = np.array([0.0, 0.0, 0.0])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_action = np.array([0.0, 0.0, 0.0])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Apply custom reward with previous action
        shaped_reward = custom_reward(obs, reward, action, self.prev_action)

        # Update previous action
        self.prev_action = action

        return obs, shaped_reward, terminated, truncated, info


def make_env(rank, render_mode=None):
    def _init():
        env = gym.make("CarRacing-v2", render_mode=render_mode)

        # Apply reward wrapper
        env = RewardWrapper(env)

        # Monitor for logging
        env = Monitor(env)

        return env

    return _init


def get_env(n_envs=4, render_mode=None, normalize=True):
    env = SubprocVecEnv([make_env(i, render_mode) for i in range(n_envs)])

    env = VecFrameStack(env, n_stack=4)
    
    env = VecTransposeImage(env)

    if normalize:
        env = VecNormalize(env, norm_obs=False, norm_reward=True)

    return env