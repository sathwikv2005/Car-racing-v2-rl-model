import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from utils import custom_reward


# Discretize actions
class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        # Define discrete actions
        self.actions = [
            np.array([0.0, 0.0, 0.0]),   # no-op
            np.array([-1.0, 0.0, 0.0]),  # left
            np.array([1.0, 0.0, 0.0]),   # right
            np.array([0.0, 1.0, 0.0]),   # accelerate
            np.array([0.0, 0.0, 0.8]),   # brake
        ]

        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action):
        return self.actions[action]


class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = custom_reward(obs, reward, action)
        return obs, reward, terminated, truncated, info


def make_env():
    def _init():
        env = gym.make("CarRacing-v2")
        env = DiscreteActionWrapper(env)
        env = RewardWrapper(env)
        env = Monitor(env)
        return env

    return _init


def get_env():
    env = DummyVecEnv([make_env()])
    env = VecFrameStack(env, n_stack=4)
    return env