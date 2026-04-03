import gymnasium as gym
import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from gymnasium.wrappers import ResizeObservation, GrayScaleObservation

from dqn_utils import custom_reward


# ✅ MINIMAL ACTION SPACE + SMOOTHING
class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.actions = [
            [0.0, 1.0, 0.0],   # straight
            [-0.5, 0.8, 0.0],  # left
            [0.5, 0.8, 0.0],   # right
        ]

        self.prev_action = 0
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action):
        # Handle VecEnv output
        if isinstance(action, (list, np.ndarray)):
            action = int(action[0])

        # 🔥 Prevent instant LEFT ↔ RIGHT flip (causes spinning)
        if abs(action - self.prev_action) == 2:
            action = self.prev_action

        self.prev_action = action

        return self.actions[action]


# ✅ REWARD WRAPPER (FIXED)
class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        if isinstance(action, (list, np.ndarray)):
            action = int(action[0])

        real_action = self.env.actions[action]

        obs, reward, terminated, truncated, info = self.env.step(real_action)

        shaped_reward = custom_reward(obs, reward, real_action)

        return obs, shaped_reward, terminated, truncated, info


def make_env(render_mode=None):
    def _init():
        env = gym.make("CarRacing-v2", render_mode=render_mode)

        env = ResizeObservation(env, (84, 84))
        env = GrayScaleObservation(env, keep_dim=True)

        env = DiscreteActionWrapper(env)
        env = RewardWrapper(env)
        env = Monitor(env)

        return env

    return _init


def get_env(render_mode=None):
    env = DummyVecEnv([make_env(render_mode)])
    env = VecFrameStack(env, n_stack=4)
    return env