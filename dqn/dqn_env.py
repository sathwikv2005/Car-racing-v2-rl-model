import gymnasium as gym
import numpy as np

from gymnasium.wrappers import ResizeObservation
from stable_baselines3.common.monitor import Monitor

from dqn_utils import RewardWrapper


# ✅ FRAME SKIP FIRST
class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=3):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0

        for _ in range(self.skip + 1):
            obs, reward, terminated, truncated, info = self.env.step(action)

            if isinstance(reward, tuple):  # safety
                reward = reward[0]

            total_reward += reward

            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


# ✅ DISCRETE ACTIONS (CRUCIAL FOR DQN)
class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.actions = [
            [0.0, 0.0, 0.0],     # no-op
            [-1.0, 0.0, 0.0],    # left
            [1.0, 0.0, 0.0],     # right
            [0.0, 1.0, 0.0],     # gas
            [0.0, 0.0, 0.8],     # brake
            [-1.0, 1.0, 0.0],    # left + gas
            [1.0, 1.0, 0.0],     # right + gas
            [-0.5, 1.0, 0.0],    # soft left
            [0.5, 1.0, 0.0],     # soft right
        ]

        self.actions = [np.array(a, dtype=np.float32) for a in self.actions]
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action):
        return self.actions[action]


# ✅ MAIN ENV
class CarRacingEnv(gym.Wrapper):
    def __init__(self, render_mode=None):
        env = gym.make("CarRacing-v2", render_mode=render_mode)
        super().__init__(env)

        self.reward_wrapper = RewardWrapper()
        self.prev_action = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reward_wrapper.reset()
        self.prev_action = np.zeros(3, dtype=np.float32)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # ✅ FIX: unpack reward properly
        reward, reward_info = self.reward_wrapper.custom_reward(
            obs, reward, action, self.prev_action
        )

        info["reward"] = reward_info

        self.prev_action = action

        return obs, reward, terminated, truncated, info


# ✅ MAKE ENV (CORRECT ORDER)
def make_env(render_mode=None):
    def _init():
        env = CarRacingEnv(render_mode=render_mode)

        env = FrameSkipWrapper(env, skip=3)
        env = DiscreteActionWrapper(env)

        env = ResizeObservation(env, (64, 64))

        env = Monitor(env)

        return env

    return _init