import gymnasium as gym
import numpy as np

from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.monitor import Monitor

from utils import RewardWrapper


class CarRacingEnv(gym.Wrapper):
    def __init__(self, render_mode=None):
        env = gym.make("CarRacing-v2", render_mode=render_mode)
        super().__init__(env)
        self.prev_action = np.array([0.0, 0.0, 0.0])

        self.reward_wrapper = RewardWrapper()
        self.prev_action = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reward_wrapper.reset()
        self.prev_action = np.zeros(self.action_space.shape)
        self.prev_action = np.array([0.0, 0.0, 0.0])
        return obs, info

    def step(self, action):
        
        

        obs, reward, terminated, truncated, info = self.env.step(action)

        reward, reward_components = self.reward_wrapper.custom_reward(
            obs, reward, action, self.prev_action
        )
        info["reward_components"] = reward_components
        
        # smooth prev action
        # if self.prev_action is not None:
        #     action = 0.7 * self.prev_action + 0.3 * action

        # Update previous action
        self.prev_action = action

        return obs, reward, terminated, truncated, info
    
class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0
        terminated = False
        truncated = False

        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


def make_env(render_mode=None):
    env = CarRacingEnv(render_mode=render_mode)

    env = FrameSkipWrapper(env, skip=3)

    env = Monitor(env)
    env = WarpFrame(env)  # grayscale + resize

    return env