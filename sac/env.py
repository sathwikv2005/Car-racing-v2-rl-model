import gymnasium as gym
import numpy as np

from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.monitor import Monitor

from utils import RewardWrapper


class CarRacingEnv(gym.Wrapper):
    def __init__(self, render_mode=None):
        env = gym.make("CarRacing-v2", render_mode=render_mode)
        super().__init__(env)

        self.reward_wrapper = RewardWrapper()
        self.prev_action = None
        self.prev_tile_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.reward_wrapper.reset()
        self.prev_action = np.array([0.0, 0.0, 0.0])
        self.prev_tile_count = 0

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        car = self.env.unwrapped.car

        if car is not None:
            vel = car.hull.linearVelocity
            speed = float(np.linalg.norm([vel[0], vel[1]]))
        else:
            speed = 0.0

        # --- TILE PROGRESS ---
        tile_count = self.env.unwrapped.tile_visited_count
        tile_progress = tile_count - self.prev_tile_count
        self.prev_tile_count = tile_count

        state_info = {
            "speed": speed,
            "tile_progress": tile_progress,
        }

        # --- APPLY CUSTOM REWARD ---
        shaped_reward, reward_info = self.reward_wrapper.custom_reward(
            obs,
            reward,
            action,
            self.prev_action,
            state_info,
        )

        info["reward_components"] = reward_info

        self.prev_action = action

        return obs, shaped_reward, terminated, truncated, info
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

    env = WarpFrame(env)  # grayscale + resize

    env = Monitor(env)

    return env