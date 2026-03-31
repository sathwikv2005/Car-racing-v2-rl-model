import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env(render_mode=None):
    def _init():
        return gym.make("CarRacing-v2", render_mode=render_mode)
    return _init

def get_env(render_mode=None):
    return DummyVecEnv([make_env(render_mode)])