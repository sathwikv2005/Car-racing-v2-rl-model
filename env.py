import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def make_env(render_mode=None):
    def _init():
        env = gym.make("CarRacing-v2", render_mode=render_mode)
        env = Monitor(env)
        return env
    return _init

def get_env(render_mode=None):
    return DummyVecEnv([make_env(render_mode)])