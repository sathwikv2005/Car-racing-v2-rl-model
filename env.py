import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

def make_env(rank, render_mode=None):
    def _init():
        env = gym.make("CarRacing-v2", render_mode=render_mode)
        env = Monitor(env)
        return env
    return _init

def get_env(n_envs=4, render_mode=None):
    env = SubprocVecEnv([make_env(i, render_mode) for i in range(n_envs)])
    env = VecFrameStack(env, n_stack=4)
    return env