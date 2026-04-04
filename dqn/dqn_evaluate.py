import time
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from dqn_env import make_env


def main():
    env = DummyVecEnv([make_env(render_mode="human")])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    # model = DQN.load("../model/dqn_carracing", env=env)
    model = DQN.load("../model/dqn/dqn_carracing_final", env=env)

    obs = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        time.sleep(0.01)

        if dones[0]:
            obs = env.reset()


if __name__ == "__main__":
    main()