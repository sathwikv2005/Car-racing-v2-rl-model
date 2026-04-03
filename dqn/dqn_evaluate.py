import time
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from dqn_env import make_env

 
def main():
    print("LOADING MODEL...")

    env = DummyVecEnv([make_env(render_mode="human")])
    env = VecFrameStack(env, n_stack=4)

    model = DQN.load("../model/dqn_carracing")

    obs = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)

        obs, rewards, dones, infos = env.step(action)

        env.render()
        time.sleep(0.03)

        if dones[0]:
            print("Episode finished. Resetting...")
            obs = env.reset()


if __name__ == "__main__":
    main()
