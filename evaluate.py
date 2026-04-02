import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from env import make_env 


def main():
    # Create environment (same as training but single env + render)
    env = DummyVecEnv([make_env(0, render_mode="human")])
    env = VecFrameStack(env, n_stack=4)

    model = PPO.load("model/ppo_carracing_custom_reward")

    obs = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        time.sleep(0.03)  # control speed for visualization

        if dones[0]:
            print("Episode finished. Resetting...")
            obs = env.reset()


if __name__ == "__main__":
    main()