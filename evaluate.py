import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage, VecNormalize
from env import make_env 


def main():
    # Create environment (same pipeline as training)
    env = DummyVecEnv([make_env(0, render_mode="human")])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    env = VecNormalize.load("model/ppo/vecnormalize.pkl", env)
    env.training = False
    env.norm_reward = False

    # slow but BEST model
    model = PPO.load("model/ppo/best/best_model")


    # high speeds due to which 90 degree turns not handled well
    # model = PPO.load("model/ppo/_ppo_carracing_custom_reward")

    # pretty good but lot of let-right jitter
    # model = PPO.load("model/ppo/ppo_carracing_800000_steps")


    # close to best in my opinion, good balance of speed and control, handles turns decently
    # model = PPO.load("model/ppo_carracing")



    obs = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        time.sleep(0.03)

        if dones[0]:
            print("Episode finished. Resetting...")
            obs = env.reset()


if __name__ == "__main__":
    main()