from env import get_env


def main():
    env = get_env(n_envs=1)

    obs = env.reset()

    for _ in range(50):
        action = [env.action_space.sample()]
        obs, rewards, dones, infos = env.step(action)

        print("Reward:", rewards[0])

        if dones[0]:
            print("Episode done")
            obs = env.reset()


if __name__ == "__main__":
    main()