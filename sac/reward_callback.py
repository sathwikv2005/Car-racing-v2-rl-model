from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class RewardLoggingCallback(BaseCallback):
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

        self.episode_rewards = []
        self.episode_lengths = []

        self.current_episode = {}

    def _on_step(self) -> bool:
        infos = self.locals["infos"]

        for info in infos:
            if "reward_components" in info:
                rc = info["reward_components"]

                # accumulate per episode
                for k, v in rc.items():
                    self.current_episode[k] = self.current_episode.get(k, 0) + v

            # detect episode end
            if "episode" in info:
                ep_len = info["episode"]["l"]
                self.episode_lengths.append(ep_len)

                # log aggregated reward components
                for k, v in self.current_episode.items():
                    self.logger.record(f"reward_ep/{k}", v)

                self.current_episode = {}

        # periodic logging (global stats)
        if self.n_calls % self.log_freq == 0:
            self.logger.record("train/steps", self.num_timesteps)

            if len(self.episode_lengths) > 0:
                self.logger.record(
                    "train/ep_len_mean",
                    np.mean(self.episode_lengths[-10:])
                )

        return True