import pickle

import gymnasium as gym
import numpy as np
from termcolor import colored
from sinergym.utils.rewards import ExpReward, LinearReward
from sinergym.utils.wrappers import NormalizeAction, NormalizeObservation


class ReinforcementNormalisation(NormalizeObservation):
    def __init__(self, env, epsilon=1e-8):
        super().__init__(env, epsilon)

    def load_rms_path(self, load_path):
        with open(load_path, "rb") as file_handler:
            self.obs_rms = pickle.load(file_handler)

    def load_rms_obj(self, rms):
        self.obs_rms = rms

    def get_rms(self):
        return self.obs_rms

    def save_rms(self, save_path):
        with open(save_path, "wb") as file_handler:
            pickle.dump(self.obs_rms, file_handler)


def make_env(env_name, reward_type="linear", gamma=None, train=False):
    def thunk():
        if reward_type == "linear":
            reward = LinearReward
        elif reward_type == "exponential":
            reward = ExpReward
        else:
            raise ValueError(f"{reward_type} is not a valid reward type!")
        
        if "5zone" in env_name:
            print(colored("Changing action space...", "light_cyan"))
            # Change range of heating and cooling setpoint to match user comfort.
            new_action_space = gym.spaces.Box(
                low=np.array([15.0, 23.0], dtype=np.float32),
                high=np.array([23.0, 30.0], dtype=np.float32),
                shape=(2,),
                dtype=np.float32,
            )
            env = gym.make(
                env_name,
                config_params={"runperiod": (1, 1, 2023, 31, 12, 2023)},
                action_space=new_action_space,
                reward=reward,
            )
            # raise
        else:
            env = gym.make(
                env_name,
                config_params={"runperiod": (1, 1, 2023, 31, 12, 2023)},
                reward=reward,
            )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # Keep to avoid base 10: '\n' error.
        env = gym.wrappers.ClipAction(env)
        env = NormalizeAction(env)
        env = ReinforcementNormalisation(env)
        if train:
            # Only normalise rewards when training.
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        return env

    return thunk
