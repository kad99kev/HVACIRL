import numpy as np
import pandas as pd
import pathlib


class EvalLogger:
    def __init__(self, obs_variables, action_variables, writer):
        self.obs_variables = obs_variables
        self.action_variables = action_variables
        self.writer = writer

        # Create path to store history files.
        writer_path = pathlib.Path(writer.log_dir).parent
        self.history_path = pathlib.Path(f"{writer_path}/history")
        self.history_path.mkdir(parents=True, exist_ok=True)
        # Separate out different variables.
        pathlib.Path(f"{self.history_path}/temperatures").mkdir(
            parents=True, exist_ok=True
        )
        pathlib.Path(f"{self.history_path}/setpoints").mkdir(
            parents=True, exist_ok=True
        )

        self.observation_data = {obs_name: [] for obs_name in self.obs_variables}
        self.action_data = {action_name: [] for action_name in self.action_variables}
        self.episodes = 0
        self.episode_data = {
            "rewards": [],
            "powers": [],
            "comfort_penalties": [],
            "power_penalties": [],
            "abs_comfort": [],
            "num_comfort_violation": 0,
            "timesteps": 0,
        }

        # To store hourly and monthly averages.
        # Outdoor Temp.
        self.monthly_outdoor_temp = {m: [] for m in range(12)}
        self.hourly_outdoor_temp = {h: [] for h in range(24)}
        # Indoor Temp.
        self.monthly_indoor_temp = {m: [] for m in range(12)}
        self.hourly_indoor_temp = {h: [] for h in range(24)}
        # Setpoints.
        self.monthly_setpoints = {
            m: {action_name: [] for action_name in self.action_variables}
            for m in range(12)
        }
        self.hourly_setpoints = {
            h: {action_name: [] for action_name in self.action_variables}
            for h in range(24)
        }

        # To store episode summaries.
        # Setup summary data for setpoints.
        self.setpoint_monthly_summary = {
            action_name: {m: [] for m in range(12)}
            for action_name in self.action_variables
        }
        self.setpoint_hourly_summary = {
            action_name: {h: [] for h in range(24)}
            for action_name in self.action_variables
        }
        # Setup summary data for indoor.
        self.indoor_temp_monthly_summary = {m: [] for m in range(12)}
        self.indoor_temp_hourly_summary = {h: [] for h in range(24)}
        # Setup summary data for outdoor.
        self.outdoor_temp_monthly_summary = {m: [] for m in range(12)}
        self.outdoor_temp_hourly_summary = {h: [] for h in range(24)}

    def log_update(self, observation, termination, info, global_step):
        # Log observations.
        # for var_, val in zip(self.obs_variables, observation):
        #     self.observation_data[var_].append(val)

        # Get hour and month information.
        hour, month = info["hour"], info["month"] - 1

        # Log actions.
        for var_, val in zip(self.action_variables, info["action"]):
            self.action_data[var_].append(val)  # Actual action.
            self.hourly_setpoints[hour][var_].append(val)
            self.monthly_setpoints[month][var_].append(val)

        # Log outdoor and indoor temperature.
        outdoor_temp = observation[self.obs_variables.index("outdoor_temperature")]
        self.hourly_outdoor_temp[hour].append(outdoor_temp)
        self.monthly_outdoor_temp[month].append(outdoor_temp)

        indoor_temp = info["temp_values"]
        self.hourly_indoor_temp[hour].append(indoor_temp)
        self.monthly_indoor_temp[month].append(indoor_temp)

        # Log info.
        self.episode_data["rewards"].append(info["reward"])
        self.episode_data["powers"].append(info["energy_values"])
        self.episode_data["comfort_penalties"].append(info["comfort_term"])
        self.episode_data["power_penalties"].append(info["energy_term"])
        self.episode_data["abs_comfort"].append(info["abs_comfort"])
        if info["comfort_term"] != 0:
            self.episode_data["num_comfort_violation"] += 1
        self.episode_data["timesteps"] += 1
        self.episode_data["time_elapsed"] = info["time_elapsed(hours)"]

    def log_episode(self, global_step):
        self.episodes += 1
        episode_metrics = {}
        # Reward.
        episode_metrics["mean_reward"] = np.mean(self.episode_data["rewards"])
        # Timesteps.
        episode_metrics["episode_length"] = self.episode_data["timesteps"]
        # Power.
        episode_metrics["mean_power"] = np.mean(self.episode_data["powers"])
        episode_metrics["cumulative_power"] = np.sum(self.episode_data["powers"])
        # Comfort Penalty.
        episode_metrics["mean_comfort_penalty"] = np.mean(
            self.episode_data["comfort_penalties"]
        )
        episode_metrics["cumulative_comfort_penalty"] = np.sum(
            self.episode_data["comfort_penalties"]
        )
        # Power Penalty.
        episode_metrics["mean_power_penalty"] = np.mean(
            self.episode_data["power_penalties"]
        )
        episode_metrics["cumulative_power_penalty"] = np.sum(
            self.episode_data["power_penalties"]
        )
        episode_metrics["episode_num"] = self.episodes

        try:
            episode_metrics["comfort_violation_time(%)"] = (
                self.episode_data["num_comfort_violation"]
                / self.episode_data["timesteps"]
                * 100
            )
        except ZeroDivisionError:
            episode_metrics["comfort_violation_time(%)"] = np.nan

        for key, metric in episode_metrics.items():
            self.writer.add_scalar(f"episode/{key}", metric, global_step)

        # Reset data for episode.
        self.episode_data = {
            "rewards": [],
            "powers": [],
            "comfort_penalties": [],
            "power_penalties": [],
            "abs_comfort": [],
            "num_comfort_violation": 0,
            "timesteps": 0,
        }

        # Observation data.
        episode_observations = {}
        for key, val in self.observation_data.items():
            episode_observations[key] = np.mean(val)

        for key, metric in episode_observations.items():
            self.writer.add_scalar(f"observations/{key}", metric, global_step)

        # Action data.
        episode_actions = {}
        for key, val in self.action_data.items():
            episode_actions[key] = np.mean(val)

        for key, metric in episode_actions.items():
            self.writer.add_scalar(f"actions/{key}", metric, global_step)

        # Monthly data.
        for month in range(12):
            # Setpoints.
            for key, metric in self.monthly_setpoints[month].items():
                avg_monthly_temp = np.mean(metric)
                self.setpoint_monthly_summary[key][month].append(avg_monthly_temp)
            # Outdoor and indoor temperatures.
            self.outdoor_temp_monthly_summary[month].append(
                np.mean(self.monthly_outdoor_temp[month])
            )
            self.indoor_temp_monthly_summary[month].append(
                np.mean(self.monthly_indoor_temp[month])
            )

        # Hourly actions.
        for hour in range(24):
            # Setpoints.
            for key, metric in self.hourly_setpoints[hour].items():
                avg_hourly_temp = np.mean(metric)
                self.setpoint_hourly_summary[key][hour].append(avg_hourly_temp)
            # Outdoor and indoor temperatures.
            self.outdoor_temp_hourly_summary[hour].append(
                np.mean(self.hourly_outdoor_temp[hour])
            )
            self.indoor_temp_hourly_summary[hour].append(
                np.mean(self.hourly_indoor_temp[hour])
            )

        # Update histroy file to track temperatures.
        pd.DataFrame(self.outdoor_temp_hourly_summary).T.to_csv(
            f"{self.history_path}/temperatures/outdoor_temp_hourly_summary.csv",
            index=False,
        )
        pd.DataFrame(self.outdoor_temp_monthly_summary).T.to_csv(
            f"{self.history_path}/temperatures/outdoor_temp_monthly_summary.csv",
            index=False,
        )
        pd.DataFrame(self.indoor_temp_hourly_summary).T.to_csv(
            f"{self.history_path}/temperatures/indoor_temp_hourly_summary.csv",
            index=False,
        )
        pd.DataFrame(self.indoor_temp_monthly_summary).T.to_csv(
            f"{self.history_path}/temperatures/indoor_temp_monthly_summary.csv",
            index=False,
        )

        # Update history file to track setpoints.
        for setpoint, values in self.setpoint_hourly_summary.items():
            pd.DataFrame(values).T.to_csv(
                f"{self.history_path}/setpoints/{setpoint}_hourly_summary.csv",
                index=False,
            )
        for setpoint, values in self.setpoint_monthly_summary.items():
            pd.DataFrame(values).T.to_csv(
                f"{self.history_path}/setpoints/{setpoint}_monthly_summary.csv",
                index=False,
            )

        # Reset values.
        self.observation_data = {obs_name: [] for obs_name in self.obs_variables}
        self.action_data = {action_name: [] for action_name in self.action_variables}
        # Outdoor.
        self.monthly_outdoor_temp = {m: [] for m in range(12)}
        self.hourly_outdoor_temp = {h: [] for h in range(24)}
        # Indoor.
        self.monthly_indoor_temp = {m: [] for m in range(12)}
        self.hourly_indoor_temp = {h: [] for h in range(24)}
        # Setpoints.
        self.monthly_setpoints = {
            m: {action_name: [] for action_name in self.action_variables}
            for m in range(12)
        }
        self.hourly_setpoints = {
            h: {action_name: [] for action_name in self.action_variables}
            for h in range(24)
        }


class TrainLogger:
    def __init__(self, env_names, obs_variables, action_variables, writer):
        self.obs_variables = obs_variables
        self.action_variables = action_variables
        self.env_names = env_names
        self.writer = writer

        # Create path to store history file.
        writer_path = pathlib.Path(writer.log_dir).parent
        self.history_path = pathlib.Path(f"{writer_path}/history")
        self.history_path.mkdir(parents=True, exist_ok=True)
        # Separate out different variables.
        pathlib.Path(f"{self.history_path}/temperatures").mkdir(
            parents=True, exist_ok=True
        )
        pathlib.Path(f"{self.history_path}/setpoints").mkdir(
            parents=True, exist_ok=True
        )

        self.observation_data = [
            {obs_name: [] for obs_name in self.obs_variables}
            for _ in range(len(env_names))
        ]
        self.action_data = [
            {action_name: [] for action_name in self.action_variables}
            for _ in range(len(env_names))
        ]
        self.episodes = [0 for _ in range(len(env_names))]
        self.episode_data = [
            {
                "rewards": [],
                "powers": [],
                "comfort_penalties": [],
                "power_penalties": [],
                "abs_comfort": [],
                "num_comfort_violation": 0,
                "timesteps": 0,
            }
            for _ in range(len(env_names))
        ]

        # To store hourly and monthly averages.
        # Outdoor Temp.
        self.monthly_outdoor_temp = [
            {m: [] for m in range(12)} for _ in range(len(env_names))
        ]
        self.hourly_outdoor_temp = [
            {h: [] for h in range(24)} for _ in range(len(env_names))
        ]
        # Indoor Temp.
        self.monthly_indoor_temp = [
            {m: [] for m in range(12)} for _ in range(len(env_names))
        ]
        self.hourly_indoor_temp = [
            {h: [] for h in range(24)} for _ in range(len(env_names))
        ]
        # Setpoints.
        self.monthly_setpoints = [
            {
                m: {action_name: [] for action_name in self.action_variables}
                for m in range(12)
            }
            for _ in range(len(env_names))
        ]
        self.hourly_setpoints = [
            {
                h: {action_name: [] for action_name in self.action_variables}
                for h in range(24)
            }
            for _ in range(len(env_names))
        ]

        # To store episode summaries.
        # Setup summary data for setpoints.
        self.setpoint_monthly_summary = [
            {
                action_name: {m: [] for m in range(12)}
                for action_name in self.action_variables
            }
            for _ in range(len(self.env_names))
        ]
        self.setpoint_hourly_summary = [
            {
                action_name: {h: [] for h in range(24)}
                for action_name in self.action_variables
            }
            for _ in range(len(self.env_names))
        ]
        # Setup summary data for indoor.
        self.indoor_temp_monthly_summary = [
            {m: [] for m in range(12)} for _ in range(len(env_names))
        ]
        self.indoor_temp_hourly_summary = [
            {h: [] for h in range(24)} for _ in range(len(env_names))
        ]
        # Setup summary data for outdoor.
        self.outdoor_temp_monthly_summary = [
            {m: [] for m in range(12)} for _ in range(len(env_names))
        ]
        self.outdoor_temp_hourly_summary = [
            {h: [] for h in range(24)} for _ in range(len(env_names))
        ]

    def log_update(self, envs, terminations, infos, global_step):
        # Split infos into separate dictionaries.
        # Source: https://stackoverflow.com/a/1780295
        split_infos = list(
            map(dict, zip(*[[(k, v) for v in value] for k, value in infos.items()]))
        )
        # print(split_infos)

        for i, env_name in enumerate(self.env_names):
            unwrapped_obs = envs.envs[i].unwrapped_observation
            terminated = terminations[i]
            info = split_infos[i]

            if "final_info" in info:
                info = info["final_info"]

            # # Log observations.
            # for var_, val in zip(self.obs_variables, unwrapped_obs):
            #     self.observation_data[i][var_].append(val)

            # Get hour and month information
            hour, month = info["hour"], info["month"] - 1

            # Log actions.
            for var_, val in zip(self.action_variables, info["action"]):
                self.action_data[i][var_].append(val)  # Actual action.
                self.hourly_setpoints[i][hour][var_].append(val)
                self.monthly_setpoints[i][month][var_].append(val)

            # Log outdoor and indoor temperature.
            outdoor_temp = unwrapped_obs[
                self.obs_variables.index("outdoor_temperature")
            ]
            self.hourly_outdoor_temp[i][hour].append(outdoor_temp)
            self.monthly_outdoor_temp[i][month].append(outdoor_temp)

            indoor_temp = info["temp_values"][0]
            self.hourly_indoor_temp[i][hour].append(indoor_temp)
            self.monthly_indoor_temp[i][month].append(indoor_temp)

            # Log info.
            self.episode_data[i]["rewards"].append(info["reward"])
            self.episode_data[i]["powers"].append(info["energy_values"])
            self.episode_data[i]["comfort_penalties"].append(info["comfort_term"])
            self.episode_data[i]["power_penalties"].append(info["energy_term"])
            self.episode_data[i]["abs_comfort"].append(info["abs_comfort"])
            if info["comfort_term"] != 0:
                self.episode_data[i]["num_comfort_violation"] += 1
            self.episode_data[i]["timesteps"] += 1
            self.episode_data[i]["time_elapsed"] = info["time_elapsed(hours)"]

    def log_episode(self, idx, global_step):
        env_name = self.env_names[idx]
        self.episodes[idx] += 1
        episode_metrics = {}
        # Reward.
        episode_metrics["mean_reward"] = np.mean(self.episode_data[idx]["rewards"])
        # Timesteps.
        episode_metrics["episode_length"] = self.episode_data[idx]["timesteps"]
        # Power.
        episode_metrics["mean_power"] = np.mean(self.episode_data[idx]["powers"])
        episode_metrics["cumulative_power"] = np.sum(self.episode_data[idx]["powers"])
        # Comfort Penalty.
        episode_metrics["mean_comfort_penalty"] = np.mean(
            self.episode_data[idx]["comfort_penalties"]
        )
        episode_metrics["cumulative_comfort_penalty"] = np.sum(
            self.episode_data[idx]["comfort_penalties"]
        )
        # Power Penalty.
        episode_metrics["mean_power_penalty"] = np.mean(
            self.episode_data[idx]["power_penalties"]
        )
        episode_metrics["cumulative_power_penalty"] = np.sum(
            self.episode_data[idx]["power_penalties"]
        )
        episode_metrics["episode_num"] = self.episodes[idx]

        try:
            episode_metrics["comfort_violation_time(%)"] = (
                self.episode_data[idx]["num_comfort_violation"]
                / self.episode_data[idx]["timesteps"]
                * 100
            )
        except ZeroDivisionError:
            episode_metrics["comfort_violation_time(%)"] = np.nan

        for key, metric in episode_metrics.items():
            self.writer.add_scalar(f"{env_name}/episode/{key}", metric, global_step)

        # Reset data for episode.
        self.episode_data[idx] = {
            "rewards": [],
            "powers": [],
            "comfort_penalties": [],
            "power_penalties": [],
            "abs_comfort": [],
            "num_comfort_violation": 0,
            "timesteps": 0,
        }

        # Observation data.
        episode_observations = {}
        for key, val in self.observation_data[idx].items():
            episode_observations[key] = np.mean(val)

        for key, metric in episode_observations.items():
            self.writer.add_scalar(f"observations/{key}", metric, global_step)

        # Action data.
        episode_actions = {}
        for key, val in self.action_data[idx].items():
            episode_actions[key] = np.mean(val)

        for key, metric in episode_actions.items():
            self.writer.add_scalar(f"actions/{key}", metric, global_step)

        # Monthly actions.
        for month in range(12):
            # Setpoints.
            for key, metric in self.monthly_setpoints[idx][month].items():
                avg_monthly_temp = np.mean(metric)
                self.setpoint_monthly_summary[idx][key][month].append(avg_monthly_temp)
            # Outdoor and indoor temperatures.
            self.outdoor_temp_monthly_summary[idx][month].append(
                np.mean(self.monthly_outdoor_temp[idx][month])
            )
            self.indoor_temp_monthly_summary[idx][month].append(
                np.mean(self.monthly_indoor_temp[idx][month])
            )

        # Hourly actions.
        for hour in range(24):
            # Setpoints.
            for key, metric in self.hourly_setpoints[idx][hour].items():
                avg_hourly_temp = np.mean(metric)
                self.setpoint_hourly_summary[idx][key][hour].append(avg_hourly_temp)
            # Outdoor and indoor temperatures.
            self.outdoor_temp_hourly_summary[idx][hour].append(
                np.mean(self.hourly_outdoor_temp[idx][hour])
            )
            self.indoor_temp_hourly_summary[idx][hour].append(
                np.mean(self.hourly_indoor_temp[idx][hour])
            )

        # Update history file to track setpoints.
        for setpoint, values in self.setpoint_hourly_summary[idx].items():
            pd.DataFrame(values).T.to_csv(
                f"{self.history_path}/setpoints/{env_name}_{setpoint}_hourly_summary.csv",
                index=False,
            )
        for setpoint, values in self.setpoint_monthly_summary[idx].items():
            pd.DataFrame(values).T.to_csv(
                f"{self.history_path}/setpoints/{env_name}_{setpoint}_monthly_summary.csv",
                index=False,
            )
        # Update histroy file to track temperatures.
        pd.DataFrame(self.outdoor_temp_hourly_summary[idx]).T.to_csv(
            f"{self.history_path}/temperatures/outdoor_temp_hourly_summary.csv",
            index=False,
        )
        pd.DataFrame(self.outdoor_temp_monthly_summary[idx]).T.to_csv(
            f"{self.history_path}/temperatures/outdoor_temp_monthly_summary.csv",
            index=False,
        )
        pd.DataFrame(self.indoor_temp_hourly_summary[idx]).T.to_csv(
            f"{self.history_path}/temperatures/indoor_temp_hourly_summary.csv",
            index=False,
        )
        pd.DataFrame(self.indoor_temp_monthly_summary[idx]).T.to_csv(
            f"{self.history_path}/temperatures/indoor_temp_monthly_summary.csv",
            index=False,
        )

        # Reset values.
        self.observation_data[idx] = {obs_name: [] for obs_name in self.obs_variables}
        self.action_data[idx] = {
            action_name: [] for action_name in self.action_variables
        }
        self.monthly_outdoor_temp[idx] = {m: [] for m in range(12)}
        self.hourly_outdoor_temp[idx] = {h: [] for h in range(24)}
        self.monthly_indoor_temp[idx] = {m: [] for m in range(12)}
        self.hourly_indoor_temp[idx] = {h: [] for h in range(24)}
        self.monthly_setpoints[idx] = {
            m: {action_name: [] for action_name in self.action_variables}
            for m in range(12)
        }
        self.hourly_setpoints[idx] = {
            h: {action_name: [] for action_name in self.action_variables}
            for h in range(24)
        }
