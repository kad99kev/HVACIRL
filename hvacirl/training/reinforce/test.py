import os
import pathlib
import random
import time

import numpy as np
import torch
import wandb
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

from hvacirl.config import parse_config
from hvacirl.utils import make_env
from hvacirl.utils.loggers import EvalLogger

from .utils import create_agent


def run(config_path, seed, model_path, expert):
    cfg = parse_config(config_path)

    test_params = cfg["test"]
    inter_layers = cfg["agent"]["layers"]
    trained_seed = int(model_path.split("/")[-4].split("_")[-1])

    # Use test seed.
    cfg["seed"] = 100 + seed

    # Initialise the environment.
    env = make_env(test_params["name"], reward_type=cfg["reward"])()
    # Load normaliser.
    rms_path = pathlib.Path(model_path).parents[0] / "rms.pkl"
    env.load_rms_path(rms_path)
    env.update_running_mean(False)

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.deterministic = cfg["torch_deterministic"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg["cuda"] else "cpu"
    )
    print(colored(f"{device} is being utilised.", "light_cyan"))

    actor = create_agent(
        inter_layers=inter_layers,
        path=model_path,
        test_env=env,
        expert=expert,
    )
    actor.to(device)
    print(colored(f"Model in train mode?: {actor.training}.", "light_cyan"))

    # Logging setup.
    save_path = pathlib.Path(config_path).parents[0] / "test"
    if "wandb" in cfg:
        wandb_config = cfg["wandb"]
        wandb_config["save_path"] = save_path
    else:
        wandb_config = None

    # wandb_config = None
    if wandb_config:
        os.environ["WANDB_API_KEY"] = wandb_config["api_key"]
        cfg["run_name"] += "_" + str(trained_seed)
        wandb.init(
            project=wandb_config["project"],
            entity=wandb_config["entity"],
            name=f"test_{'expert' if expert else 'rl'}_{cfg['run_name']}",
            dir=wandb_config["save_path"],
            sync_tensorboard=True,
            config={
                "seed": cfg["seed"],
                "mode": "test",
                "total_timesteps": test_params["total_timesteps"],
                "log_interval": test_params["log_interval"],
                "layers": cfg["agent"]["layers"],
                "env_name": "-".join(test_params["name"].split("-")[1:4]),
                "algorithm": cfg["algorithm"],
                "experiment": cfg["experiment"],
                "reward": cfg["reward"],
                "agent": "expert" if expert else "rl",
            },
        )
        writer = SummaryWriter(f"{save_path}/tensorboard/")

        # Setup evaluation logger.
        obs_variables = env.get_wrapper_attr("observation_variables")
        action_variables = env.get_wrapper_attr("action_variables")
        eval_logger = EvalLogger(obs_variables, action_variables, writer)

    # Start controller.
    start_time = time.time()
    next_obs, _ = env.reset(seed=cfg["seed"])
    next_obs = torch.Tensor(next_obs).reshape(1, -1).to(device)
    for global_step in range(test_params["total_timesteps"]):
        if global_step % test_params["log_interval"] == 0:
            print(
                colored(
                    f"Timesteps: {global_step + 1} / {test_params['total_timesteps']}",
                    "light_cyan",
                )
            )

        with torch.no_grad():
            action = actor(next_obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(
            action.cpu().numpy()[0]
        )
        # print(next_obs)

        # Logging.
        if wandb_config:
            # Log step values.
            unwrapped_obs = env.unwrapped_observation
            eval_logger.log_update(unwrapped_obs, terminated, info, global_step)

        # Log episode values.
        if terminated:
            print(
                colored(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}",
                    "light_magenta",
                )
            )
            if wandb_config:
                writer.add_scalar(
                    "episode/episodic_return",
                    info["episode"]["r"],
                    global_step,
                )
                writer.add_scalar(
                    "episode/episodic_length",
                    info["episode"]["l"],
                    global_step,
                )
                eval_logger.log_episode(global_step)

            # Reset environment.
            next_obs, _ = env.reset()

        # Convert next obs to a tensor.
        next_obs = torch.Tensor(next_obs).reshape(1, -1).to(device)

        if global_step % test_params["log_interval"] == 0:
            print(
                colored(
                    f"SPS: {int(global_step / (time.time() - start_time))}",
                    "light_magenta",
                )
            )

    env.close()
    writer.close()
