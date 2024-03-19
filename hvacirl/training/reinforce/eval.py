import random
import time

import numpy as np
import torch
from termcolor import colored

from hvacirl.utils import make_env


def evaluate_agent(agent, seed, train_cfg, rms):
    eval_cfg = train_cfg.copy()
    eval_params = eval_cfg["eval"]

    # Use eval seed.
    eval_cfg["seed"] = 100 + seed

    # Initialise the environment.
    env = make_env(eval_params["name"], reward_type=eval_cfg["reward"])()
    env.load_rms_obj(rms)
    env.update_running_mean(False)

    # TRY NOT TO MODIFY: seeding
    random.seed(eval_cfg["seed"])
    np.random.seed(eval_cfg["seed"])
    torch.manual_seed(eval_cfg["seed"])
    torch.backends.cudnn.deterministic = eval_cfg["torch_deterministic"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() and eval_cfg["cuda"] else "cpu"
    )
    print(colored(f"{device} is being utilised.", "light_cyan"))

    # Start controller.
    start_time = time.time()
    next_obs, _ = env.reset(seed=eval_cfg["seed"])
    next_obs = torch.Tensor(next_obs).reshape(1, -1).to(device)
    for global_step in range(eval_params["total_timesteps"]):
        if global_step % eval_params["log_interval"] == 0:
            print(
                colored(
                    f"Timesteps: {global_step + 1} / {eval_params['total_timesteps']}",
                    "light_cyan",
                )
            )

        with torch.no_grad():
            action = agent.actor(next_obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(
            action.cpu().numpy()[0]
        )

        # Log episode values.
        if terminated:
            print(
                colored(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}",
                    "light_magenta",
                )
            )

            return info["episode"]["r"]

        # Convert next obs to a tensor.
        next_obs = torch.Tensor(next_obs).reshape(1, -1).to(device)

        if global_step % eval_params["log_interval"] == 0:
            print(
                colored(
                    f"SPS: {int(global_step / (time.time() - start_time))}",
                    "light_magenta",
                )
            )
