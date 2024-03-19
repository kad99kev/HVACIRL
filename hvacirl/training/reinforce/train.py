import os
import pathlib
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

from hvacirl.config import parse_config
from hvacirl.utils import make_env
from hvacirl.utils.loggers import TrainLogger

from .eval import evaluate_agent
from .utils import create_agent


def run(config_path, seed, model_path=None, scratch=False):
    # Configuration setup.
    cfg = parse_config(config_path)
    cfg["seed"] = seed
    train_params = cfg["reinforce"]
    env_names = train_params["name"]

    # Hyperparameter setup.
    hp = "scratch" if scratch else "finetune"
    agent_hyperparams = cfg["agent"][hp]
    agent_hyperparams["total_timesteps"] = train_params["total_timesteps"]
    agent_hyperparams["num_envs"] = len(env_names)
    agent_hyperparams["batch_size"] = int(
        agent_hyperparams["num_envs"] * agent_hyperparams["num_steps"]
    )
    agent_hyperparams["minibatch_size"] = int(
        agent_hyperparams["batch_size"] // agent_hyperparams["num_minibatches"]
    )
    CHECKPOINT_FREQ = cfg["agent"]["checkpoint_freq"]

    # TRY NOT TO MODIFY: seeding.
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.deterministic = cfg["torch_deterministic"]

    # Environment setup.
    sub_envs = [
        make_env(
            env_name,
            reward_type=cfg["reward"],
            gamma=agent_hyperparams["gamma"],
            train=True,
        )
        for i, env_name in enumerate(env_names)
    ]
    envs = gym.vector.SyncVectorEnv(sub_envs)
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    # WandB setup
    save_path = pathlib.Path(config_path).parents[0] / (
        "scratch" if scratch else "finetune"
    )
    if "wandb" in cfg:
        wandb_config = cfg["wandb"]
        wandb_config["save_path"] = save_path
    else:
        wandb_config = None

    # wandb_config = None
    if wandb_config:
        os.environ["WANDB_API_KEY"] = wandb_config["api_key"]
        cfg["run_name"] += "_" + str(cfg["seed"])
        init_config = dict(
            **agent_hyperparams,
            **{
                "seed": cfg["seed"],
                "cuda": cfg["cuda"],
                "mode": "scratch" if scratch else "finetune",
                "save_freq": CHECKPOINT_FREQ,
                "layers": cfg["agent"]["layers"],
                "env_name": "-".join(env_names[0].split("-")[1:4]),
                "algorithm": cfg["algorithm"],
                "reward": cfg["reward"],
                "experiment": cfg["experiment"],
                "agent": "rl",
            },
        )
        wandb.init(
            project=wandb_config["project"],
            entity=wandb_config["entity"],
            name=f"{'scratch' if scratch else 'finetune'}_{cfg['run_name']}",
            dir=wandb_config["save_path"],
            sync_tensorboard=True,
            config=init_config,
        )
        writer = SummaryWriter(f"{save_path}/tensorboard/")

        # Initialise logger.
        obs_variables = envs.envs[0].get_wrapper_attr("observation_variables")
        action_variables = envs.envs[0].get_wrapper_attr("action_variables")
        train_logger = TrainLogger(env_names, obs_variables, action_variables, writer)

        # Create folder for model saving.
        model_save_path = pathlib.Path(f"{save_path}/checkpoints/")
        model_save_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg["cuda"] else "cpu"
    )
    print(colored(f"{device} is being utilised.", "light_cyan"))

    # Agent setup.
    if not scratch:
        agent = create_agent(
            inter_layers=cfg["agent"]["layers"],
            env=envs,
            path=model_path,
        )
    else:
        # Create new agent.
        agent = create_agent(inter_layers=cfg["agent"]["layers"], env=envs)

    agent.to(device)  # Change device.
    print(colored(agent, "light_blue"))

    # Storage setup
    obs = torch.zeros(
        (agent_hyperparams["num_steps"], agent_hyperparams["num_envs"])
        + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (agent_hyperparams["num_steps"], agent_hyperparams["num_envs"])
        + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros(
        (agent_hyperparams["num_steps"], agent_hyperparams["num_envs"])
    ).to(device)
    rewards = torch.zeros(
        (agent_hyperparams["num_steps"], agent_hyperparams["num_envs"])
    ).to(device)
    dones = torch.zeros(
        (agent_hyperparams["num_steps"], agent_hyperparams["num_envs"])
    ).to(device)
    values = torch.zeros(
        (agent_hyperparams["num_steps"], agent_hyperparams["num_envs"])
    ).to(device)

    # Start training
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=cfg["seed"])
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(agent_hyperparams["num_envs"]).to(device)
    num_updates = (
        agent_hyperparams["total_timesteps"] // agent_hyperparams["batch_size"]
    )
    print(
        colored(f"{num_updates} updates will be performed for training.", "light_cyan")
    )

    if not scratch and agent_hyperparams["critic_learning"]:
        c_update_end = (
            agent_hyperparams["critic_end"] // agent_hyperparams["batch_size"]
        )
        c_update_start = (
            agent_hyperparams["critic_start"] // agent_hyperparams["batch_size"]
        )

        a_update_end = agent_hyperparams["actor_end"] // agent_hyperparams["batch_size"]
        a_update_start = (
            agent_hyperparams["actor_start"] // agent_hyperparams["batch_size"]
        )

        # Critic optimiser.
        c_optimizer = optim.Adam(
            agent.critic.parameters(), lr=agent_hyperparams["start_critic_lr"], eps=1e-5
        )
        # Actor optimiser.
        a_optimizer = optim.Adam(
            agent.actor.parameters(), lr=agent_hyperparams["start_actor_lr"], eps=1e-5
        )

        critic_len = c_update_end - c_update_start
        # Calculate decay for critic.
        lr_decay = (
            c_optimizer.param_groups[0]["lr"] - agent_hyperparams["end_critic_lr"]
        ) / critic_len

        actor_len = a_update_end - a_update_start
        # Calculate warmup for actor.
        lr_warmup = (
            agent_hyperparams["end_actor_lr"] - a_optimizer.param_groups[0]["lr"]
        ) / actor_len
    else:
        optimizer = optim.Adam(
            agent.parameters(), lr=agent_hyperparams["learning_rate"], eps=1e-5
        )

    for update in range(1, num_updates + 1):
        print(colored(f"Running update {update} of {num_updates}.", "light_cyan"))
        # If critic learning, then decay lr and warmup lr based on current timestep.
        # Stop decay once decay/warmup completes.
        if not scratch and agent_hyperparams["critic_learning"]:
            if update > c_update_start and update < (c_update_end + 1):
                # Decay critic.
                lr_now = c_optimizer.param_groups[0]["lr"] - lr_decay
                c_optimizer.param_groups[0]["lr"] = lr_now

            if update > a_update_start and update < (a_update_end + 1):
                # Warmup actor.
                lr_now = a_optimizer.param_groups[0]["lr"] + lr_warmup
                a_optimizer.param_groups[0]["lr"] = lr_now

        # Simple annealing of learning rate.
        elif agent_hyperparams["anneal_lr"]:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * agent_hyperparams["learning_rate"]
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, agent_hyperparams["num_steps"]):
            global_step += 1 * agent_hyperparams["num_envs"]
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                done
            ).to(device)

            # Log step values.
            if wandb_config:
                train_logger.log_update(envs, terminated, infos, global_step)

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for i, info in enumerate(infos["final_info"]):
                # Skip the envs that are not done
                if info is None:
                    continue
                print(
                    colored(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}",
                        "light_magenta",
                    )
                )
                if wandb_config:
                    writer.add_scalar(
                        f"{env_names[i]}/episode/episodic_return",
                        info["episode"]["r"],
                        global_step,
                    )
                    writer.add_scalar(
                        f"{env_names[i]}/episode/episodic_length",
                        info["episode"]["l"],
                        global_step,
                    )
                    # Log episode.
                    train_logger.log_episode(i, global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(agent_hyperparams["num_steps"])):
                if t == agent_hyperparams["num_steps"] - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t]
                    + agent_hyperparams["gamma"] * nextvalues * nextnonterminal
                    - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + agent_hyperparams["gamma"]
                    * agent_hyperparams["gae_lambda"]
                    * nextnonterminal
                    * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(agent_hyperparams["batch_size"])
        clipfracs = []

        for epoch in range(agent_hyperparams["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(
                0, agent_hyperparams["batch_size"], agent_hyperparams["minibatch_size"]
            ):
                end = start + agent_hyperparams["minibatch_size"]
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > agent_hyperparams["clip_coef"])
                        .float()
                        .mean()
                        .item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if agent_hyperparams["norm_adv"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio,
                    1 - agent_hyperparams["clip_coef"],
                    1 + agent_hyperparams["clip_coef"],
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if agent_hyperparams["clip_vloss"]:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -agent_hyperparams["clip_coef"],
                        agent_hyperparams["clip_coef"],
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                # PPO loss.
                loss = (
                    pg_loss
                    - agent_hyperparams["ent_coef"] * entropy_loss
                    + v_loss * agent_hyperparams["vf_coef"]
                )

                if not scratch and agent_hyperparams["critic_learning"]:
                    # Train critic and actor optimiser if training individually.
                    c_optimizer.zero_grad()
                    a_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        agent.parameters(), agent_hyperparams["max_grad_norm"]
                    )
                    c_optimizer.step()
                    a_optimizer.step()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        agent.parameters(), agent_hyperparams["max_grad_norm"]
                    )
                    optimizer.step()

            if agent_hyperparams["target_kl"] is not None:
                if approx_kl > agent_hyperparams["target_kl"]:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if wandb_config:
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if not scratch and agent_hyperparams["critic_learning"]:
                writer.add_scalar(
                    "learning_rate/critic_lr",
                    c_optimizer.param_groups[0]["lr"],
                    global_step,
                )
                writer.add_scalar(
                    "learning_rate/actor_lr",
                    a_optimizer.param_groups[0]["lr"],
                    global_step,
                )
            else:
                writer.add_scalar(
                    "learning_rate/optimizer_lr",
                    optimizer.param_groups[0]["lr"],
                    global_step,
                )
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar(
                "time/SPS", int(global_step / (time.time() - start_time)), global_step
            )
            writer.add_scalar(
                "time/time_elapsed", time.time() - start_time, global_step
            )
        print(
            colored(
                f"SPS: {int(global_step / (time.time() - start_time))}", "light_magenta"
            )
        )

        # Model saving.
        if wandb_config:
            if update % CHECKPOINT_FREQ == 0 or update == num_updates:
                print(colored("Saving model...", "light_blue"))
                torch.save(agent.state_dict(), f"{model_save_path}/agent_{update}.pt")
                envs.envs[0].save_rms(f"{model_save_path}/rms.pkl")
                wandb.save(
                    f"{model_save_path}/agent_{update}.pt",
                    base_path=model_save_path.parent,
                    policy="now",
                )
                print(colored("Model saved!", "light_blue"))
                eval_return = evaluate_agent(
                    agent, seed, cfg, envs.envs[0].get_rms()
                )
                writer.add_scalar(
                    f"{env_names[i]}/episode/eval_reward",
                    eval_return,
                    global_step,
                )

    envs.close()
    writer.close()
