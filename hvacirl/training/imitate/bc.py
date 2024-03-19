import os
import pathlib
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

from hvacirl.config import parse_config

from ..models import ppo_model
from .utils import prepare_loaders


def run(config_path, seed):
    # Configuration setup
    cfg = parse_config(config_path)
    cfg["seed"] = seed
    imitation_cfg = cfg["imitation"]

    df = pd.read_csv(imitation_cfg["data_path"])

    X = df.iloc[:, :-2]
    y = df.iloc[:, -2:]

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.deterministic = cfg["torch_deterministic"]

    # WandB setup
    save_path = pathlib.Path(config_path).parents[0] / "imitate"
    if "wandb" in cfg:
        wandb_config = cfg["wandb"]
        wandb_config["save_path"] = save_path
    else:
        wandb_config = None

    # wandb_config = None
    if wandb_config:
        os.environ["WANDB_API_KEY"] = wandb_config["api_key"]
        cfg["run_name"] += "_" + str(cfg["seed"])
        wandb.init(
            project=wandb_config["project"],
            entity=wandb_config["entity"],
            name=f"imitate_{cfg['run_name']}",
            dir=wandb_config["save_path"],
            sync_tensorboard=True,
            config=dict(
                **imitation_cfg,
                **{
                    "seed": cfg["seed"],
                    "cuda": cfg["cuda"],
                    "mode": "imitate",
                    "layers": cfg["agent"]["layers"],
                    "algorithm": cfg["algorithm"],
                    "reward": cfg["reward"],
                    "experiment": cfg["experiment"],
                    "agent": "expert",
                },
            ),
        )
        writer = SummaryWriter(f"{save_path}/tensorboard/")

    # Create folder for model saving.
    model_save_path = pathlib.Path(f"{save_path}/models/")
    model_save_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg["cuda"] else "cpu"
    )
    print(colored(f"{device} is being utilised.", "light_cyan"))

    model = ppo_model.Actor(X.shape[-1], y.shape[-1], cfg["agent"]["layers"])
    model = model.to(device)

    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters())

    total_samples = 0
    stop = False
    loss_per_epoch = []

    normaliser, loaders = prepare_loaders(
        X, y, imitation_cfg["batch_size"], cfg["seed"]
    )
    train_loader, test_loader = loaders

    pbar = tqdm.tqdm(total=imitation_cfg["max_samples"])
    ent_weight = imitation_cfg["ent_weight"]
    while True:
        losses = 0
        steps = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Train imitation actor
            _, log_prob, entropy = model(X_batch, action=y_batch)

            ent_loss = -ent_weight * entropy.mean()
            neglogprob = -log_prob.mean()
            loss = neglogprob + ent_loss

            losses += loss.item()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            steps += X_batch.size(0)
            if (total_samples + steps) >= imitation_cfg["max_samples"]:
                stop = True
                break

        total_samples += steps
        avg_loss = losses / (len(train_loader))
        loss_per_epoch.append(avg_loss)
        pbar.set_description(f"Average train loss: {avg_loss}")
        pbar.update(steps)

        if wandb_config:
            writer.add_scalar("losses/train_epoch_loss", avg_loss, steps)

        if stop:
            break

    pbar.close()

    model.eval()
    test_losses = 0
    for batch_index, (X_batch, y_batch) in enumerate(test_loader):
        with torch.no_grad():
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch, deterministic=True)
            loss = criterion(y_pred, y_batch)
            test_losses += loss.item()

    avg_test_loss = test_losses / len(test_loader)
    print(f"Average test loss: {avg_test_loss}")
    if wandb_config:
        writer.add_scalar("losses/test_loss", avg_test_loss, steps)

    # Save actor and normaliser.
    torch.save(
        model.state_dict(), f"{model_save_path}/actor_{imitation_cfg['max_samples']}.pt"
    )
    normaliser.save_rms(f"{model_save_path}/rms.pkl")
