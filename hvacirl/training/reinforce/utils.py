import gymnasium as gym
import numpy as np
import torch
from termcolor import colored

from ..models import ppo_model


def create_agent(inter_layers, **kwargs):
    """
    Create a agent either using a pre-trained model or from scratch.

    Args:
        inter_layers: Intermediate layers structure.
        **actor_path: Pre-trained model path.
        **agent_path: Finetuned agent path.
        **test_env: If testing, use this environment to load trained weights.
        **env: Current gymnasium environment instance.

    Returns:
        Deep neural network agent.
    """
    if "path" in kwargs:
        # Load previous model based on different conditions.
        # env: Load model with no changes.
        # test_env: Load model with no changes for evaluation.
        if "env" in kwargs:
            # Load pre-trained model without changes.
            # (Only works for environment with same observation and action spaces).
            print(colored("Model path detected!", "light_blue"))
            print(colored("Loading saved model...", "light_blue"))

            obs_shape = np.array(kwargs["env"].observation_space.shape).prod()
            act_shape = np.array(kwargs["env"].action_space.shape).prod()

            # Create agent and load based on shape info.
            agent = ppo_model.Agent(
                obs_shape,
                act_shape,
                inter_layers,
            )
            agent.actor.load_state_dict(torch.load(kwargs["path"], map_location="cpu"))

            print(colored("Model loaded!", "light_blue"))
            return agent
        elif "test_env" in kwargs:
            # If testing, load pre-trained model without changes.
            print(colored("Model path detected!", "light_blue"))
            print(colored("Loading saved model...", "light_blue"))

            obs_shape = np.array(kwargs["test_env"].observation_space.shape).prod()
            act_shape = np.array(kwargs["test_env"].action_space.shape).prod()

            # Create agent and load based on shape info.
            agent = ppo_model.Agent(
                obs_shape,
                act_shape,
                inter_layers,
            )

            if kwargs["expert"]:
                # If evaluating imitation learning agent.
                agent.actor.load_state_dict(
                    torch.load(kwargs["path"], map_location="cpu")
                )
            else:
                # If evaluating RL agent.
                agent.load_state_dict(torch.load(kwargs["path"], map_location="cpu"))

            actor = agent.actor
            actor.eval()
            print(colored("Model loaded!", "light_blue"))

            return actor
    else:
        # If no pre-trained model will be used, create a new model.
        print(colored("No model path detected!", "light_blue"))
        print(colored("Creating new model...", "light_blue"))

        agent = ppo_model.Agent(
            kwargs["env"].single_observation_space.shape,
            kwargs["env"].single_action_space.shape,
            inter_layers,
        )
        print(colored("Model created!", "light_blue"))
        return agent
