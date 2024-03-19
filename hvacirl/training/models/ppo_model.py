import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Layer initialisation.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class FeatureNetwork(nn.Module):
    def __init__(self, layers_list):
        """
        Hidden layers for Actor and Critic.
        """
        super().__init__()
        self.layers_list = layers_list
        self.network = nn.Sequential()
        prev_out_shape = self.layers_list[0]
        for layer in self.layers_list:
            self.network.append(nn.Linear(prev_out_shape, layer))
            self.network.append(nn.Tanh())
            prev_out_shape = layer

    def forward(self, x):
        return self.network(x)


class Critic(nn.Module):
    def __init__(self, observation_shape, inter_layers=None):
        """
        Critic.
        """
        super().__init__()
        self.obs_shape = observation_shape
        self.inter_layers_list = [64] if inter_layers is None else inter_layers
        self.input_layer = nn.Sequential(
            layer_init(nn.Linear(self.obs_shape, self.inter_layers_list[0])), nn.Tanh()
        )
        self.feature_network = FeatureNetwork(self.inter_layers_list)
        self.output_layer = layer_init(
            nn.Linear(self.inter_layers_list[-1], 1), std=1.0
        )

    def __str__(self):
        return f"{str(self.input_layer)}\n{str(self.feature_network)}\n{str(self.output_layer)}"

    def forward(self, inputs):
        inp_outs = self.input_layer(inputs)
        feature_net_outs = self.feature_network(inp_outs)
        outs = self.output_layer(feature_net_outs)
        return outs


class Actor(nn.Module):
    def __init__(self, observation_shape, action_shape, inter_layers=None):
        """
        Actor.
        """
        super().__init__()
        self.obs_shape = observation_shape
        self.act_shape = action_shape
        self.inter_layers_list = [64] if inter_layers is None else inter_layers
        self.input_layer = nn.Sequential(
            layer_init(nn.Linear(self.obs_shape, self.inter_layers_list[0])), nn.Tanh()
        )
        self.feature_network = FeatureNetwork(self.inter_layers_list)
        self.output_layer = layer_init(
            nn.Linear(self.inter_layers_list[-1], self.act_shape), std=0.01
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(self.act_shape)))

    def __str__(self):
        return f"{str(self.input_layer)}\n{str(self.feature_network)}\n{str(self.output_layer)}"

    def forward(self, inputs, action=None, deterministic=False):
        # Actor mean
        inp_outs = self.input_layer(inputs)
        inter_outs = self.feature_network(inp_outs)
        action_mean = self.output_layer(inter_outs)

        # Actor std
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        # Return mode for deterministic actions
        if deterministic:
            return probs.mode
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)


class Agent(nn.Module):
    def __init__(self, observation_shape, action_shape, inter_layers=None):
        """
        Actor-Critic Agent.
        """
        super().__init__()
        self.obs_shape = np.array(observation_shape).prod()
        self.act_shape = np.array(action_shape).prod()
        self.inter_layers_list = [64] if inter_layers is None else inter_layers

        self.critic = Critic(self.obs_shape, self.inter_layers_list)
        self.actor = Actor(self.obs_shape, self.act_shape, self.inter_layers_list)

    def __str__(self):
        model_str = ""
        model_str += f"Critic:\n{str(self.critic)}\n\nActor:\n{str(self.actor)}"
        return model_str

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action, log_prob_sum, prob_sum = self.actor(x, action=action)
        critic_out = self.critic(x)
        return action, log_prob_sum, prob_sum, critic_out
