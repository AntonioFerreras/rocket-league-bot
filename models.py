"""
File: discrete_policy.py
Author: Matthew Allen

Description:
    An implementation of a feed-forward neural network which parametrizes a discrete distribution over a space of actions.
"""

from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from rlgym.api import AgentID
from torch.distributions.utils import probs_to_logits

from rlgym_learn_algos.ppo import Actor, Critic

class Linear(torch.nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        compute_type = input.dtype

        w = self.weight.to(compute_type)
        b = self.bias.to(compute_type) if self.bias is not None else None

        return torch.nn.functional.linear(input, w, b)

class BasicCritic(Critic[AgentID, np.ndarray]):
    def __init__(self, input_size, layer_sizes, device, dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        assert (
            len(layer_sizes) != 0
        ), "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"
        layers = [Linear(input_size, layer_sizes[0]), nn.ReLU()]

        prev_size = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers.append(Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(Linear(layer_sizes[-1], 1))
        self.model = nn.Sequential(*layers).to(self.device)

    def forward(self, agent_id_list, obs_list) -> torch.Tensor:
        obs = torch.as_tensor(
            np.array(obs_list), dtype=self.dtype, device=self.device
        )
        output = self.model(obs)
        assert output.dtype == self.dtype
        return output.to(torch.float32)


class DiscreteFF(Actor[AgentID, np.ndarray, np.ndarray]):
    def __init__(self, input_size, n_actions, layer_sizes, device, dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        assert (
            len(layer_sizes) != 0
        ), "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"
        layers = [Linear(input_size, layer_sizes[0]), nn.ReLU()]
        prev_size = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers.append(Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(Linear(layer_sizes[-1], n_actions))
        layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*layers).to(self.device)
        self.n_actions = n_actions

    def get_output(self, obs_list: List[np.ndarray]) -> torch.Tensor:
        obs = torch.as_tensor(
            np.array(obs_list), dtype=self.dtype, device=self.device
        )
        probs = self.model(obs)
        probs = torch.clamp(probs, min=1e-11, max=1)
        return probs

    def get_action(
        self, agent_id_list, obs_list, **kwargs
    ) -> Tuple[Iterable[np.ndarray], torch.Tensor]:
        probs = self.get_output(obs_list)
        if "deterministic" in kwargs and kwargs["deterministic"]:
            action = probs.cpu().numpy().argmax(axis=-1)
            return action, torch.zeros(action.shape)

        action = torch.multinomial(probs, 1, True)
        log_prob: torch.Tensor = torch.log(probs).gather(-1, action)

        return action.cpu().numpy(), log_prob.squeeze().to(
            device="cpu", non_blocking=True
        )

    def get_backprop_data(self, agent_id_list, obs_list, acts, **kwargs):
        probs = self.get_output(obs_list)
        acts_tensor = torch.as_tensor(np.array(acts)).to(self.device)
        logits = probs_to_logits(probs)
        min_real = torch.finfo(logits.dtype).min
        logits = torch.clamp(logits, min=min_real)
        entropy = -(logits * probs).sum(dim=-1)
        action_logits = logits.gather(-1, acts_tensor)

        return action_logits.to(self.device), entropy.to(self.device).mean()
