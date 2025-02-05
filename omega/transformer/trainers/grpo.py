from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import omega.transformer.hyperparameters as hp
from omega.transformer.typing import Vector


def get_group_advantages(rewards: Vector, indices: Vector) -> Vector:
    group_means = torch.zeros_like(rewards)  # [batch_size]
    group_stds = torch.zeros_like(rewards)  # [batch_size]

    for group_idx in torch.unique(indices):
        group_mask = (indices == group_idx)  # [batch_size] boolean mask
        group_rewards = rewards[group_mask]  # [group_size] default 64

        mean = group_rewards.mean()
        std = group_rewards.std()

        group_means[group_mask] = mean
        group_stds[group_mask] = std + 1e-8

    advantages = (rewards - group_means) / group_stds  # [batch_size]
    return advantages


def get_kl_divergence(policy_logits: Tensor, reference_logits: Tensor) -> Tensor:
    policy_probabilities = F.softmax(policy_logits, dim=-1)
    reference_probabilities = F.softmax(reference_logits, dim=-1)

    return torch.sum(
        policy_probabilities * (torch.log(policy_probabilities + 1e-8) - torch.log(reference_probabilities + 1e-8)),
        dim=-1
    )


@dataclass
class GroupRelativePolicyOptimizationConfig:
    kl_coefficient: float = 0.04
    group_size: int = hp.GRPO_GROUP_SIZE


class GroupRelativePolicyOptimizationLoss(nn.Module):
    def __init__(self, policy: nn.Module, reference: nn.Module, scorer: nn.Module,
                 config: Optional[GroupRelativePolicyOptimizationConfig] = None):
        super().__init__()
        self.policy = policy
        self.reference = reference
        self.scorer = scorer
        self.config = config or GroupRelativePolicyOptimizationConfig()

        for parameter in self.policy.parameters():
            parameter.requires_grad = False

        for parameter in self.reference.parameters():
            parameter.requires_grad = False

    def forward(self, embeddings: Tensor, policy_logits: Tensor) -> Tensor:
        with torch.no_grad():
            rewards = self.scorer(embeddings)
            reference_logits = self.reference(embeddings)

        group_indices = torch.arange(len(embeddings)) // self.config.group_size
        advantages = get_group_advantages(rewards, group_indices)
        kl = get_kl_divergence(policy_logits, reference_logits)

        policy_loss = -(advantages * F.log_softmax(policy_logits, dim=-1)).mean()
        loss = policy_loss + self.config.kl_coefficient * kl
        return loss
