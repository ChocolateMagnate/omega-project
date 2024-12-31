import torch
import torch.nn as nn
import torch.jit as jit
import torch.nn.functional as F
from torch import Tensor

import omega.transformer.layers as layers
import omega.transformer.hyperparameters as hp
from omega.transformer.typing import Vector


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.expansion = nn.Sequential(
            nn.Linear(hp.HIDDEN_SIZE, hp.THOUGHT_SIZE),
            nn.GELU()
        )

    def forward(self, last_hidden_states: Tensor) -> Tensor:
        return self.expansion(last_hidden_states)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.compression = nn.Sequential(
            nn.Linear(hp.THOUGHT_SIZE, hp.HIDDEN_SIZE),
            nn.GELU()
        )

    def forward(self, token_thought_vectors: Tensor) -> Tensor:
        return self.compression(token_thought_vectors)


class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.centroids = nn.Parameter(torch.rand(hp.NUMBER_OF_EXPERTS, hp.TOPIC_CLUSTER_SIZE))
        self.distillation = nn.Sequential(
            nn.Linear(hp.HIDDEN_SIZE, hp.TOPIC_CLUSTER_SIZE),
            nn.GELU()
        )

    def forward(self, last_hidden_states: Tensor, k: int) -> tuple[Vector, Vector]:
        distilled = self.distillation(last_hidden_states)
        distilled = F.normalize(distilled, dim=-1, eps=1e-8)
        centroids = F.normalize(self.centroids, dim=-1, eps=1e-8)
        distances = torch.cdist(distilled, centroids)
        probabilities = torch.softmax(-distances, dim=-1)
        top_k_probabilities, top_k_indices = torch.topk(probabilities, k, dim=1)
        return top_k_probabilities, top_k_indices


class MixtureOfExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = Router()
        self.experts = nn.ModuleList([
            layers.Highway(hp.HORIZON, hp.THOUGHT_SIZE, hp.THOUGHT_RANK_SIZE)
            for _ in range(hp.NUMBER_OF_EXPERTS)
        ])

    @jit.script_method
    def forward(self, premise: Tensor) -> Tensor:
        chosen_expert_probabilities, chosen_expert_indices = self.router(premise)

        consensus = torch.zeros_like(premise)
        expert_futures = []

        for expert_idx, expert in enumerate(self.experts):
            expert_mask = (chosen_expert_indices == expert_idx)
            tokens_for_expert = premise[expert_mask]

            if tokens_for_expert.size(0) > 0:
                future = jit.fork(expert, tokens_for_expert)
                expert_futures.append((future, expert_mask, expert_idx))

        for future, expert_mask, expert_idx in expert_futures:
            expert_output = jit.wait(future)
            consensus[expert_mask] = expert_output * chosen_expert_probabilities[expert_mask, expert_idx].unsqueeze(-1)

        return consensus



class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.moe = MixtureOfExperts()
        self.decoder = Decoder()

    def forward(self, last_hidden_states: Tensor) -> Tensor:
        premise = self.encoder(last_hidden_states).mean(dim=-1)
        conclusion = self.moe(premise)
        final_states = self.decoder(conclusion)
        return final_states