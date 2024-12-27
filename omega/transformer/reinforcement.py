import torch
import torch.nn as nn
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
        self.experts = nn.ModuleList([layers.Highway(hp.HORIZON, hp.THOUGHT_SIZE, hp.THOUGHT_RANK_SIZE)
                                     for _ in range(hp.NUMBER_OF_EXPERTS)])

    def forward(self, last_hidden_states: Tensor) -> Tensor:
        chosen_expert_probabilities, chosen_expert_indices = self.router(last_hidden_states)
        outputs = torch.zeros(torch.Size(hp.NUMBER_OF_EXPERTS))
        for chosen_expert_index in chosen_expert_indices:
            with torch.cuda.Stream():
                expert = self.experts[chosen_expert_index]
                output = expert(last_hidden_states)
                outputs[chosen_expert_index] = output
        torch.cuda.synchronize()



class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.moe = MixtureOfExperts()
        self.decoder = Decoder()

    def forward(self, last_hidden_states: Tensor) -> Tensor:
        token_thought_states = self.encoder(last_hidden_states)
