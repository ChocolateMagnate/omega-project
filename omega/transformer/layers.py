import torch
import torch.nn as nn
from torch import Tensor

import omega.transformer.hyperparameters as hp


class HighwayLayer(nn.Module):
    def __init__(self, hidden_features: int, activation: nn.Module = nn.GELU, bias: float = -1.0):
        super().__init__()
        self.projection = nn.Linear(hidden_features, hidden_features)
        self.transform = nn.Linear(hidden_features, hidden_features)
        self.activation = activation
        self.transform.bias.data.fill_(bias)

    def forward(self, x: Tensor) -> Tensor:
        projected = self.projection(x)
        activated = self.activation(projected)
        transformed = torch.sigmoid(self.transform(activated))

        # Highway network formula: t * H(x) + (1-t) * x
        # where t is transform gate and H(x) is the projected transformation
        return transformed * activated + (1 - transformed) * x

class Highway(nn.Module):
    def __init__(self, hidden_features: int, activation: nn.Module = nn.GELU, bias: float = -1.0):
        super().__init__()
        self.race = nn.ModuleList(
            [HighwayLayer(hidden_features, activation, bias)
             for _ in range(hp.HIGHWAY_DEPTH)]
        )

    def forward(self, x: Tensor) -> Tensor:
        for highway in self.race:
            x = highway(x)
        return x
