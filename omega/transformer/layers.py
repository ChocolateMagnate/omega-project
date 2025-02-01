import torch
import torch.nn as nn
from torch import Tensor

from omega.transformer.typing import Vector

class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank_features: int | None = None):
        super().__init__()
        if rank_features is None:
            self.projection = nn.Linear(in_features, out_features)
        else:
            self.projection = nn.Sequential(
                nn.Linear(in_features, rank_features),
                nn.Linear(rank_features, out_features)
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class HighwayLayer(nn.Module):
    def __init__(self, hidden_features: int, rank_features: int | None = None,
                 activation: nn.Module = nn.GELU, bias: float = -1.0):
        super().__init__()
        self.projection = LowRankLinear(hidden_features, hidden_features, rank_features)
        self.transform = LowRankLinear(hidden_features, hidden_features, rank_features)
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
    def __init__(self, depth: int, hidden_features: int, rank_features: int | None = None,
                 activation: nn.Module = nn.GELU, bias: float = -1.0):
        super().__init__()
        self.depth = depth
        self.race = nn.ModuleList(
            [HighwayLayer(hidden_features, rank_features, activation, bias)
             for _ in range(depth)]
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Vector]:
        steps = torch.tensor(self.depth)
        for idx, highway in enumerate(self.race):
            x = highway(x)
            steps[idx] = x
        return x, steps
