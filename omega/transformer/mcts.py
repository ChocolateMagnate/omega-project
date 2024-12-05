import torch
import torch.nn as nn
from torch import Tensor


class MCTSStepBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        hidden_features = out_features * 2
        self.step = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.LayerNorm(hidden_features),
            nn.Dropout(0.1),
            nn.Linear(hidden_features, out_features),
            nn.ReLU(),
            nn.LayerNorm(out_features)
        )

        self.projection = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.step(x) + self.projection(x)

class MonteCarloTreeSearch(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, number_of_blocks: int = 3):
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.stairs = nn.ModuleList([
            MCTSStepBlock(hidden_size, hidden_size)
            for _ in range(number_of_blocks)
        ])
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        for step in self.stairs:
            x = step(x)
        x = self.decoder(x)
        return x
