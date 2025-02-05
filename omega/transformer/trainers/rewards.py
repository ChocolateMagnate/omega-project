import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CosineSimilarityReward(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return F.cosine_similarity(x, y)

