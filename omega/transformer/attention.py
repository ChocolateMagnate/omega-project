import math

import torch
import torch.nn as nn
from torch import Tensor

import omega.transformer.hyperparameters as hp


class FlashAttentionBlock(nn.Module):
    def __init__(self, streams: list[list[torch.cuda.Stream]]):
        super().__init__()

        self.streams = streams
        # Scale is used to prevent dor product between key and query vectors to grow too large.
        self.scale = 1.0 / math.sqrt(hp.HIDDEN_SIZE)

        self.v_projection = nn.Linear(hp.HIDDEN_SIZE, hp.HIDDEN_SIZE)
        self.k_projection = nn.Linear(hp.HIDDEN_SIZE, hp.HIDDEN_SIZE)
        self.q_projection = nn.Linear(hp.HIDDEN_SIZE, hp.HIDDEN_SIZE)
        self.out_projection = nn.Linear(hp.HIDDEN_SIZE, hp.HIDDEN_SIZE)

    def forward(self, batch: Tensor) -> Tensor:
        batch_size, sequence_length, _ = batch.shape
        number_of_tiles = (sequence_length + hp.TILE_SIZE - 1) // hp.TILE_SIZE

        v = self.v_projection(batch)  # [batch_size, sequence_length, hidden_size]
        k = self.k_projection(batch)  # [batch_size, sequence_length, hidden_size]
        q = self.q_projection(batch)  # [batch_size, sequence_length, hidden_size]

        output = torch.zeros_like(q)
        normalizing_factor = torch.zeros((batch_size, sequence_length, 1))

        for q_tile_idx in range(number_of_tiles):
            q_start_idx = q_tile_idx * hp.TILE_SIZE
            q_end_idx = min(q_start_idx + hp.TILE_SIZE, sequence_length)
            q_tile = q[:, q_start_idx:q_end_idx]  # [batch_size, tile_size, hidden_size]

            for kv_tile_idx in range(number_of_tiles):
                kv_start_idx = kv_tile_idx * hp.TILE_SIZE
                kv_end_idx = min(kv_start_idx + hp.TILE_SIZE, sequence_length)

                stream = self.streams[q_tile_idx][kv_tile_idx]
                with torch.cuda.stream(stream):
                    k_tile = k[:, kv_start_idx:kv_end_idx]  # [batch_size, tile_size, hidden_size]
                    v_tile = v[:, kv_start_idx:kv_end_idx]  # [batch_size, tile_size, hidden_size]

                    attention_scores = torch.bmm(q_tile, k_tile.transpose(-2, -1)) * self.scale
                    attention_weights = torch.softmax(attention_scores, dim=-1)
                    weighted_value_vectors = torch.bmm(attention_weights, v_tile)

                    output[:, q_start_idx:q_end_idx] += weighted_value_vectors
                    normalizing_factor[:, q_start_idx:q_end_idx] += attention_weights.sum(dim=-1, keepdim=True)

        torch.cuda.synchronize()
        output = output / (normalizing_factor + 1e-6)
        out = self.out_projection(output)
        return out


class FlashAttention(nn.Module):
    def __init__(self, attention_layers: int):
        super().__init__()
        self.streams = [[torch.cuda.Stream() for _ in range(hp.TILE_SIZE)] for _ in range(hp.TILE_SIZE)]
        self.blocks = nn.ModuleList([
            FlashAttentionBlock(self.streams)
            for _ in range(attention_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x
