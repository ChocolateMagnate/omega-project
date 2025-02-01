import math

import torch
import torch.nn as nn
from torch import Tensor
from flash_attn import flash_attn_func

import omega.transformer.hyperparameters as hp


class FlashAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalization = nn.LayerNorm(hp.HIDDEN_SIZE)
        self.v_projection = nn.Linear(hp.HIDDEN_SIZE, hp.HIDDEN_SIZE)
        self.k_projection = nn.Linear(hp.HIDDEN_SIZE, hp.HIDDEN_SIZE)
        self.q_projection = nn.Linear(hp.HIDDEN_SIZE, hp.HIDDEN_SIZE)
        self.out_projection = nn.Linear(hp.HIDDEN_SIZE, hp.HIDDEN_SIZE)

    def forward(self, batch: Tensor) -> Tensor:
        assert batch.dim() == 3, f"Expected 3D tensor, got {batch.dim()}D"
        assert batch.size(-1) == hp.HIDDEN_SIZE, f"Expected hidden size {hp.HIDDEN_SIZE}, got {batch.size(-1)}"

        residual = batch
        batch_size, sequence_length, _ = batch.shape
        batch = self.normalization(batch)

        v = self.v_projection(batch).view(batch_size, sequence_length, hp.NUMBER_OF_HEADS, hp.HEAD_SIZE).transpose(1, 2)
        k = self.k_projection(batch).view(batch_size, sequence_length, hp.NUMBER_OF_HEADS, hp.HEAD_SIZE).transpose(1, 2)
        q = self.q_projection(batch).view(batch_size, sequence_length, hp.NUMBER_OF_HEADS, hp.HEAD_SIZE).transpose(1, 2)

        attention = flash_attn_func(q, k, v, hp.ATTENTION_DROPOUT_RATE,
                                    softmax_scale=1.0/math.sqrt(hp.HEAD_SIZE),
                                    causal=True)

        transposition = attention.transpose(1, 2).contiguous()
        recombination = transposition.view(batch_size, sequence_length, hp.HIDDEN_SIZE)

        out = self.out_projection(recombination)
        return out + residual


class FlashAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.stairs = nn.ModuleList([
            FlashAttentionLayer() for _ in range(hp.NUMBER_OF_ATTENTION_LAYERS)
        ])

    def forward(self, embeddings: Tensor) -> Tensor:
        hidden_states = embeddings
        for step in self.stairs:
            hidden_states = step(hidden_states)
        return hidden_states
