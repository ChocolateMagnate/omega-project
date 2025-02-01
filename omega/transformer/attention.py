import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import omega.transformer.hyperparameters as hp
from omega.transformer.typing import Vector


@contextlib.contextmanager
def best_attention_backend(vector: Vector):
    is_flash_attention_available = (
        torch.cuda.is_available() and
        vector.dtype in [torch.float16, torch.bfloat16] and
        all(dim % 8 == 0 for dim in vector.shape[-2:]) and
        vector.is_cuda
    )

    if is_flash_attention_available:
        with nn.attention.sdpa_kernel(enable_flash=True):
            yield
    else:
        with nn.attention.sdpa_kernel(enable_mem_efficient=True):
            yield


class FlashAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalization = nn.LayerNorm(hp.HIDDEN_SIZE)
        self.v_projection = nn.Linear(hp.HIDDEN_SIZE, hp.HIDDEN_SIZE)
        self.k_projection = nn.Linear(hp.HIDDEN_SIZE, hp.HIDDEN_SIZE)
        self.q_projection = nn.Linear(hp.HIDDEN_SIZE, hp.HIDDEN_SIZE)
        self.out_projection = nn.Linear(hp.HIDDEN_SIZE, hp.HIDDEN_SIZE)
        self.dropout = nn.Dropout(hp.ATTENTION_DROPOUT_RATE)

    def forward(self, batch: Tensor) -> Tensor:
        assert batch.dim() == 3, f"Expected 3D tensor, got {batch.dim()}D"
        assert batch.size(-1) == hp.HIDDEN_SIZE, f"Expected hidden size {hp.HIDDEN_SIZE}, got {batch.size(-1)}"

        residual = batch
        batch_size, sequence_length, _ = batch.shape
        batch = self.normalization(batch)

        v = self.v_projection(batch).view(batch_size, sequence_length, hp.NUMBER_OF_HEADS, hp.HEAD_SIZE).transpose(1, 2)
        k = self.k_projection(batch).view(batch_size, sequence_length, hp.NUMBER_OF_HEADS, hp.HEAD_SIZE).transpose(1, 2)
        q = self.q_projection(batch).view(batch_size, sequence_length, hp.NUMBER_OF_HEADS, hp.HEAD_SIZE).transpose(1, 2)

        with best_attention_backend(q):
            # Every day, I pray to Lord for blessing PyTorch with this function.
            attentions = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        transposition = attentions.transpose(1, 2).contiguous()
        recombination = transposition.view(batch_size, sequence_length, hp.HIDDEN_SIZE)

        out = self.out_projection(recombination)
        return self.dropout(out + residual)


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
