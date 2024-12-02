import torch
import torch.nn as nn
from torch import Tensor

import omega.transformer.cmd as cmd
import omega.transformer.hyperparameters as hp
import omega.transformer.layers as layers
from omega.transformer.tokenizer import PAD_TOKEN_ID
from omega.transformer.typing import Vector


class FactorizedEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=cmd.OMEGA_TOKENIZER_VOCABULARY_SIZE,
            embedding_dim=hp.BOTTLENECK_SIZE,
            padding_idx=PAD_TOKEN_ID
        )

        self.expansion = nn.Linear(
            in_features=hp.BOTTLENECK_SIZE,
            out_features=hp.HIDDEN_SIZE,
            bias=False  # Following PaLM example, we don't use bias.
        )

    def forward(self, input_ids: Vector) -> Tensor:
        compressed_embeddings = self.embeddings(input_ids)
        expanded_embeddings = self.expansion(compressed_embeddings)
        return expanded_embeddings


class LinearAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.v_projection = nn.Linear(hp.HIDDEN_SIZE, hp.HIDDEN_SIZE)
        self.k_projection = nn.Linear(hp.HIDDEN_SIZE, hp.HIDDEN_SIZE)
        self.q_projection = nn.Linear(hp.HIDDEN_SIZE, hp.HIDDEN_SIZE)

        self.kernel = layers.Highway(hp.HIDDEN_SIZE)
        self.out_projection = nn.Linear(hp.HIDDEN_SIZE, hp.HIDDEN_SIZE)

    def forward(self, batch: Tensor) -> Tensor:
        v = self.v_projection(batch)  # [batch_size, sequence_length, hidden_size]
        k = self.k_projection(batch)  # [batch_size, sequence_length, hidden_size]
        q = self.q_projection(batch)  # [batch_size, sequence_length, hidden_size]

        k = self.kernel(k)            # [batch_size, sequence_length, hidden_size]
        q = self.kernel(q)            # [batch_size, sequence_length, hidden_size]

        # Linear attention formula: φ(q) @ (φ(k) @ v)
        # This approximates softmax(QK^T)V while being linear in sequence length
        context = torch.bmm(k.transpose(1, 2), v)  # [batch_size, hidden_size, hidden_size]
        attention = torch.bmm(q, context)          # [batch_size, sequence_length, hidden_size]
        out = self.out_projection(attention)
        return out

