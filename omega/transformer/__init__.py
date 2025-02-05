import torch
import torch.nn as nn

import omega.transformer.cmd as cmd
import omega.transformer.hyperparameters as hp
from omega.transformer.brain import Brain
from omega.transformer.attention import FlashAttention
from omega.transformer.tokenizer import PAD_TOKEN_ID, OmegaTokenizer
from omega.transformer.typing import Matrix


class Omega(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(cmd.OMEGA_TOKENIZER_VOCABULARY_SIZE, hp.HIDDEN_SIZE, padding_idx=PAD_TOKEN_ID)
        self.attention = FlashAttention()
        self.brain = Brain()

    def forward(self, batched_token_ids: Matrix) -> Matrix:
        batched_token_embeddings = self.embeddings[batched_token_ids]
        last_hidden_states = self.attention(batched_token_embeddings)
        activations, conclusion = self.brain(last_hidden_states)



