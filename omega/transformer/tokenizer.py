import os

import torch
from torch import Tensor
import sentencepiece as spm

if os.environ.get("OMEGA_TOKENIZER_PATH", None) is not None:
    sp = spm.SentencePieceProcessor()
    sp.load(os.environ["OMEGA_TOKENIZER_PATH"])
elif os.environ.get("OMEGA_TOKENIZER_DATA", None) is not None:
    spm.SentencePieceTrainer.train(
        input=os.environ["OMEGA_TOKENIZER_DATA"],
        model_prefix='en-sp',
        model_type="bpe",
        vocab_size=10000,
    )
else:
    raise FileNotFoundError("Tokenizer model cannot be found.")


def tokenize(sequences: list[str]) -> Tensor:
    tokenized_embeddings = [torch.tensor(sp.encode(sequence)) for sequence in sequences]
    return torch.nn.utils.rnn.pad_sequence(tokenized_embeddings, batch_first=True, padding_value=sp.pad_id())
