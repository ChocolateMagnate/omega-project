import torch
import datasets
import multiprocessing
from torch import Tensor
import sentencepiece as spm

import omega.transformer.cmd as cmd
from omega.transformer.typing import Matrix

if cmd.OMEGA_TOKENIZER_DATA is not None:
    dataset = datasets.load_dataset(
        cmd.OMEGA_HUGGINGFACE_DATASET,
        split=f"train[:{cmd.OMEGA_HUGGINGFACE_CHUNKS}]",
        streaming=True,
    )
    dataset.to_json(cmd.OMEGA_TOKENIZER_DATA, lines=True)

    spm.SentencePieceTrainer.Train(
        input=cmd.OMEGA_TOKENIZER_DATA,
        model_type=cmd.OMEGA_TOKENIZER_MODEL_TYPE,
        vocab_size=cmd.OMEGA_TOKENIZER_VOCABULARY_SIZE,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="[PAD]", unk_piece="[UNK]",  bos_piece="[BOS]", eos_piece="[EOS]",
        character_coverage=1.0,
        num_threads = multiprocessing.cpu_count(),
        split_digits=True,
        byte_fallback=True,
        allow_whitespace_only_pieces=False,
        normalization_rule_name="nmt_nfkc"
    )


class OmegaTokenizer:
    def __init__(self, tag: str = "latest"):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(cmd.OMEGA_TOKENIZER_PATH + ":" + tag)

    def encode(self, texts: list[str]) -> Tensor:
        return torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(self.sp.Encode(text)) for text in texts],
            batch_first=True,
            padding_value=self.sp.pad_id()
        )

    def decode(self, output: Matrix) -> list[str]:
        return [self.sp.Decode(batch) for batch in output]
