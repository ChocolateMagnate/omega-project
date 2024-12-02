import os
import multiprocessing

import torch
import sentencepiece as spm
import datasets as ds

import omega.transformer.cmd as cmd
from omega.transformer.typing import Vector, Matrix


def fit_tokenizer_model():
    datum = []
    datasets = cmd.OMEGA_HUGGINGFACE_DATASETS.split(":")
    for dataset in datasets:
        data = ds.load_dataset(
            dataset,
            split=f"train[:{cmd.OMEGA_HUGGINGFACE_CHUNKS}]",
            streaming=True,
        )
        datum.append(data)
    mixed_dataset = ds.concatenate_datasets(datum)
    mixed_dataset.to_json(cmd.OMEGA_TOKENIZER_DATA, lines=True)

    spm.SentencePieceTrainer.Train(
        input=cmd.OMEGA_TOKENIZER_DATA,
        model_type=cmd.OMEGA_TOKENIZER_MODEL_TYPE,
        vocab_size=cmd.OMEGA_TOKENIZER_VOCABULARY_SIZE,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="[PAD]", unk_piece="[UNK]", bos_piece="[BOS]", eos_piece="[EOS]",
        character_coverage=1.0,
        num_threads=multiprocessing.cpu_count(),
        split_digits=True,
        byte_fallback=True,
        allow_whitespace_only_pieces=False,
        normalization_rule_name="nmt_nfkc"
    )


class OmegaTokenizer:
    def __init__(self, tag: str = "latest"):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(cmd.OMEGA_TOKENIZER_PATH + ":" + tag)

    def encode(self, text: str) -> Matrix:
        return torch.nn.utils.rnn.pad_sequence(
            torch.tensor(self.sp.Encode(text)),
            batch_first=True,
            padding_value=self.sp.pad_id()
        )

    def decode(self, output: Vector) -> str:
        return self.sp.Decode(output)


if cmd.OMEGA_TOKENIZER_DATA is not None and not os.path.exists(cmd.OMEGA_TOKENIZER_PATH):
    fit_tokenizer_model()
