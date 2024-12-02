import os
import multiprocessing

import torch
import datasets as ds
import sentencepiece as spm

import omega.transformer.cmd as cmd
from omega.transformer.typing import Vector, Matrix

PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3


def fit_tokenizer_model():
    datasets = cmd.OMEGA_HUGGINGFACE_DATASETS.split(":")
    with open(cmd.OMEGA_TOKENIZER_DATA, "w") as file:
        for dataset in datasets:
            data = ds.load_dataset(dataset, split=f"train", streaming=True, trust_remote_code=True)
            subset = data.take(cmd.OMEGA_HUGGINGFACE_CHUNKS)
            text_columns = [
                name for name, feature in subset.features.items()
                if str(feature).startswith("Value(dtype=\'string\'")
            ]
            for example in subset:
                text = " ".join(str(example[column]) for column in text_columns)
                file.write(text + "\n")

    spm.SentencePieceTrainer.Train(
        input=cmd.OMEGA_TOKENIZER_DATA,
        model_prefix=cmd.OMEGA_TOKENIZER_PATH,
        model_type=cmd.OMEGA_TOKENIZER_MODEL_TYPE,
        vocab_size=cmd.OMEGA_TOKENIZER_VOCABULARY_SIZE,
        pad_id=PAD_TOKEN_ID, unk_id=UNK_TOKEN_ID, bos_id=BOS_TOKEN_ID, eos_id=EOS_TOKEN_ID,
        pad_piece="[PAD]", unk_piece="[UNK]", bos_piece="[BOS]", eos_piece="[EOS]",
        character_coverage=1.0,
        num_threads=multiprocessing.cpu_count(),
        split_digits=True,
        byte_fallback=True,
        allow_whitespace_only_pieces=False,
        normalization_rule_name="nmt_nfkc"
    )


class OmegaTokenizer:
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(cmd.OMEGA_TOKENIZER_PATH + ".model")

    def encode(self, text: str) -> Vector:
        return torch.tensor(self.sp.Encode(text, add_bos=True, add_eos=False))

    def decode(self, output: Vector) -> str:
        return self.sp.Decode(output.cpu().tolist())

    def encode_many(self, texts: list[str]) -> Matrix:
        # Unfortunately, sentencepiece does not provide a simple or comprehensive way how to tokenize a batch of
        # sequences at the same time, thus we are forced to cast all of it into sequential list processing. We
        # encourage anyone for whom this may become an issue to employ a queue design pattern where while GPU is
        # processing a batch, CPU would be tokenizing, which is viable since sentencepiece is optimized in C++.
        sequences = [torch.tensor(self.sp.Encode(text, add_bos=True, add_eos=False))
                     for text in texts]

        return torch.nn.utils.rnn.pad_sequence(
            sequences,
            padding_value=PAD_TOKEN_ID,
            batch_first=True
        )

    def decode_many(self, output: Matrix) -> list[str]:
        return [self.decode(text) for text in output]


if cmd.OMEGA_TOKENIZER_DATA is not None and not os.path.exists(cmd.OMEGA_TOKENIZER_PATH + ".model"):
    fit_tokenizer_model()
