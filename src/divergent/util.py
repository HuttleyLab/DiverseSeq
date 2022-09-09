import pickle

from pathlib import Path
from typing import Any, Union

import blosc2

from cogent3 import make_seq, open_
from cogent3.app import composable
from cogent3.app import typing as c3_types
from cogent3.parse.fasta import MinimalFastaParser


def _label_func(label):
    return label.split()[0]


@composable.define_app(app_type=composable.LOADER)
def load_bytes(path: c3_types.IdentifierType) -> bytes:
    path = Path(path)
    return path.read_bytes()


@composable.define_app
def blosc_decompress(data: bytes) -> bytes:
    return blosc2.decompress(data)


@composable.define_app
def blosc_compress(data: bytes) -> bytes:
    return blosc2.compress(
        data,
        shuffle=blosc2.Filter.BITSHUFFLE,
    )


@composable.define_app
def pickle_data(data: Any) -> bytes:
    return pickle.dumps(data)


@composable.define_app
def unpickle_data(data: bytes) -> bytes:
    return pickle.loads(data)


@composable.define_app(app_type="loader")
class seq_from_fasta:
    def __init__(self, label_func=_label_func, max_length=None) -> None:
        self.label_func = label_func
        self.max_length = max_length

    def main(self, path: c3_types.IdentifierType) -> c3_types.SeqType:
        with open_(path) as infile:
            data = list(iter(MinimalFastaParser(infile.read().splitlines())))
            assert len(data) == 1
            name, seq = data[0]
            name = self.label_func(name)

        if self.max_length:
            seq = seq[: self.max_length]

        seq = make_seq(seq, name=name, moltype="dna")
        seq.source = str(path)
        return seq
