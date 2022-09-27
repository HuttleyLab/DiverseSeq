import pickle
import warnings

from pathlib import Path
from typing import Any, Union

import blosc2

from attrs import define
from cogent3 import make_seq, make_table, open_
from cogent3.app import composable
from cogent3.app import typing as c3_types
from cogent3.parse.fasta import MinimalFastaParser


def _label_func(label):
    return label.split()[0]


@composable.define_app
def faster_load_fasta(path: c3_types.IdentifierType, label_func=_label_func) -> dict:
    with open_(path) as infile:
        result = {}
        for n, s in MinimalFastaParser(infile.read().splitlines()):
            n = label_func(n)
            if n in result and result[n] != s:
                warnings.warn(
                    f"duplicated seq label {n!r} in {path}, but different seqs",
                    UserWarning,
                )
            result[n] = s
        return result


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
class bundle_data:
    def __init__(self):
        self.prep = pickle_data() + blosc_compress()

    T = Union["sparse_vector", "unique_kmers"]

    def main(self, data: T) -> tuple[bytes, str]:
        source = data.source
        return self.prep(data), source


@composable.define_app
def unpickle_data(data: bytes) -> Any:
    return pickle.loads(data)


@define
class filename_seqname:
    source: str
    name: str


@composable.define_app(app_type="loader")
class seq_from_fasta:
    def __init__(self, label_func=_label_func, max_length=None) -> None:
        self.max_length = max_length
        self.loader = faster_load_fasta(label_func=label_func)

    def main(self, identifier: filename_seqname) -> c3_types.SeqType:
        name = identifier.name
        data = self.loader(identifier.source)
        seq = data[name]

        if self.max_length:
            seq = seq[: self.max_length]

        seq = make_seq(seq, name=name, moltype="dna")
        seq.source = identifier.source
        return seq


def get_seq_identifiers(paths, label_func=_label_func) -> list[filename_seqname]:
    loader = faster_load_fasta(label_func=label_func)
    records = []
    for path in paths:
        seq_names = list(loader(path))
        records.extend(filename_seqname(str(path), n) for n in seq_names)
    return records


