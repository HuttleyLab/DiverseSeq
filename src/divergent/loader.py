import warnings

from pathlib import Path

from attrs import define
from cogent3 import make_seq, make_table, open_
from cogent3.app import composable
from cogent3.app import typing as c3_types
from cogent3.parse.fasta import MinimalFastaParser


def _label_func(label):
    return label.split()[0]


def _label_from_filename(path):
    return Path(path).stem.split(".")[0]


def load_tsv(path):
    with open_(path) as infile:
        data = infile.read().splitlines()
    header = data[0].strip().split("\t")
    data = [l.strip().split("\t") for l in data[1:]]
    return make_table(header, data=data)


@composable.define_app(app_type=composable.LOADER)
def load_bytes(path: c3_types.IdentifierType) -> bytes:
    path = Path(path)
    return path.read_bytes()


@composable.define_app(app_type=composable.LOADER)
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
            result[n] = s.replace("-", "")
        return result


@define
class filename_seqname:
    source: str
    name: str


@composable.define_app(app_type=composable.LOADER)
class seq_from_fasta:
    def __init__(self, label_func=_label_func, max_length=None, moltype="dna") -> None:
        self.max_length = max_length
        self.loader = faster_load_fasta(label_func=label_func)
        self.moltype = moltype

    def main(
        self, identifier: filename_seqname | c3_types.IdentifierType
    ) -> c3_types.SeqType:
        path = identifier.source if hasattr(identifier, "source") else identifier
        data = self.loader(path)
        if hasattr(identifier, "source"):
            name = identifier.name
            seq = data[name]
        else:
            name, seq = list(data.items())[0]

        if self.max_length:
            seq = seq[: self.max_length]

        seq = make_seq(seq, name=name, moltype=self.moltype)
        seq.source = getattr(identifier, "source", identifier)
        return seq


def get_seq_identifiers(paths, label_func=_label_func) -> list[filename_seqname]:
    loader = faster_load_fasta(label_func=label_func)
    records = []
    for path in paths:
        seq_names = list(loader(path))
        records.extend(filename_seqname(str(path), n) for n in seq_names)
    return records


@composable.define_app(app_type=composable.LOADER)
class concat_seqs:
    def __init__(
        self, label_func: callable = _label_from_filename, max_length: int | None = None
    ) -> None:
        self.label_func = label_func
        self.max_length = max_length
        self.loader = faster_load_fasta(label_func=label_func)

    def main(self, path: c3_types.IdentifierType) -> c3_types.SeqType:
        data = self.loader(path)
        seq = "N".join(data.values())

        if self.max_length:
            seq = seq[: self.max_length]

        return make_seq(seq, name=self.label_func(path), moltype="dna")


