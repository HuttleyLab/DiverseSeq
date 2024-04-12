import warnings

from pathlib import Path
from typing import Optional, Union

from attrs import define
from cogent3 import make_seq, make_table, open_
from cogent3.app import typing as c3_types
from cogent3.app.composable import LOADER, define_app
from cogent3.app.data_store import (
    OVERWRITE,
    DataMember,
    DataStoreABC,
    DataStoreDirectory,
    Mode,
)
from cogent3.format.fasta import alignment_to_fasta
from cogent3.parse.fasta import MinimalFastaParser

from divergent import util as dv_utils
from divergent.data_store import HDF5DataStore
from divergent.record import SeqArray, SeqRecord


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


@define_app(app_type=LOADER)
def load_bytes(path: c3_types.IdentifierType) -> bytes:
    path = Path(path)
    return path.read_bytes()


@define_app(app_type=LOADER)
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


def get_seq_identifiers(paths, label_func=_label_func) -> list[filename_seqname]:
    loader = faster_load_fasta(label_func=label_func)
    records = []
    for path in paths:
        seq_names = list(loader(path))
        records.extend(filename_seqname(str(path), n) for n in seq_names)
    return records


@define_app(app_type=LOADER)
class concat_seqs:
    def __init__(
        self, label_func: callable = _label_from_filename, max_length: int | None = None
    ) -> None:
        self.label_func = label_func
        self.max_length = max_length
        self.loader = faster_load_fasta(label_func=label_func)

    def main(self, path: c3_types.IdentifierType) -> c3_types.SeqType:
        data = self.loader(path)
        seq = "-".join(data.values())

        if self.max_length:
            seq = seq[: self.max_length]

        return make_seq(seq, name=self.label_func(path), moltype="dna")


@define_app(app_type=LOADER)
class seqarray_from_fasta:
    def __init__(self, label_func=_label_func, max_length=None, moltype="dna") -> None:
        self.max_length = max_length
        self.loader = faster_load_fasta(label_func=label_func)
        self.moltype = moltype
        self.str2arr = dv_utils.str2arr(moltype=self.moltype)

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
        return SeqArray(
            seqid=name,
            data=self.str2arr(seq),
            moltype=self.moltype,
            source=getattr(identifier, "source", identifier),
        )


@define_app
class dvgt_seq_file_to_data_store:
    def __init__(
        self,
        dest: Optional[c3_types.IdentifierType] = None,
        limit: Optional[int] = None,
        mode: Union[str, Mode] = OVERWRITE,
    ):
        self.dest = dest
        self.limit = limit
        self.mode = mode
        self.loader = faster_load_fasta()

    def main(self, fasta_path: c3_types.IdentifierType) -> DataStoreABC:
        outpath = Path(self.dest) if self.dest else Path(fasta_path).with_suffix("")
        outpath.mkdir(parents=True, exist_ok=True)
        out_dstore = DataStoreDirectory(
            source=outpath, mode=self.mode, suffix=".fa", limit=self.limit
        )

        seqs = self.loader(fasta_path)

        for seq_id, seq_data in seqs.items():
            fasta_seq_data = alignment_to_fasta({seq_id: seq_data}, block_size=80)
            out_dstore.write(unique_id=seq_id, data=fasta_seq_data)

        return out_dstore


@define_app(app_type=LOADER)
class dvgt_load_seqs:
    def __init__(self, moltype: str = "dna"):
        """load fasta sequences from a data store

        Parameters
        ----------
        moltype
            molecular type

        Notes
        -----
        Assumes each fasta file contains a single sequence so only takes the first
        """
        self.moltype = moltype
        self.str2arr = dv_utils.str2arr(moltype=self.moltype)

    def main(self, data_member: DataMember) -> SeqArray:
        seq_path = Path(data_member.data_store.source) / data_member.unique_id

        with open_(seq_path) as infile:
            for _, seq in MinimalFastaParser(infile.read().splitlines()):
                seq.replace("-", "")

        return SeqArray(
            seqid=data_member.unique_id,
            data=self.str2arr(seq),
            moltype=self.moltype,
            source=data_member.data_store.source,
        )


@define_app(app_type=LOADER)
def dvgt_load_prepped_seqs(data_member: DataMember) -> SeqArray:
    """loads prepped sequences from a dvgtseqs data store"""
    seq = data_member.data_store.read(unique_id=data_member.unique_id)
    attrs = data_member.data_store.get_attrs(unique_id=data_member.unique_id)
    moltype = attrs.get("moltype", None)
    source = attrs.get("source", None)

    return SeqArray(
        seqid=data_member.unique_id, data=seq, moltype=moltype, source=source
    )


@define_app(app_type=LOADER)
def dvgt_load_records(dstore: HDF5DataStore) -> SeqArray:
    """loads prepped sequences from a dvgtseqs data store"""
    records = []
    for data_member in dstore.members:
        name = data_member.unique_id
        data = data_member.data_store.read(data_member.unique_id)
        attrs = data_member.data_store.get_attrs(unique_id=data_member.unique_id)
        length = int(attrs.get("length", None))

        records.append(SeqRecord(kcounts=data, name=name, length=length))
    return records
