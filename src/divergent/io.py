import string
import warnings
from pathlib import Path

from attrs import define
from cogent3.app import typing as c3_types
from cogent3.app.composable import LOADER, WRITER, define_app
from cogent3.app.data_store import (
    OVERWRITE,
    DataMember,
    DataStoreABC,
    DataStoreDirectory,
    Mode,
    get_unique_id,
)
from cogent3.core import new_alphabet
from cogent3.format.fasta import seqs_to_fasta
from cogent3.parse.fasta import iter_fasta_records

from divergent import util as dv_utils
from divergent.data_store import HDF5DataStore
from divergent.record import SeqArray

converter = new_alphabet.convert_alphabet(
    string.ascii_lowercase.encode("utf8"),
    string.ascii_uppercase.encode("utf8"),
    delete=b"\n\r\\l\t- ",
)


def _label_func(label):
    return label.split()[0]


def _label_from_filename(path):
    return Path(path).stem.split(".")[0]


@define_app(app_type=LOADER)
def faster_load_fasta(path: c3_types.IdentifierType, label_func=_label_func) -> dict:
    result = {}
    for n, s in iter_fasta_records(path, converter=converter):
        n = label_func(n)
        if n in result and result[n] != s:
            warnings.warn(
                f"duplicated seq label {n!r} in {path}, but different seqs",
                UserWarning,
            )
        result[n] = s.decode("utf8")
    return result


@define
class filename_seqname:
    source: str
    name: str


@define_app(app_type=LOADER)
class dvgt_load_seqs:
    """Load and proprocess sequences from seq datastore"""

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
        for n, seq in iter_fasta_records(seq_path, converter=converter):
            break

        return SeqArray(
            seqid=data_member.unique_id,
            data=self.str2arr(seq.decode("utf8")),
            moltype=self.moltype,
            source=data_member.data_store.source,
        )


@define_app(app_type=WRITER)
class dvgt_write_prepped_seqs:
    """Write preprocessed seqs to a dvgtseq datastore"""

    def __init__(
        self,
        dest: c3_types.IdentifierType,
        limit: int = None,
        id_from_source: callable = get_unique_id,
    ):
        self.dest = dest
        self.data_store = HDF5DataStore(self.dest, limit=limit)
        self.id_from_source = id_from_source

    def main(
        self,
        data: SeqArray,
        identifier: str | None = None,
    ) -> c3_types.IdentifierType:
        unique_id = identifier or self.id_from_source(data.unique_id)
        return self.data_store.write(
            unique_id=unique_id,
            data=data.data,
            moltype=data.moltype,
            source=str(data.source),
        )


@define_app(app_type=WRITER)
class dvgt_write_seq_store:
    """Write a seq datastore with data from a single fasta file"""

    def __init__(
        self,
        dest: c3_types.IdentifierType | None = None,
        limit: int | None = None,
        mode: str | Mode = OVERWRITE,
    ):
        self.dest = dest
        self.limit = limit
        self.mode = mode
        self.loader = faster_load_fasta()

    def main(self, fasta_path: c3_types.IdentifierType) -> DataStoreABC:
        outpath = Path(self.dest) if self.dest else Path(fasta_path).with_suffix("")
        outpath.mkdir(parents=True, exist_ok=True)
        out_dstore = DataStoreDirectory(
            source=outpath,
            mode=self.mode,
            suffix=".fa",
            limit=self.limit,
        )

        seqs = self.loader(fasta_path)

        for seq_id, seq_data in seqs.items():
            fasta_seq_data = seqs_to_fasta({seq_id: seq_data}, block_size=80)
            out_dstore.write(unique_id=seq_id, data=fasta_seq_data)

        return out_dstore
