import dataclasses
import functools
import string
import typing
from pathlib import Path

from attrs import define
from cogent3.app import typing as c3_types
from cogent3.app.composable import (
    LOADER,
    NON_COMPOSABLE,
    WRITER,
    define_app,
    source_proxy,
)
from cogent3.app.data_store import (
    OVERWRITE,
    DataMember,
    DataStoreABC,
    DataStoreDirectory,
    Mode,
)
from cogent3.core import alphabet as c3_alpha
from cogent3.format import fasta as format_fasta
from cogent3.parse import fasta, genbank
from numpy import ndarray

from diverse_seq import util as dvs_utils

converter_fasta = c3_alpha.convert_alphabet(
    string.ascii_lowercase.encode("utf8"),
    string.ascii_uppercase.encode("utf8"),
    delete=b"\n\r\t- ",
)

converter_genbank = c3_alpha.convert_alphabet(
    string.ascii_lowercase.encode("utf8"),
    string.ascii_uppercase.encode("utf8"),
    delete=b"\n\r\t- 0123456789",
)


@define
class filename_seqname:
    source: str
    name: str


def get_format_parser(seq_path: str | Path, seq_format: str) -> typing.Iterable:
    return (
        fasta.iter_fasta_records(seq_path, converter=converter_fasta)
        if seq_format == "fasta"
        else genbank.iter_genbank_records(
            seq_path,
            converter=converter_genbank,
            convert_features=None,
        )
    )


@dataclasses.dataclass(frozen=True)
class SeqArray:
    """A SeqArray stores an array of indices that map to the canonical characters
    of the moltype of the original sequence."""

    seqid: str
    data: ndarray
    moltype: str
    source: str = None

    def __len__(self) -> int:
        return len(self.data)


@define_app(app_type=LOADER)
class dvs_load_seqs:
    """Load and proprocess sequences from seq datastore"""

    def __init__(self, moltype: str = "dna", seq_format: str = "fasta") -> None:
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
        self.str2arr = dvs_utils.str2arr(moltype=self.moltype)
        self.seq_format = seq_format

    def main(self, data_member: DataMember) -> SeqArray:
        seq_path = Path(data_member.data_store.source) / data_member.unique_id
        parser = get_format_parser(seq_path, self.seq_format)
        seqs = [s for _, s, *_ in parser]
        return SeqArray(
            seqid=data_member.unique_id,
            data=self.str2arr(b"-".join(seqs).decode("utf8")),  # pylint: disable=not-callable
            moltype=self.moltype,
            source=data_member.data_store.source,
        )


@functools.singledispatch
def get_unique_id(val: typing.Any) -> str | None:
    for attr in ("unique_id", "seqid", "name", "source"):
        if result := getattr(val, attr, None):
            return get_unique_id(result)
    return None


@get_unique_id.register
def _get_unique_id_path(val: Path) -> str | None:
    # have to intercept Path objects as they have a name attribute
    return val.with_suffix("").name


@get_unique_id.register
def _get_unique_id_str(val: str) -> str | None:
    # have to intercept Path objects as they have a name attribute
    return get_unique_id(Path(val))


@get_unique_id.register
def _get_unique_id_seqarray(val: SeqArray) -> str | None:
    return get_unique_id(val.seqid)


@define_app(app_type=WRITER)
class dvs_write_seqs:
    """Write seqs as numpy arrays"""

    def __init__(
        self,
        data_store,
        id_from_source: callable = get_unique_id,
    ):
        self.data_store = data_store
        self.id_from_source = id_from_source

    def main(
        self,
        data: SeqArray,
        identifier: str | None = None,
    ) -> c3_types.IdentifierType:
        data = data.obj if isinstance(data, source_proxy) else data
        unique_id = identifier or self.id_from_source(data)
        metadata = ({"source": str(data.source)} if data.source else {}) | {
            "moltype": data.moltype
        }
        return self.data_store.write(
            unique_id,
            data.data.tobytes(),
            metadata=metadata,
        )


@define_app(app_type=NON_COMPOSABLE)
class dvs_file_to_dir:
    """Convert a single sequence file into a directory data store"""

    def __init__(
        self,
        dest: c3_types.IdentifierType | None = None,
        seq_format: str = "fasta",
        limit: int | None = None,
        mode: str | Mode = OVERWRITE,
    ):
        """
        Parameters
        ----------
        dest
            name for the output directory, defaults to name of the input file
        limit
            the number of sequences to be written defaults to all
        mode
            directory creation mode, by default OVERWRITE
        """
        self.dest = dest
        self.limit = limit
        self.mode = mode
        self.seq_format = seq_format

    def main(self, seq_path: c3_types.IdentifierType) -> DataStoreABC:
        outpath = Path(self.dest) if self.dest else Path(seq_path).with_suffix("")
        outpath.mkdir(parents=True, exist_ok=True)
        out_dstore = DataStoreDirectory(
            source=outpath,
            mode=self.mode,
            suffix=".fa",
            limit=self.limit,
        )

        parser = get_format_parser(seq_path, self.seq_format)
        seqs = {n: seq.replace(b"-", b"") for n, seq, *_ in parser}

        for seq_id, seq_data in seqs.items():
            fasta_seq_data = format_fasta.seqs_to_fasta(
                {seq_id: seq_data.decode("utf8")},
                block_size=1_000_000_000,
            )
            out_dstore.write(unique_id=seq_id, data=fasta_seq_data)

        return out_dstore
