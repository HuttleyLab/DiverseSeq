import contextlib
import functools
import os
import pathlib
import re
import sys
import typing

import numpy
from cogent3 import get_moltype
from cogent3.app import composable
from cogent3.app import typing as c3_types
from rich import text as rich_text

from diverse_seq import _dvs as dvs

try:
    from wakepy.keep import running as keep_running

    # trap flaky behaviour on linux
    with keep_running():
        ...

except (NotImplementedError, ImportError):
    keep_running = contextlib.nullcontext

if typing.TYPE_CHECKING:
    from click.core import Context, Option


@composable.define_app
class str2arr:
    """convert string to array of uint8"""

    def __init__(self, moltype: str = "dna", max_length: int | None = None) -> None:
        self.max_length = max_length
        mt = get_moltype(moltype)
        self.alphabet = mt.most_degen_alphabet()

    def main(self, data: str) -> numpy.ndarray:
        if self.max_length:
            data = data[: self.max_length]

        return self.alphabet.to_indices(data)


@composable.define_app
class arr2str:
    """convert array of uint8 to str"""

    def __init__(self, moltype: str = "dna", max_length: int | None = None) -> None:
        self.max_length = max_length
        mt = get_moltype(moltype)
        self.alphabet = mt.most_degen_alphabet()

    def main(self, data: numpy.ndarray) -> str:
        if self.max_length:
            data = data[: self.max_length]

        return self.alphabet.from_indices(data)


# we allow for file suffixes to include compression extensions
_fasta_format = re.compile("(fasta|mfa|faa|fna|fa)([.][a-zA-Z0-9]+)?$")
_genbank_format = re.compile("(genbank|gbk|gb|gbff)([.][a-zA-Z0-9]+)?$")


def get_seq_file_format(suffix: str) -> str:
    """returns format string

    Notes
    -----
    Based on cogent3 suffixes, returns either 'fasta' or 'genbank'
    or None.
    """
    if _fasta_format.match(suffix):
        return "fasta"
    return "genbank" if _genbank_format.match(suffix) else None


def determine_chunk_size(total_items, num_chunks):
    """chunk sizes for distributing items into approximately equal chunks"""
    base_chunk_size = total_items // num_chunks
    remainder = total_items % num_chunks

    return [
        base_chunk_size + 1 if i < remainder else base_chunk_size
        for i in range(num_chunks)
    ]


def chunked(iterable, num_chunks, verbose=False):
    sizes = determine_chunk_size(len(iterable), num_chunks)
    if verbose:
        print(f"chunk sizes: {sizes}")

    cum_sizes = numpy.array(sizes).cumsum()
    starts = [0] + cum_sizes[:-1].tolist()
    start_stop = numpy.array([starts, cum_sizes]).T
    for start, end in start_stop:
        yield iterable[start:end]


def _comma_sep_or_file(
    ctx: "Context",  # noqa: ARG001
    param: "Option",  # noqa: ARG001
    include: str,
) -> list[str] | None:
    if include is None:
        return None
    if pathlib.Path(include).is_file():
        path = pathlib.Path(include)
        names = path.read_text().splitlines()
        return [name.strip() for name in names]
    return [n.strip() for n in include.split(",") if n.strip()]


def _hide_progress(
    ctx: "Context",  # noqa: ARG001
    param: "Option",  # noqa: ARG001
    hide_progress: str,
) -> bool:
    return True if "DVS_HIDE_PROGRESS" in os.environ else hide_progress


def _check_dstore(
    ctx: "Context",  # noqa: ARG001
    param: "Option",  # noqa: ARG001
    path: pathlib.Path,
) -> pathlib.Path:
    """makes sure minimum number of unique sequences are in the store"""
    from diverse_seq import _dvs as dvs

    store = dvs.make_zarr_store(str(path))
    seqids = list(store.unique_seqids)
    min_num = 5
    if len(seqids) >= min_num:
        return path

    msg = f"SKIPPING: '{path}' does not have ≥{min_num} sequences!"
    print_colour(msg, "red")
    sys.exit(1)


class _printer:
    from rich.console import Console

    def __init__(self) -> None:
        self._console = self.Console()

    def __call__(self, txt: str, colour: str):
        """print text in colour"""
        msg = rich_text.Text(txt)
        msg.stylize(colour)
        self._console.print(msg)


def get_sample_data_path() -> pathlib.Path:
    """returns path to sample data file"""
    from diverse_seq import data

    path = pathlib.Path(data.__path__[0]) / "brca1.fa"

    path = path.absolute()
    if not path.exists():
        msg = f"sample data file {str(path)!r} does not exist"
        raise ValueError(msg)

    return pathlib.Path(path)


print_colour = _printer()


def populate_inmem_zstore(seqcoll: c3_types.SeqsCollectionType) -> dvs.ZarrStoreWrapper:
    """returns an in-memory ZarrStoreWrapper populated with sequences from seqcoll"""
    degapped = seqcoll.degap()
    # make an in-memory ZarrStore
    zstore = dvs.make_zarr_store()
    for seq in degapped.seqs:
        arr = numpy.array(seq)
        zstore.write(seq.name, arr.tobytes())
    return zstore


@functools.cache
def _get_canonical_states(moltype: str) -> bytes:
    mt = get_moltype(moltype)
    canonical = "".join(mt.alphabet)
    v = mt.alphabet.to_indices(canonical)
    if not (0 <= min(v) < max(v) < len(canonical)):
        msg = f"indices of canonical states {canonical} not sequential {v}"
        raise ValueError(msg)
    return canonical.encode("utf8")
