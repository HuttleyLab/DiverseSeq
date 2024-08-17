import contextlib
import pickle
import re

import blosc2
from cogent3 import get_moltype
from cogent3.app import composable
from cogent3.app import typing as c3_types
from numpy import array, ndarray, uint8


@composable.define_app
def blosc_decompress(data: bytes) -> bytes:
    return blosc2.decompress(data)


@composable.define_app
def blosc_compress(data: bytes) -> bytes:
    return blosc2.compress(
        data,
        filter=blosc2.Filter.BITSHUFFLE,
    )


@composable.define_app
def pickle_data(data: c3_types.SerialisableType) -> bytes:
    return pickle.dumps(data)


@composable.define_app
def unpickle_data(data: bytes) -> c3_types.SerialisableType:
    return pickle.loads(data)


@composable.define_app
class str2arr:
    """convert string to array of uint8"""

    def __init__(self, moltype: str = "dna", max_length=None):
        moltype = get_moltype(moltype)
        self.canonical = "".join(moltype)
        self.max_length = max_length
        extended = "".join(list(moltype.alphabets.degen))
        self.translation = b"".maketrans(
            extended.encode("utf8"),
            "".join(chr(i) for i in range(len(extended))).encode("utf8"),
        )

    def main(self, data: str) -> ndarray:
        if self.max_length:
            data = data[: self.max_length]

        b = data.encode("utf8").translate(self.translation)
        return array(memoryview(bytearray(b)), dtype=uint8)


@composable.define_app
class arr2str:
    """convert array of uint8 to str"""

    def __init__(self, moltype: str = "dna", max_length: int = None):
        moltype = get_moltype(moltype)
        self.canonical = "".join(moltype)
        self.max_length = max_length
        extended = "".join(list(moltype.alphabets.degen))
        self.translation = b"".maketrans(
            "".join(chr(i) for i in range(len(extended))).encode("utf8"),
            extended.encode("utf8"),
        )

    def main(self, data: ndarray) -> str:
        if self.max_length:
            data = data[: self.max_length]

        b = data.tobytes().translate(self.translation)
        return bytearray(b).decode("utf8")


@contextlib.contextmanager
def fake_wake(*args, **kwargs):
    yield


# we allow for file suffixes to include compression extensions
_fasta_format = re.compile("(fasta|mfa|faa|fna|fa)([.][a-zA-Z]+)?$")
_genbank_format = re.compile("(genbank|gbk|gb|gbff)([.][a-zA-Z]+)?$")


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
