import pickle

from typing import Any

import blosc2

from cogent3 import get_moltype
from cogent3.app import composable
from numpy import array, ndarray, uint8


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

    def main(self, data: "vector") -> tuple[bytes, str]:
        source = data.source
        return self.prep(data), source


@composable.define_app
def unpickle_data(data: bytes) -> Any:
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
