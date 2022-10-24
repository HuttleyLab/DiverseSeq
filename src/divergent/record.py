"""defines basic data type for storing an individual sequence record"""
import contextlib

from collections.abc import MutableSequence
from functools import lru_cache, singledispatch
from math import fabs
from typing import Dict, Optional, Union

import numba

from attrs import asdict, define, field, validators
from cogent3 import get_moltype
from cogent3.app import composable
from cogent3.app import typing as c3_types
from cogent3.core.alphabet import get_array_type
from numpy import array
from numpy import divmod as np_divmod
from numpy import isclose as np_isclose
from numpy import log2, ndarray, uint8, uint64
from numpy import unique as np_unique
from numpy import zeros

from divergent import util as dv_utils


NumType = Union[float, int]
PosDictType = Dict[int, NumType]


def _gettype(name):
    import numpy

    if name[-1].isdigit():
        return getattr(numpy, name)
    else:
        return {"int": int, "float": float}[name]


@define(slots=True)
class unique_kmers:
    data: ndarray
    num_states: int
    k: int
    source: str = None
    name: str = None

    def __init__(
        self,
        *,
        num_states: int,
        k: int,
        data: ndarray = None,
        source: str = None,
        name: str = None,
    ):
        """

        Parameters
        ----------
        num_states
            num_states
        k
            length of k-mer
        data
            dict of {k-mer index: NumType}
        """
        self.num_states = num_states
        self.k = k
        self.data = array(
            [] if data is None else data, dtype=get_array_type(num_states ** k)
        )
        self.source = source
        self.name = name

    def __len__(self):
        return len(self.data)

    def __getstate__(self):
        return asdict(self)

    def __setstate__(self, data):
        for k, v in data.items():
            setattr(self, k, v)
        return self

    def to_kmer_strings(self, states: str) -> list[str]:
        with contextlib.suppress(AttributeError):
            states = states.encode("utf8")
        return indices_to_seqs(self.data, states, self.k)


@define(slots=True)
class sparse_vector(MutableSequence):
    data: PosDictType
    vector_length: int
    default: Optional[NumType] = field(init=False)
    dtype: type = float
    source: str = None
    name: str = None

    def __init__(
        self,
        *,
        vector_length: int,
        data: PosDictType = None,
        dtype: type = float,
        source: str = None,
        name: str = None,
    ):
        """

        Parameters
        ----------
        vector_length
            num_states**k
        data
            dict of {k-mer index: NumType}
        dtype
        """
        self.vector_length = vector_length
        self.dtype = dtype

        data = data or {}
        self.default = dtype(0)
        self.data = {i: n for i, n in data.items() if not np_isclose(n, 0)}
        self.source = source
        self.name = name

    def __setitem__(self, key: int, value: NumType):
        if np_isclose(value, 0):
            return
        try:
            key = key.item()
        except AttributeError:
            pass
        self.data[key] = self.dtype(value)

    def __getitem__(self, key: int) -> NumType:
        return self.data.get(key, self.default)

    def __delitem__(self, key: int):
        try:
            del self.data[key]
        except KeyError:
            pass

    def __len__(self) -> int:
        return self.vector_length

    def __iter__(self) -> NumType:
        yield from (self[i] for i in range(len(self)))

    def __getstate__(self):
        return asdict(self)

    def __setstate__(self, data):
        for k, v in data.items():
            setattr(self, k, v)
        return self

    def _sub_vector(self, data, other) -> dict:
        assert len(self) == len(other)
        for pos in other.data.keys() & data.keys():
            val = self[pos] - other[pos]
            val = 0 if np_isclose(val, 0) else val
            data[pos] = val
        return data

    def _sub_scalar(self, data, scalar) -> dict:
        scalar = self.dtype(scalar)
        for pos, num in data.items():
            val = data[pos] - scalar
            val = 0 if np_isclose(val, 0) else val
            data[pos] = val

        return data

    def __sub__(self, other):
        func = (
            self._sub_vector if isinstance(other, self.__class__) else self._sub_scalar
        )
        data = func({**self.data}, other)
        return self.__class__(
            data=data, vector_length=self.vector_length, dtype=self.dtype
        )

    def __isub__(self, other):
        func = (
            self._sub_vector if isinstance(other, self.__class__) else self._sub_scalar
        )
        self.data = func(self.data, other)
        return self

    def _add_vector(self, data, other) -> dict:
        assert len(self) == len(other)
        for pos, num in other.data.items():
            data[pos] = self[pos] + num
        return data

    def _add_scalar(self, data, scalar) -> dict:
        scalar = self.dtype(scalar)
        for pos, num in data.items():
            data[pos] += scalar
        return data

    def __add__(self, other):
        # we are creating a new instance
        func = (
            self._add_vector if isinstance(other, self.__class__) else self._add_scalar
        )
        data = func({**self.data}, other)
        return self.__class__(
            data=data, vector_length=self.vector_length, dtype=self.dtype
        )

    def __iadd__(self, other):
        func = (
            self._add_vector if isinstance(other, self.__class__) else self._add_scalar
        )
        self.data = func(self.data, other)
        return self

    def _truediv_vector(self, data, other) -> dict:
        assert len(self) == len(other)
        for pos, num in other.data.items():
            data[pos] = self[pos] / num
        return data

    def _truediv_scalar(self, data, scalar) -> dict:
        scalar = self.dtype(scalar)
        for pos, num in data.items():
            data[pos] = self[pos] / scalar
        return data

    def __truediv__(self, other):
        func = (
            self._truediv_vector
            if isinstance(other, self.__class__)
            else self._truediv_scalar
        )
        # we are creating a new instance
        data = func({**self.data}, other)
        return self.__class__(data=data, vector_length=self.vector_length, dtype=float)

    def __itruediv__(self, other):
        func = (
            self._truediv_vector
            if isinstance(other, self.__class__)
            else self._truediv_scalar
        )
        self.data = func(self.data, other)
        self.dtype = float
        return self

    def insert(self, index: int, value: NumType) -> None:
        self[index] = value

    def sum(self):
        return sum(self.data.values())

    @property
    def array(self) -> ndarray:
        arr = zeros(len(self), dtype=self.dtype)

        for index, val in self.data.items():
            arr[index] = val
        return arr

    def iter_nonzero(self) -> NumType:
        yield from (v for _, v in sorted(self.data.items()))

    @property
    def entropy(self):
        kfreqs = array(list(self.iter_nonzero()))
        # taking absolute value due to precision issues
        return fabs(-(kfreqs * log2(kfreqs)).sum())

    def to_rich_dict(self):
        data = asdict(self)
        data["dtype"] = data["dtype"].__name__
        data.pop("default")
        # convert coords to str so orjson copes
        data["data"] = list(data["data"].items())
        return data

    @classmethod
    def from_dict(cls, data):
        data["data"] = dict(data["data"])
        data["dtype"] = _gettype(data["dtype"])
        return cls(**data)


@numba.jit
def coord_conversion_coeffs(num_states, k):
    """coefficients for multi-dimensional coordinate conversion into 1D index"""
    return array([num_states ** (i - 1) for i in range(k, 0, -1)])


@numba.jit
def coord_to_index(coord, coeffs):
    """converts a multi-dimensional coordinate into a 1D index"""
    return (coord * coeffs).sum()


@numba.jit
def index_to_coord(index, coeffs):
    """converts a 1D index into a multi-dimensional coordinate"""
    ndim = len(coeffs)
    coord = zeros(ndim, dtype=uint64)
    remainder = index
    for i in range(ndim):
        n, remainder = np_divmod(remainder, coeffs[i])
        coord[i] = n
    return coord


def indices_to_seqs(indices: ndarray, states: bytes, k: int) -> list[str]:
    """convert indices from k-dim into sequence

        Parameters
    ----------
    indices
        array of (len(states), k) dimensioned indices
    states
        the ordered characters, e.g. b"TCAG"
    """
    arr = indices_to_bytes(indices, states, k)
    return [bytearray(kmer).decode("utf8") for kmer in arr]


@numba.jit
def indices_to_bytes(indices: ndarray, states: bytes, k: int) -> ndarray:
    """convert indices from k-dim into bytes

        Parameters
    ----------
    indices
        array of (len(states), k) dimensioned indices
    states
        the ordered characters, e.g. b"TCAG"
    k
        dimensions

    Raises
    -----
    IndexError if an index not in states

    Returns
    -------
    uint8 array.

    Notes
    -----
    Use indices_to_seqs to have the results returned as strings
    """
    result = zeros((len(indices), k), dtype=uint8)
    coeffs = coord_conversion_coeffs(len(states), k)
    num_states = len(states)
    for i in range(len(indices)):
        index = indices[i]
        coord = index_to_coord(index, coeffs)
        for j in range(k):
            if coord[j] >= num_states:
                raise IndexError("index out of character range")
            result[i][j] = states[coord[j]]

    return result


@numba.jit
def kmer_indices(seq: ndarray, result: ndarray, num_states: int, k: int) -> ndarray:
    """return 1D indices for valid k-mers

    Parameters
    ----------
    seq
        numpy array of uint, assumed that canonical characters have
        sequential indexes which are all < num_states
    result
        array to be written into
    num_states
        defines range of possible ints at a position
    k
        k-mer size

    Returns
    -------
    result
    """
    coeffs = coord_conversion_coeffs(num_states, k)
    skip_until = 0
    for i in range(k):
        if seq[i] >= num_states:
            skip_until = i + 1

    result_idx = 0
    for i in range(len(result)):
        if seq[i + k - 1] >= num_states:
            skip_until = i + k

        if i < skip_until:
            continue

        index = (seq[i : i + k] * coeffs).sum()
        result[result_idx] = index
        result_idx += 1

    return result[:result_idx]


@numba.jit
def kmer_counts(seq: ndarray, num_states: int, k: int) -> ndarray:
    """return freqs of valid k-mers using 1D indices

    Parameters
    ----------
    seq
        numpy array of uint, assumed that canonical characters have
        sequential indexes which are all < num_states
    result
        array to be written into
    num_states
        defines range of possible ints at a position
    k
        k-mer size

    Returns
    -------
    result
    """
    coeffs = coord_conversion_coeffs(num_states, k)
    kfreqs = zeros(num_states ** k, dtype=uint64)
    skip_until = 0
    for i in range(k):
        if seq[i] >= num_states:
            skip_until = i + 1

    for i in range(len(seq) - k + 1):
        if seq[i + k - 1] >= num_states:
            skip_until = i + k

        if i < skip_until:
            continue

        kmer = seq[i : i + k]
        index = (kmer * coeffs).sum()
        kfreqs[index] += 1

    return kfreqs


def _gt_zero(instance, attribute, value):
    if value <= 0:
        raise ValueError(f"must be > 0, not {value}")


@singledispatch
def _make_kcounts(data) -> sparse_vector:
    raise TypeError(f"type {type(data)} not supported")


@_make_kcounts.register
def _(data: ndarray):
    nonzero = {i: v for i, v in enumerate(data.tolist()) if v}
    return sparse_vector(
        vector_length=len(data), data=nonzero, dtype=_gettype(data.dtype.name)
    )


@_make_kcounts.register
def _(data: sparse_vector):
    return data


@define(slots=True, order=True, hash=True)
class SeqRecord:
    """representation of a single sequence as kmer counts"""

    kcounts: sparse_vector = field(eq=False, converter=_make_kcounts)
    name: str = field(validator=validators.instance_of(str), eq=True)
    length: int = field(validator=[validators.instance_of(int), _gt_zero], eq=True)
    delta_jsd: float = field(
        init=False,
        validator=validators.instance_of(float),
        default=0.0,
        eq=False,
    )

    @property
    def size(self):
        return len(self.kfreqs)

    @property
    def entropy(self):
        return self.kfreqs.entropy

    @property
    def kfreqs(self):
        kcounts = self.kcounts
        return kcounts / kcounts.sum()


class _seq_to_kmers:
    def __init__(self, k: int, moltype: str):
        """compute k-mers

        Parameters
        ----------
        k : int
            size of k-mers
        moltype : MolType
            cogent3 molecular type

        Raises
        ------
        ValueError
            if the mapping from characters to integers is not sequential

        Notes
        -----
        The sequence is converted to indices using states of canonical characters
        defined by moltype. Each k-mer is then a k-dimension coordinate. We
        convert those into a 1D coordinate. Use indices_to_seqs to convert
        indices back into k-mer sequences.
        """
        self.k = k
        self.canonical = _get_canonical_states(moltype)
        self.seq2array = dv_utils.str2arr(moltype=moltype)
        self.compress_pickled = dv_utils.pickle_data() + dv_utils.blosc_compress()


@composable.define_app
class seq_to_kmer_counts(_seq_to_kmers):
    def main(self, seq: c3_types.SeqType) -> sparse_vector:
        kwargs = dict(
            vector_length=len(self.canonical) ** self.k,
            dtype=int,
            source=getattr(seq, "source", None),
            name=seq.name,
        )
        seq = self.seq2array(seq._seq)
        if self.k > 7:
            result = _seq_to_all_kmers(seq, self.canonical, self.k)
            indices, counts = np_unique(result, return_counts=True)
            counts = dict(zip(indices.tolist(), counts.tolist()))
            del result
        else:  # just make a dense array
            counts = kmer_counts(seq, len(self.canonical), self.k)
            counts = {i: c for i, c in enumerate(counts) if c}

        kwargs["data"] = counts
        return sparse_vector(**kwargs)


@composable.define_app
class seq_to_unique_kmers(_seq_to_kmers):
    def main(self, seq: c3_types.SeqType) -> unique_kmers:
        result = _seq_to_all_kmers(seq, self.canonical, self.k)
        kwargs = dict(
            num_states=len(self.canonical),
            k=self.k,
            source=getattr(seq, "source", None),
            name=seq.name,
        )
        kwargs["data"] = np_unique(result)
        del result
        return unique_kmers(**kwargs)


def _get_canonical_states(moltype: str) -> bytes:
    moltype = get_moltype(moltype)
    canonical = list(moltype.alphabet)
    v = moltype.alphabet.to_indices(canonical)
    if not (0 <= min(v) < max(v) < len(canonical)):
        raise ValueError(f"indices of canonical states {canonical} not sequential {v}")
    return "".join(canonical).encode("utf8")


def _seq_to_all_kmers(seq: ndarray, states: bytes, k: int) -> ndarray:
    """return all valid k-mers from seq"""
    # positions with non-canonical characters are assigned value outside range
    num_states = len(states)
    dtype = get_array_type(num_states ** k)

    # k-mers that include an index for ambiguity character are excluded
    result = zeros(len(seq) - k + 1, dtype=dtype)
    result = kmer_indices(seq, result, num_states, k)
    return result


@composable.define_app
def indices2str(r: unique_kmers, states: str) -> tuple[str, str]:
    return r.name.rsplit(".", maxsplit=1)[0], ",".join(r.to_kmer_strings(states))
