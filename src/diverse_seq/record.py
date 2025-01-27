"""defines basic data type for storing an individual sequence record"""

import dataclasses
import functools
from collections.abc import Iterator
from math import fabs, isclose

import numba
import typing_extensions
from attrs import asdict, define, field, validators
from cogent3 import get_moltype
from cogent3.app import composable
from cogent3.app import data_store as c3_data_store
from cogent3.app import typing as c3_types
from cogent3.core import new_sequence as c3_new_seq
from cogent3.core import sequence as c3_seq
from numpy import (
    array,
    dtype,
    errstate,
    log2,
    min_scalar_type,
    nan_to_num,
    ndarray,
    uint8,
    uint64,
    zeros,
)
from numpy import divmod as np_divmod

from diverse_seq import util as dvs_utils

NumType = float | int
PosDictType = dict[int, NumType]


@functools.singledispatch
def _gettype(name) -> type:
    try:
        return name.type
    except AttributeError as e:
        msg = f"type {type(name)} not supported"
        raise TypeError(msg) from e


@_gettype.register
def _(name: str) -> type:
    import numpy

    if name[-1].isdigit():
        return getattr(numpy, name)
    return {"int": int, "float": float}[name]


@_gettype.register
def _(name: type) -> type:
    return name


@dataclasses.dataclass(slots=True)
class lazy_kmers:
    data: c3_data_store.DataMember | ndarray
    k: int
    dtype: type = uint64
    num_states: int = dataclasses.field(init=False)
    moltype: dataclasses.InitVar[str] = dataclasses.field(default="dna")

    def __post_init__(self, moltype: str) -> None:
        self.num_states = len(_get_canonical_states(moltype))

    def __array__(
        self,
        dtype: dtype | None = None,
        copy: bool | None = None,
    ) -> ndarray[int]:
        data = self.data if isinstance(self.data, ndarray) else self.data.read()
        return kmer_counts(data, self.num_states, self.k, dtype=self.dtype)


@functools.singledispatch
def _make_data(data, size: int | None = None, dtype: type = None) -> ndarray:
    raise NotImplementedError


@_make_data.register
def _(data: ndarray, size: int | None = None, dtype: type = None) -> ndarray:
    return data


@_make_data.register
def _(data: lazy_kmers, size: int | None = None, dtype: type = None) -> ndarray:
    return data


@_make_data.register
def _(data: None, size: int | None = None, dtype: type = None) -> ndarray:
    return zeros(size, dtype=dtype)


@_make_data.register
def _(data: dict, size: int | None = None, dtype: type = None) -> ndarray:
    result = zeros(size, dtype=dtype)
    for i, n in data.items():
        if isclose(n, 0):
            continue
        result[i] = n
    return result


# convert this to a dataclass
# write a wrapper for the data store member so it has a __array__ method
@define(slots=True)
class vector:
    data: ndarray
    vector_length: int
    default: NumType | None = field(init=False)
    dtype: type = float
    source: str = None
    name: str = None

    def __init__(
        self,
        *,
        vector_length: int,
        data: ndarray = None,
        dtype: type = float,
        source: str = None,
        name: str = None,
    ) -> None:
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
        dtype = _gettype(dtype)
        self.dtype = dtype

        data = _make_data(data, size=vector_length, dtype=dtype)
        self.default = dtype(0)
        self.data = data
        self.source = source
        self.name = name

    def __setitem__(self, index: int, value: NumType) -> None:
        self.data[index] = value

    def __getitem__(self, index: int) -> NumType:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[NumType]:
        yield from self.data

    def __getstate__(self) -> dict:
        return asdict(self)

    def __setstate__(self, data: dict) -> typing_extensions.Self:
        for k, v in data.items():
            setattr(self, k, v)
        return self

    def __sub__(self, other: typing_extensions.Self) -> typing_extensions.Self:
        data = self.data - other
        return self.__class__(
            data=data,
            vector_length=self.vector_length,
            dtype=self.dtype,
        )

    def __isub__(self, other: typing_extensions.Self) -> typing_extensions.Self:
        self.data -= other
        return self

    def __add__(self, other: typing_extensions.Self) -> typing_extensions.Self:
        # we are creating a new instance
        data = self.data + other
        return self.__class__(
            data=data,
            vector_length=self.vector_length,
            dtype=self.dtype,
        )

    def __iadd__(self, other: typing_extensions.Self) -> typing_extensions.Self:
        self.data += other
        return self

    def __truediv__(self, other: typing_extensions.Self) -> typing_extensions.Self:
        # we are creating a new instance
        with errstate(divide="ignore", invalid="ignore"):
            data = nan_to_num(self.data / other, nan=0.0, copy=False)
        return self.__class__(data=data, vector_length=self.vector_length, dtype=float)

    def __itruediv__(self, other: typing_extensions.Self) -> typing_extensions.Self:
        with errstate(divide="ignore", invalid="ignore"):
            data = nan_to_num(self.data / other, nan=0.0, copy=False)
        self.dtype = float
        self.data = data
        return self

    def sum(self) -> NumType:
        return self.data.sum()

    def iter_nonzero(self) -> Iterator[NumType]:
        yield from (v for v in self.data if v)

    @property
    def entropy(self) -> float:
        non_zero = self.data[self.data > 0]
        kfreqs = non_zero if self.dtype == float else non_zero / non_zero.sum()
        # taking absolute value due to precision issues
        return fabs(-(kfreqs * log2(kfreqs)).sum())

    def __array__(
        self,
        dtype: dtype | None = None,
        copy: bool | None = None,
    ) -> ndarray[int]:
        if not isinstance(self.data, ndarray):
            self.data = array(self.data)
        return self.data


@numba.jit(nopython=True)
def coord_conversion_coeffs(num_states, k):  # pragma: no cover
    """coefficients for multi-dimensional coordinate conversion into 1D index"""
    return array([num_states ** (i - 1) for i in range(k, 0, -1)])


@numba.jit(nopython=True)
def coord_to_index(coord, coeffs):  # pragma: no cover
    """converts a multi-dimensional coordinate into a 1D index"""
    return (coord * coeffs).sum()


@numba.jit(nopython=True)
def index_to_coord(index, coeffs):  # pragma: no cover
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


@numba.jit(nopython=True)
def indices_to_bytes(
    indices: ndarray,
    states: bytes,
    k: int,
) -> ndarray:  # pragma: no cover
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
                msg = "index out of character range"
                raise IndexError(msg)
            result[i][j] = states[coord[j]]

    return result


@numba.jit(nopython=True)
def kmer_counts(
    seq: ndarray,
    num_states: int,
    k: int,
    dtype: dtype = uint64,
) -> ndarray:  # pragma: no cover
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
    kfreqs = zeros(num_states**k, dtype=dtype)
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
        msg = f"must be > 0, not {value}"
        raise ValueError(msg)


@functools.singledispatch
def _make_kcounts(data) -> vector:
    msg = f"type {type(data)} not supported"
    raise TypeError(msg)


@_make_kcounts.register
def _(data: ndarray):
    nonzero = {i: v for i, v in enumerate(data.tolist()) if v}
    return vector(
        vector_length=len(data),
        data=nonzero,
        dtype=_gettype(data.dtype.name),
    )


@_make_kcounts.register
def _(data: vector):
    return data


@dataclasses.dataclass(frozen=True)
class SeqArray:
    """A SeqArray stores an array of indices that map to the canonical characters
    of the moltype of the original sequence. Use rep_seq.util.arr2str() to
    recapitulate the original sequence."""

    seqid: str
    data: ndarray
    moltype: str
    source: str = None

    def __len__(self) -> int:
        return len(self.data)


@define(slots=True, order=True, hash=True)
class KmerSeq:
    """representation of a single sequence as kmer counts"""

    kcounts: vector = field(eq=False, converter=_make_kcounts)
    name: str = field(validator=validators.instance_of(str), eq=True)
    delta_jsd: float = field(
        init=False,
        validator=validators.instance_of(float),
        default=0.0,
        eq=False,
    )

    @property
    def size(self) -> int:
        return self.kcounts.vector_length

    @functools.cached_property
    def entropy(self) -> float:
        return self.kfreqs.entropy

    @functools.cached_property
    def kfreqs(self) -> vector:
        kcounts = array(self.kcounts)
        kcounts = kcounts.astype(float)
        kfreqs = kcounts / kcounts.sum()
        return vector(
            data=kfreqs,
            vector_length=len(kfreqs),
            dtype=float,
        )


@composable.define_app
class seq_to_seqarray:
    """Convert a cogent3 Sequence to a SeqArray"""

    def __init__(
        self,
        moltype: str = "dna",
    ) -> None:
        self.moltype = moltype
        self.str2arr = dvs_utils.str2arr(moltype=self.moltype)

    def main(self, seq: c3_types.SeqType) -> SeqArray:
        return SeqArray(
            seqid=seq.name,
            data=self.str2arr(str(seq)),
            moltype=self.moltype,
            source=seq.info.source or seq.name,
        )


@functools.singledispatch
def make_kmerseq(data, *, dtype: dtype, k: int, moltype: str) -> KmerSeq:
    msg = f"type {type(data)} not supported"
    raise TypeError(msg)


@make_kmerseq.register
def _(data: SeqArray, *, dtype: dtype, k: int, moltype: str) -> KmerSeq:
    vec = lazy_kmers(
        data=data.data,
        k=k,
        moltype=moltype,
        dtype=dtype,
    )
    kwargs = {
        "vector_length": vec.num_states,
        "dtype": dtype,
        "source": data.source,
        "name": data.seqid,
        "data": vec,
    }

    return KmerSeq(
        kcounts=vector(**kwargs),
        name=kwargs["name"],
    )


@make_kmerseq.register
def _(data: c3_data_store.DataMember, *, dtype: dtype, k: int, moltype: str) -> KmerSeq:
    vec = lazy_kmers(
        data=data,
        k=k,
        moltype=moltype,
        dtype=dtype,
    )
    kwargs = {
        "vector_length": vec.num_states,
        "dtype": dtype,
        "source": data.data_store.source,
        "name": data.unique_id,
        "data": vec,
    }

    return KmerSeq(
        kcounts=vector(**kwargs),
        name=kwargs["name"],
    )


@make_kmerseq.register
def _(data: c3_seq.Sequence, *, dtype: dtype, k: int, moltype: str) -> KmerSeq:
    cnvrt = dvs_utils.str2arr(moltype=moltype)
    vec = lazy_kmers(
        data=cnvrt(str(data)),  # pylint: disable=not-callable
        k=k,
        moltype=moltype,
        dtype=dtype,
    )
    kwargs = {
        "vector_length": vec.num_states,
        "dtype": dtype,
        "source": data.info.source,
        "name": data.name,
        "data": vec,
    }

    return KmerSeq(
        kcounts=vector(**kwargs),
        name=kwargs["name"],
    )


@make_kmerseq.register
def _(data: c3_new_seq.Sequence, *, dtype: dtype, k: int, moltype: str) -> KmerSeq:
    cnvrt = dvs_utils.str2arr(moltype=moltype)
    vec = lazy_kmers(
        data=cnvrt(str(data)),  # pylint: disable=not-callable
        k=k,
        moltype=moltype,
        dtype=dtype,
    )
    kwargs = {
        "vector_length": vec.num_states,
        "dtype": dtype,
        "source": data.info.source,
        "name": data.name,
        "data": vec,
    }

    return KmerSeq(
        kcounts=vector(**kwargs),
        name=kwargs["name"],
    )


class _make_kmerseq_init:
    def __init__(self, k: int, moltype: str) -> None:
        """
        Parameters
        ----------
        k
            size of k-mers
        moltype
            cogent3 molecular type
        """
        self.moltype = moltype
        self.k = k
        self.num_states = len(_get_canonical_states(moltype))
        self.dtype = min_scalar_type(self.num_states**k)


@composable.define_app
class seqarray_to_kmerseq(_make_kmerseq_init):
    """converts a sequence numpy array to a KmerSeq"""

    def main(self, seq: SeqArray) -> KmerSeq:
        return make_kmerseq(seq, dtype=self.dtype, k=self.k, moltype=self.moltype)


@composable.define_app
class member_to_kmerseq(_make_kmerseq_init):
    """creates a KmerSeq from a HDF5 datastore member"""

    def main(self, member: c3_data_store.DataMember) -> KmerSeq:
        return make_kmerseq(member, dtype=self.dtype, k=self.k, moltype=self.moltype)


@functools.cache
def _get_canonical_states(moltype: str) -> bytes:
    moltype = get_moltype(moltype)
    canonical = list(moltype.alphabet)
    v = moltype.alphabet.to_indices(canonical)
    if not (0 <= min(v) < max(v) < len(canonical)):
        msg = f"indices of canonical states {canonical} not sequential {v}"
        raise ValueError(msg)
    return "".join(canonical).encode("utf8")
