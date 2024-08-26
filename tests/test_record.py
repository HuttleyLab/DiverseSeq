from collections import Counter
from itertools import product
from pathlib import Path

import numpy
import pytest
from cogent3 import make_seq
from numpy import (
    array,
    nextafter,
    ravel_multi_index,
    uint16,
    uint64,
    unravel_index,
    zeros,
)
from numpy.testing import assert_allclose

from diverse_seq.record import (
    KmerSeq,
    SeqArray,
    _gettype,
    coord_conversion_coeffs,
    coord_to_index,
    index_to_coord,
    indices_to_seqs,
    kmer_counts,
    lazy_kmers,
    member_to_kmerseq,
    seqarray_to_kmerseq,
    vector,
)
from diverse_seq.util import str2arr

DATADIR = Path(__file__).parent / "data"


seq5 = make_seq("ACGCG", name="null", moltype="dna")
seq0 = make_seq("", name="null", moltype="dna")
seq_nameless = make_seq("ACGTGTT", moltype="dna")
seq4eq = make_seq("ACGT", name="null", moltype="dna")
seq4one = make_seq("AAAA", name="null", moltype="dna")


@pytest.fixture
def seqarray():
    return SeqArray(
        seqid="seq1",
        data=numpy.array([0, 1, 2, 3]),
        moltype="dna",
        source="seq1_source",
    )


@pytest.mark.parametrize("seq,entropy", ((seq4eq, 2.0), (seq4one, 0.0)))
def test_seqrecord_entropy(seq, entropy):
    arr = str2arr()(str(seq))  # pylint: disable=not-callable
    kcounts = kmer_counts(arr, 4, 1)
    sr = KmerSeq(kcounts=kcounts, name=seq.name)
    assert sr.entropy == entropy


@pytest.mark.parametrize("name", (1, 1.0))
def test_seqrecord_invalid_types(name):
    kcounts = array([1, 2, 3])
    with pytest.raises(TypeError):
        KmerSeq(kcounts=kcounts, name=name)


@pytest.mark.parametrize("kcounts", ((0, 1, 2), [0, 1, 2]))
def test_seqrecord_invalid_kcounts(kcounts):
    with pytest.raises(TypeError):
        KmerSeq(kcounts=kcounts, name="n1")


def test_seqrecord_compare():
    arr = str2arr()(str(seq5))  # pylint: disable=not-callable
    kcounts = kmer_counts(arr, 4, 2)
    kwargs = dict(kcounts=kcounts)
    sr1 = KmerSeq(name="n1", **kwargs)
    sr1.delta_jsd = 1.0
    sr2 = KmerSeq(name="n2", **kwargs)
    sr2.delta_jsd = 2.0
    sr3 = KmerSeq(name="n3", **kwargs)
    sr3.delta_jsd = 34.0

    rec = sorted((sr3, sr1, sr2))
    assert rec[0].delta_jsd == 1 and rec[-1].delta_jsd == 34


def test_sparse_vector_create():
    from numpy import array

    data = {2: 3, 3: 9}
    # to constructor
    v1 = vector(data=data, vector_length=2**2, dtype=int)
    expect = array([0, 0, 3, 9])
    assert_allclose(v1.data, expect)

    # or via set item individually
    v2 = vector(vector_length=2**2)
    for index, count in data.items():
        v2[index] = count

    assert (v1.data == v2.data).all()


@pytest.mark.parametrize("cast", (float, int))
def test_sparse_vector_add_vector(cast):
    # adds two vectors
    data = {2: 3, 3: 9}
    v1 = vector(data=data, vector_length=2**2, dtype=cast)

    # add to zero vector
    v0 = vector(vector_length=2**2, dtype=cast)
    v_ = v1 + v0
    assert (v_.data == v1.data).all()
    assert v_.data is not v1.data

    # add to self
    v3 = v1 + v1
    assert (v3.data == v1.data * 2).all()

    # add in-place
    v1 += v1
    assert (v1.data == v3.data).all()


@pytest.mark.parametrize("zero", (0, 0.0, nextafter(0.0, 1.0) / 2))
def test_sparse_vector_add_zero(zero):
    # does not include
    expect = {2: 3.0, 3: 9.0}
    expect = numpy.zeros(4, dtype=float)
    expect[2] = 3.0
    expect[3] = 9.0
    data = expect[:]
    data[1] = 0.0
    v1 = vector(vector_length=2**2, data=data, dtype=float)
    assert (v1.data == expect).all()


@pytest.mark.parametrize("cast", (float, int))
def test_sparse_vector_add_scalar(cast):
    # adds two vectors
    data = {2: 3, 3: 9}
    v1 = vector(data=data, vector_length=2**2, dtype=cast)

    # add to zero vector
    v_ = v1 + 0
    assert (v_.data == v1.data).all()
    assert v_.data is not v1.data

    # add in place
    expect = v1.data[:]
    expect += 5
    v1 += 5
    assert (v1.data == expect).all()


@pytest.mark.parametrize("cast", (float, int))
def test_sparse_vector_sub_vector(cast):
    # sub two vectors
    data = {2: 3, 3: 9}
    v1 = vector(data=data, vector_length=2**2, dtype=cast)
    # sub self
    v3 = v1 - v1
    assert_allclose(v3.data, 0)

    data2 = {2: 3, 3: 10}
    v2 = vector(data=data2, vector_length=2**2, dtype=cast)
    v3 = v2 - v1
    expect = numpy.zeros(4, dtype=cast)
    expect[3] = cast(1)
    assert (v3.data == expect).all()

    # sub in-place
    orig = v1
    expect = v1.data - v2.data
    v1 -= v2
    assert v1 is orig
    assert (v1.data == expect).all()

    # negative allowed
    data = {2: 3, 3: 9}
    v1 = vector(data=data, vector_length=2**2, dtype=cast)
    data2 = {2: 6, 3: 10}
    v2 = vector(data=data2, vector_length=2**2, dtype=cast)
    v3 = v1 - v2
    assert v3[2] == -3


@pytest.mark.parametrize("cast", (float, int))
def test_sparse_vector_sub_scalar(cast):
    # subtracts two vectors
    data = {2: 3, 3: 9}
    v1 = vector(data=data, vector_length=2**2, dtype=cast)

    # subtract zero
    v_ = v1 - 0
    assert (v_.data == v1.data).all()
    assert v_.data is not v1.data

    # subtract in place
    expect = v1.data - 2
    v1 -= 2
    assert (v1.data == expect).all()


@pytest.mark.parametrize("cast", (float, int))
def test_sparse_vector_sub_elementwise(cast):
    data = {2: 3, 3: 9}
    v2 = vector(data=data, vector_length=2**2, dtype=cast)
    v2[1] -= 99
    assert v2[1] == -99
    text = "float" if cast == float else "int"
    assert v2[1].dtype.name.startswith(text)


@pytest.mark.parametrize("cast", (float, int))
def test_sparse_vector_elementwise(cast):
    data = {2: 3, 3: 9}
    v2 = vector(data=data, vector_length=2**2, dtype=cast)
    v2[1] += 99
    assert v2[1] == 99

    text = "float" if cast == float else "int"
    assert v2[1].dtype.name.startswith(text)


def test_sparse_vector_sum():
    sv = vector(vector_length=2**2, dtype=int)
    assert sv.sum() == 0


def test_sparse_vector_iter_nonzero():
    data = {3: 9, 2: 3}
    sv = vector(data=data, vector_length=4**2, dtype=int)
    got = list(sv.iter_nonzero())
    assert got == [3, 9]


def test_sv_iter():
    sv = vector(data={2: 1, 1: 1, 3: 1, 0: 1}, vector_length=2**2, dtype=int)
    sv /= sv.sum()
    got = list(sv.iter_nonzero())
    assert_allclose(got, 0.25)


def test_sv_pickling():
    import pickle

    o = vector(data={2: 1, 1: 1, 3: 1, 0: 1}, vector_length=2**2, dtype=int)
    p = pickle.dumps(o)
    u = pickle.loads(p)
    assert str(u) == str(o)


@pytest.mark.parametrize("cast", (float, int))
def test_sparse_vector_div_vector(cast):
    # adds two vectors
    data1 = {2: 6, 3: 18}
    v1 = vector(data=data1, vector_length=2**2, dtype=cast)

    # factored by 3
    data2 = {2: 3, 3: 3}
    v2 = vector(data=data2, vector_length=2**2, dtype=cast)
    expect = numpy.nan_to_num(v1.data / v2.data, nan=0.0)
    v3 = v1 / v2
    assert (v3.data == expect).all()

    # different factors
    data2 = {2: 3, 3: 6}
    v2 = vector(data=data2, vector_length=2**2, dtype=cast)
    v3 = v1 / v2
    expect = numpy.nan_to_num(v1.data / v2.data, nan=0.0)
    assert (v3.data == expect).all()

    # div in-place
    orig = v1
    v1 /= v2
    assert v1 is orig
    assert (v1.data == expect).all()


@pytest.mark.parametrize("cast", (float, int))
def test_sparse_vector_div_scalar(cast):
    # adds two vectors
    data1 = {2: 6, 3: 18}
    v1 = vector(data=data1, vector_length=2**2, dtype=cast)

    # factored by 3
    v2 = v1 / 3
    expect = numpy.nan_to_num(v1.data / 3, nan=0.0)
    assert (v2.data == expect).all()
    assert v2 is not v1

    # div in-place
    v1 /= 3
    assert (v1.data == expect).all()


@pytest.mark.parametrize("num_states,ndim", ((2, 1), (4, 2), (4, 4)))
def test_interconversion_of_coords_indices(num_states, ndim):
    # make sure match numpy functions
    coeffs = array(coord_conversion_coeffs(num_states, ndim))
    for coord in product(*(list(range(num_states)),) * ndim):
        coord = array(coord, dtype=uint64)
        nidx = ravel_multi_index(coord, dims=(num_states,) * ndim)
        idx = coord_to_index(coord, coeffs)
        assert idx == nidx
        ncoord = unravel_index(nidx, shape=(num_states,) * ndim)
        got = index_to_coord(idx, coeffs)
        assert (got == ncoord).all()
        assert (got == coord).all()


def test_coord2index_fail():
    coord = array((0, 1), dtype=uint64)
    coeffs = array(coord_conversion_coeffs(2, 4))
    # dimension of coords inconsistent with coeffs
    with pytest.raises(ValueError):
        coord_to_index(coord, coeffs)


_seqs = ("ACGGCGGTGCA", "ACGGNGGTGCA", "ANGGCGGTGNA")
_ks = (1, 2, 3)


def test_indices_to_seqs():
    indices = array([0, 8], dtype=uint16)
    states = b"TCAG"
    result = indices_to_seqs(indices, states, 2)
    assert result == ["TT", "AT"]
    indices = array([0, 16], dtype=uint16)
    with pytest.raises(IndexError):
        # 16 is outside range
        indices_to_seqs(indices, states, 2)


@pytest.mark.parametrize("seq,k", tuple(product(_seqs, _ks)))
def test_kmer_freqs(seq, k):
    seq = make_seq(seq, moltype="dna")
    states = (list(seq.moltype),) * k
    states = ["".join(v) for v in product(*states)]
    counts = Counter(seq.iter_kmers(k))
    expect = zeros(len(states), dtype=int)
    for kmer, v in counts.items():
        if kmer not in states:
            continue
        expect[states.index(kmer)] = v

    seq2array = str2arr()
    arr = seq2array(str(seq))  # pylint: disable=not-callable
    got = kmer_counts(arr, 4, k)
    assert (got == expect).all()


def test_composable():
    from cogent3.app.composable import define_app

    @define_app
    def matched_sr(records: list[KmerSeq]) -> bool:
        return True

    sr = KmerSeq(kcounts=numpy.array([1, 2, 3, 4], dtype=int), name="a")
    app = matched_sr()  # pylint: disable=no-value-for-parameter
    got = app([sr])  # pylint: disable=not-callable
    assert got


dtype_name = lambda x: numpy.dtype(x).name


@pytest.mark.parametrize(
    "dtype",
    ("int", "float", dtype_name(numpy.int32), dtype_name(numpy.float64)),
)
def test__gettype(dtype):
    if dtype[-1].isdigit():
        expect = getattr(numpy, dtype)
    else:
        expect = int if dtype == "int" else float
    got = _gettype(dtype)
    assert got == expect


@pytest.mark.parametrize("k", (1, 2, 3))
def test_seqarray_to_record(seqarray, k):
    s2r = seqarray_to_kmerseq(k=k, moltype="dna")
    rec = s2r(seqarray)  # pylint: disable=not-callable

    assert rec.name == "seq1"
    rec.kfreqs.sum() == len(seqarray) - k + 1


@pytest.fixture(scope="function")
def dstore():
    from diverse_seq.data_store import HDF5DataStore

    store = HDF5DataStore("test.h5", in_memory=True)
    yield store
    store.close()


def test_lazy_kmers(dstore, seqarray):
    member = dstore.write(unique_id="seq1", data=seqarray.data)

    lazy = lazy_kmers(member=member, k=2, moltype="dna")
    s2k = seqarray_to_kmerseq(k=2, moltype="dna")
    expect = s2k(seqarray)  # pylint: disable=not-callable
    assert_allclose(numpy.array(lazy), expect.kcounts)


def test_member_to_vector(dstore, seqarray):
    member = dstore.write(unique_id="seq1", data=seqarray.data)
    lazy = lazy_kmers(member=member, k=2, moltype="dna")
    vec = vector(data=lazy, vector_length=lazy.num_states)
    assert_allclose(numpy.array(vec), numpy.array(lazy))


def test_member_to_kmerseq(dstore, seqarray):
    member = dstore.write(unique_id="seq1", data=seqarray.data)
    new_app = member_to_kmerseq(k=2, moltype="dna")
    got = new_app(member)  # pylint: disable=not-callable
    assert isinstance(got, KmerSeq)
    old_app = seqarray_to_kmerseq(k=2, moltype="dna")
    expect = old_app(seqarray)  # pylint: disable=not-callable
    assert_allclose(got.kcounts, expect.kcounts)
