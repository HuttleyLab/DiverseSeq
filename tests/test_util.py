import numpy
import pytest
from cogent3 import get_moltype
from numpy.testing import assert_allclose

from divergent import util as dvgt_util


def test_str2arr():
    dna = get_moltype("dna")
    s = "ACGTT"
    expect = dna.alphabet.to_indices(s)
    app = dvgt_util.str2arr()
    g = app(s)
    assert (g == expect).all()
    g = app("ACGNT")
    assert g[-2] > 3  # index for non-canonical character > num_states


@pytest.mark.parametrize("seq", ("ACGTT", "ACGNT", "AYGTT", ""))
def test_arr2str(seq):
    app = dvgt_util.str2arr() + dvgt_util.arr2str()
    got = app(seq)
    assert got == seq


@pytest.mark.parametrize("suffix", ("fa", "fasta", "genbank", "gbk", "gbk.gz"))
def test_get_seq_format(suffix):
    assert dvgt_util.get_seq_file_format(suffix) in ("fasta", "genbank")


@pytest.mark.parametrize("suffix", ("gbkgz", "paml"))
def test_get_seq_format_unknown(suffix):
    assert dvgt_util.get_seq_file_format(suffix) is None


def test_determine_chunk_size():
    got = list(dvgt_util.determine_chunk_size(10, 3))
    assert got == [4, 3, 3]


def test_chunked():
    data = list(range(10))
    got = list(dvgt_util.chunked(data, 3))
    expect = [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert got == expect


def test_summary_stats():
    data = numpy.random.randint(low=0, high=5, size=100)
    stats = dvgt_util.summary_stats(data)
    assert_allclose(stats.mean, data.mean())
    assert_allclose(stats.std, data.std(ddof=1))
    assert_allclose(stats.cov, data.std(ddof=1) / data.mean())


@pytest.fixture(params=(False, True))
def includes(request, tmp_path):
    names = ["a", "b", "c"]
    if request.param:
        inc = tmp_path / "includes.txt"
        inc.write_text("\n".join(names), encoding="utf-8")
        return inc
    return ",".join(names)


def test_parse_include_arg(includes):
    got = dvgt_util._comma_sep_or_file(includes)
    assert got == ["a", "b", "c"]
