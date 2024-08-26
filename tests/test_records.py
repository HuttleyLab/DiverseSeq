import pytest
from cogent3 import load_unaligned_seqs, make_unaligned_seqs
from cogent3.maths.measure import jsd
from numpy.testing import assert_allclose

from diverse_seq import records as dvs_records
from diverse_seq.record import KmerSeq, kmer_counts
from diverse_seq.util import str2arr


@pytest.fixture
def seqcoll():
    data = {"a": "AAAA", "b": "AAAA", "c": "TTTT", "d": "ACGT"}
    return make_unaligned_seqs(data, moltype="dna")


def _get_kfreqs_per_seq(seqs, k=1):
    app = str2arr()
    result = {}
    for seq in seqs.seqs:
        arr = app(str(seq))  # pylint: disable=not-callable
        freqs = kmer_counts(arr, 4, k)
        result[seq.name] = freqs
    return result


def _make_records(kcounts, seqcoll):
    return [
        KmerSeq(kcounts=kcounts[s.name], name=s.name)
        for s in seqcoll.seqs
        if s.name in kcounts
    ]


@pytest.mark.parametrize("k", (1, 2))
def test_total_jsd(seqcoll, k):
    kcounts = _get_kfreqs_per_seq(seqcoll, k=k)
    records = [
        KmerSeq(kcounts=kcounts[s.name], name=f"{s.name}-{k}") for s in seqcoll.seqs
    ]
    sr = dvs_records.SummedRecords.from_records(records)
    freqs = {n: v.astype(float) / v.sum() for n, v in kcounts.items()}
    expect = jsd(*list(freqs.values()))
    assert_allclose(sr.total_jsd, expect)


def test_add(seqcoll):
    k = 1
    kcounts = _get_kfreqs_per_seq(seqcoll, k=k)
    records = _make_records(kcounts, seqcoll)
    sr = dvs_records.SummedRecords.from_records(records[:-1])
    orig = id(sr)
    sr = sr + records[-1]
    assert id(sr) != orig
    freqs = {n: v.astype(float) / v.sum() for n, v in kcounts.items()}
    expect = jsd(*list(freqs.values()))
    assert_allclose(sr.total_jsd, expect)


def test_sub(seqcoll):
    k = 1
    kcounts = _get_kfreqs_per_seq(seqcoll, k=k)
    records = _make_records(kcounts, seqcoll)
    sr = dvs_records.SummedRecords.from_records(records)
    assert sr.size == 4
    orig = id(sr)
    sr = sr - records[-1]
    assert id(sr) != orig
    assert sr.size == 3
    freqs = {n: v.astype(float) / v.sum() for n, v in kcounts.items()}
    expect = jsd(*[freqs[name] for name in sr.iter_record_names()])
    assert_allclose(sr.total_jsd, expect)


@pytest.mark.parametrize(
    "exclude,expect",
    (("a", False), ("b", False), ("c", True), ("d", True)),
)
def test_increases_jsd(seqcoll, exclude, expect):
    kcounts = _get_kfreqs_per_seq(seqcoll, k=1)
    records = {r.name: r for r in _make_records(kcounts, seqcoll)}
    excluded = records.pop(exclude)
    sr = dvs_records.SummedRecords.from_records(list(records.values()))
    assert sr.increases_jsd(excluded) == expect


def test_mean_delta_jsd(seqcoll):
    k = 1
    kcounts = _get_kfreqs_per_seq(seqcoll, k=k)
    records = [KmerSeq(kcounts=kcounts[s.name], name=s.name) for s in seqcoll.seqs]
    sr_with_a = dvs_records.SummedRecords.from_records(records)
    sr_without_a = dvs_records.SummedRecords.from_records(
        [r for r in records if r.name != "a"],
    )
    assert sr_without_a.mean_delta_jsd > sr_with_a.mean_delta_jsd


def test_replaced_lowest(seqcoll):
    k = 1
    kcounts = _get_kfreqs_per_seq(seqcoll, k=k)
    records = _make_records(kcounts, seqcoll)
    sr = dvs_records.SummedRecords.from_records(records[:-1])
    lowest = sr.lowest
    nsr = sr.replaced_lowest(records[-1])

    assert nsr is not sr
    assert sr.size == nsr.size == len(records) - 1
    # make sure previous lowest not present at all
    assert nsr.lowest is not lowest
    for r in nsr.records:
        assert r is not lowest
    # make sure new record is present
    assert any(r is records[-1] for r in nsr.records + [nsr.lowest])


def test_max_divergent(seqcoll):
    k = 1
    kcounts = _get_kfreqs_per_seq(seqcoll, k=k)
    records = _make_records(kcounts, seqcoll)
    got = dvs_records.max_divergent(records, min_size=2, max_size=2, stat="stdev")
    assert got.size == 2


def test_most_divergent(seqcoll):
    k = 1
    kcounts = _get_kfreqs_per_seq(seqcoll, k=k)
    records = _make_records(kcounts, seqcoll)
    got = dvs_records.most_divergent(records, size=3)
    assert got.size == 3
    assert got.to_table().shape == (3, 2)


def test_all_records(seqcoll):
    k = 1
    kcounts = _get_kfreqs_per_seq(seqcoll, k=k)
    records = _make_records(kcounts, seqcoll)
    got = dvs_records.most_divergent(records, size=3).all_records()
    assert len(got) == 3
    assert isinstance(got[0], KmerSeq)


@pytest.fixture(scope="session")
def brca1_coll(DATA_DIR):
    return load_unaligned_seqs(DATA_DIR / "brca1.fasta", moltype="dna").degap()


def test_merge_summed_records(DATA_DIR, brca1_coll):
    path = DATA_DIR / "brca1.dvseqs"
    names = brca1_coll.names
    app = dvs_records.dvs_nmost(seq_store=path, n=5, k=1)
    sr1 = app(names[:10])  # pylint: disable=not-callable
    sr2 = app(names[10:20])  # pylint: disable=not-callable
    rnames1 = sr1.record_names
    rnames2 = sr2.record_names
    assert set(rnames1) != set(rnames2)
    got = dvs_records.dvs_final_nmost()([sr1, sr2])
    assert len(got.record_names) == 5


def test_dvs_select_max(brca1_coll):
    app = dvs_records.dvs_select_max(k=1, min_size=2, max_size=5)
    got = app(brca1_coll)  # pylint: disable=not-callable
    assert 2 <= got.num_seqs <= 5
    app = dvs_records.dvs_select_max(k=1, min_size=2, max_size=5, seed=123)
    got = app(brca1_coll)  # pylint: disable=not-callable
    assert 2 <= got.num_seqs <= 5


@pytest.mark.parametrize("include", ("Human", ["Human"], ["Human", "Mouse"]))
def test_dvs_select_max_include(brca1_coll, include):
    app = dvs_records.dvs_select_max(k=1, min_size=2, max_size=5, include=include)
    got = app(brca1_coll)  # pylint: disable=not-callable
    include = {include} if isinstance(include, str) else set(include)
    assert include <= set(got.names)


def test_dvs_select_nmost(brca1_coll):
    app = dvs_records.dvs_select_nmost(k=1, n=5)
    got = app(brca1_coll)  # pylint: disable=not-callable
    assert got.num_seqs == 5
    # try setting a seed
    app = dvs_records.dvs_select_nmost(k=1, n=5, seed=123)
    got = app(brca1_coll)  # pylint: disable=not-callable
    assert got.num_seqs == 5


@pytest.mark.parametrize("include", ("Human", ["Human"], ["Human", "Mouse"]))
def test_dvs_select_nmost_keep(brca1_coll, include):
    app = dvs_records.dvs_select_nmost(k=1, n=5, include=include)
    got = app(brca1_coll)  # pylint: disable=not-callable
    include = {include} if isinstance(include, str) else set(include)
    assert include <= set(got.names)
