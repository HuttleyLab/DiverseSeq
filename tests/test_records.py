import pytest
from cogent3 import load_unaligned_seqs, make_unaligned_seqs
from cogent3.maths.measure import jsd
from numpy.testing import assert_allclose

from divergent import records as dvgt_records
from divergent.record import KmerSeq, kmer_counts
from divergent.util import str2arr


@pytest.fixture()
def seqcoll():
    data = {"a": "AAAA", "b": "AAAA", "c": "TTTT", "d": "ACGT"}
    return make_unaligned_seqs(data, moltype="dna")


def _get_kfreqs_per_seq(seqs, k=1):
    app = str2arr()
    result = {}
    for seq in seqs.seqs:
        arr = app(str(seq))
        freqs = kmer_counts(arr, 4, k)
        result[seq.name] = freqs
    return result


def _make_records(kcounts, seqcoll):
    return [
        KmerSeq(kcounts=kcounts[s.name], name=s.name, length=len(s))
        for s in seqcoll.seqs
        if s.name in kcounts
    ]


@pytest.mark.parametrize("k", (1, 2))
def test_total_jsd(seqcoll, k):
    kcounts = _get_kfreqs_per_seq(seqcoll, k=k)
    records = [
        KmerSeq(kcounts=kcounts[s.name], name=f"{s.name}-{k}", length=len(s))
        for s in seqcoll.seqs
    ]
    sr = dvgt_records.SummedRecords.from_records(records)
    freqs = {n: v.astype(float) / v.sum() for n, v in kcounts.items()}
    expect = jsd(*list(freqs.values()))
    assert_allclose(sr.total_jsd, expect)


def test_add(seqcoll):
    k = 1
    kcounts = _get_kfreqs_per_seq(seqcoll, k=k)
    records = _make_records(kcounts, seqcoll)
    sr = dvgt_records.SummedRecords.from_records(records[:-1])
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
    sr = dvgt_records.SummedRecords.from_records(records)
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
    sr = dvgt_records.SummedRecords.from_records(list(records.values()))
    assert sr.increases_jsd(excluded) == expect


def test_mean_delta_jsd(seqcoll):
    k = 1
    kcounts = _get_kfreqs_per_seq(seqcoll, k=k)
    records = [
        KmerSeq(kcounts=kcounts[s.name], name=s.name, length=len(s))
        for s in seqcoll.seqs
    ]
    sr_with_a = dvgt_records.SummedRecords.from_records(records)
    sr_without_a = dvgt_records.SummedRecords.from_records(
        [r for r in records if r.name != "a"],
    )
    assert sr_without_a.mean_delta_jsd > sr_with_a.mean_delta_jsd


def test_replaced_lowest(seqcoll):
    k = 1
    kcounts = _get_kfreqs_per_seq(seqcoll, k=k)
    records = _make_records(kcounts, seqcoll)
    sr = dvgt_records.SummedRecords.from_records(records[:-1])
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
    got = dvgt_records.max_divergent(records, min_size=2)
    assert got.size == 2


def test_most_divergent(seqcoll):
    k = 1
    kcounts = _get_kfreqs_per_seq(seqcoll, k=k)
    records = _make_records(kcounts, seqcoll)
    got = dvgt_records.most_divergent(records, size=3)
    assert got.size == 3
    assert got.to_table().shape == (3, 2)


def test_all_records(seqcoll):
    k = 1
    kcounts = _get_kfreqs_per_seq(seqcoll, k=k)
    records = _make_records(kcounts, seqcoll)
    got = dvgt_records.most_divergent(records, size=3).all_records()
    assert len(got) == 3
    assert isinstance(got[0], KmerSeq)


def test_merge_summed_records(DATA_DIR):
    path = DATA_DIR / "brca1.dvgtseqs"
    names = load_unaligned_seqs(DATA_DIR / "brca1.fasta").names
    app = dvgt_records.dvgt_nmost(seq_store=path, n=5, k=1)
    sr1 = app(names[:10])
    sr2 = app(names[10:20])
    rnames1 = sr1.record_names
    rnames2 = sr2.record_names
    got = dvgt_records.dvgt_final_nmost()([sr1, sr2])
    final_selected = got.record_names
    assert len(final_selected) == 5
    assert set(final_selected) != set(rnames1)
    assert set(final_selected) != set(rnames2)


def test_dvgt_select_max(DATA_DIR):
    seqs = load_unaligned_seqs(DATA_DIR / "brca1.fasta", moltype="dna").degap()
    app = dvgt_records.dvgt_select_max(k=1, min_size=2, max_size=5)
    got = app(seqs)
    assert 2 <= got.num_seqs <= 5
    app = dvgt_records.dvgt_select_max(k=1, min_size=2, max_size=5, seed=123)
    got = app(seqs)
    assert 2 <= got.num_seqs <= 5


def test_dvgt_select_nmost(DATA_DIR):
    seqs = load_unaligned_seqs(DATA_DIR / "brca1.fasta", moltype="dna").degap()
    app = dvgt_records.dvgt_select_nmost(k=1, n=5)
    got = app(seqs)
    assert got.num_seqs == 5
    # try setting a seed
    app = dvgt_records.dvgt_select_nmost(k=1, n=5, seed=123)
    got = app(seqs)
    assert got.num_seqs == 5
