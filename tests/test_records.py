import pytest
from cogent3 import make_unaligned_seqs
from cogent3.maths.measure import jsd
from divergent.record import SeqRecord, kmer_counts
from divergent.records import SummedRecords, max_divergent, most_divergent
from divergent.util import str2arr
from numpy.testing import assert_allclose


@pytest.fixture()
def seqcoll():
    data = {"a": "AAAA", "b": "AAAA", "c": "TTTT", "d": "ACGT"}
    seqs = make_unaligned_seqs(data, moltype="dna")
    return seqs


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
        SeqRecord(kcounts=kcounts[s.name], name=s.name, length=len(s))
        for s in seqcoll.seqs
        if s.name in kcounts
    ]


@pytest.mark.parametrize("k", (1, 2))
def test_total_jsd(seqcoll, k):
    kcounts = _get_kfreqs_per_seq(seqcoll, k=k)
    records = [
        SeqRecord(kcounts=kcounts[s.name], name=f"{s.name}-{k}", length=len(s))
        for s in seqcoll.seqs
    ]
    sr = SummedRecords.from_records(records)
    freqs = {n: v.astype(float) / v.sum() for n, v in kcounts.items()}
    expect = jsd(*list(freqs.values()))
    assert_allclose(sr.total_jsd, expect)


def test_add(seqcoll):
    k = 1
    kcounts = _get_kfreqs_per_seq(seqcoll, k=k)
    records = _make_records(kcounts, seqcoll)
    sr = SummedRecords.from_records(records[:-1])
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
    sr = SummedRecords.from_records(records)
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
    sr = SummedRecords.from_records(list(records.values()))
    assert sr.increases_jsd(excluded) == expect


def test_mean_delta_jsd(seqcoll):
    k = 1
    kcounts = _get_kfreqs_per_seq(seqcoll, k=k)
    records = [
        SeqRecord(kcounts=kcounts[s.name], name=s.name, length=len(s))
        for s in seqcoll.seqs
    ]
    sr_with_a = SummedRecords.from_records(records)
    sr_without_a = SummedRecords.from_records([r for r in records if r.name != "a"])
    assert sr_without_a.mean_delta_jsd > sr_with_a.mean_delta_jsd


def test_replaced_lowest(seqcoll):
    k = 1
    kcounts = _get_kfreqs_per_seq(seqcoll, k=k)
    records = _make_records(kcounts, seqcoll)
    sr = SummedRecords.from_records(records[:-1])
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
    got = max_divergent(records, min_size=2)
    assert got.size == 2


def test_most_divergent(seqcoll):
    k = 1
    kcounts = _get_kfreqs_per_seq(seqcoll, k=k)
    records = _make_records(kcounts, seqcoll)
    got = most_divergent(records, size=3)
    assert got.size == 3
