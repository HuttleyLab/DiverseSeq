from collections import Counter
from pathlib import Path

import numpy
import pytest

from cogent3 import make_unaligned_seqs
from cogent3.maths.measure import jsd
from numpy.testing import assert_allclose

from divergent.record import SeqRecord, kmer_counts
from divergent.records import SummedRecords
from divergent.util import str2arr


@pytest.fixture
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
