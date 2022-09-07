from collections import Counter
from pathlib import Path

import numpy
import pytest

from cogent3 import make_unaligned_seqs
from cogent3.maths.measure import jsd
from numpy.testing import assert_allclose

from divergent.record import SeqRecord
from divergent.records import SummedRecords


@pytest.fixture
def seqcoll():
    data = {"a": "AAAA", "b": "AAAA", "c": "TTTT", "d": "ACGT"}
    seqs = make_unaligned_seqs(data, moltype="dna")
    return seqs


def _get_kfreqs_per_seq(seqs, k=1):
    alpha = tuple(seqs.moltype.alphabet.get_word_alphabet(k))
    result = {}
    for seq in seqs.seqs:
        counts = Counter(seq.iter_kmers(k=k))
        freqs = numpy.zeros(4 ** k, dtype=float)
        for kmer, count in counts.items():
            index = alpha.index(kmer)
            freqs[index] = count
        freqs /= freqs.sum()
        result[seq.name] = freqs
    return result


@pytest.mark.parametrize("k", (1, 2))
def test_total_jsd(seqcoll, k):
    freqs = _get_kfreqs_per_seq(seqcoll, k=k)
    records = [SeqRecord(k=k, seq=s, moltype="dna") for s in seqcoll.seqs]
    sr = SummedRecords.from_records(records)
    expect = jsd(*list(freqs.values()))
    assert_allclose(sr.total_jsd, expect)


def test_add(seqcoll):
    k = 1
    freqs = _get_kfreqs_per_seq(seqcoll, k=k)
    records = [SeqRecord(k=k, seq=s, moltype="dna") for s in seqcoll.seqs]
    sr = SummedRecords.from_records(records[:-1])
    orig = id(sr)
    sr = sr + records[-1]
    assert id(sr) != orig
    expect = jsd(*list(freqs.values()))
    assert_allclose(sr.total_jsd, expect)


def test_sub(seqcoll):
    k = 1
    freqs = _get_kfreqs_per_seq(seqcoll, k=k)
    records = [SeqRecord(k=k, seq=s, moltype="dna") for s in seqcoll.seqs]
    sr = SummedRecords.from_records(records)
    assert sr.size == 4
    orig = id(sr)
    sr = sr - records[-1]
    assert id(sr) != orig
    assert sr.size == 3
    expect = jsd(*[freqs[name] for name in sr.iter_record_names()])
    assert_allclose(sr.total_jsd, expect)
