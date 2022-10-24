"""summed records contain multiple SeqRecord instances. This container class
efficiently computes jsd

I laid this out in my comment https://github.com/GavinHuttley/BIOL3208-bensilke/issues/34#issuecomment-1105874023

convert a sequence collection into a SummedRecords instance, which supports
the following applications

Find N
-------
identify the N'th most divergent sequences

Most divergent
--------------
identify the set of sequences that maximise delta-JSD


SummedRecords is the container that simplifies these applications

"""
from math import fsum

from attrs import define, field
from numpy import array, ndarray

from divergent.record import SeqRecord, sparse_vector


# needs a jsd method for a new sequence
# needs summed entropy scores
# needs combined array of summed freqs to allow efficient calculation of entropy
# on creation, computes total JSD, then delta_jsd (contribution of each sequence)
# seq records sorted by the latter; stored summed quantities for n and n-1 (with
# the least contributor omitted)


def _jsd(summed_freqs: sparse_vector, summed_entropy: float, n: int):
    kfreqs = summed_freqs / n
    entropy = summed_entropy / n
    return kfreqs.entropy - entropy


def _summed_stats(records: list[SeqRecord]) -> tuple[sparse_vector, float]:
    # takes series of records and sums quantitative parts
    sv = records[0].kfreqs
    vec = sparse_vector(vector_length=len(sv), dtype=float)
    entropies = []
    for record in records:
        vec += record.kfreqs
        entropies.append(record.entropy)
    return vec, fsum(entropies)


def _delta_jsd(
    total_kfreqs: sparse_vector,
    total_entropies: ndarray,
    records: list[SeqRecord],
) -> list[SeqRecord]:
    """measures contribution of each record to the total JSD"""
    n = len(records)
    total_jsd = _jsd(total_kfreqs, total_entropies, n)
    result = []
    for record in records:
        summed_kfreqs = total_kfreqs - record.kfreqs
        summed_entropies = total_entropies - record.entropy
        jsd = _jsd(summed_kfreqs, summed_entropies, n - 1)
        record.delta_jsd = total_jsd - jsd
        result.append(record)
    return result


@define
class SummedRecords:
    """use the from_records clas method to construct the instance"""

    records: list[SeqRecord]
    total_kfreqs: sparse_vector
    total_entropies: ndarray
    total_jsd: float
    size: int = field(init=False)
    lowest: SeqRecord = field(init=False)

    def __init__(
        self,
        records: list[SeqRecord],
        summed_kfreqs: sparse_vector,
        summed_entropies: ndarray,
        total_jsd: float,
    ):
        self.total_jsd = total_jsd
        self.lowest = records[0]
        self.records = records[1:]
        self.size = len(records)
        self.total_kfreqs = summed_kfreqs - self.lowest.kfreqs
        self.total_entropies = summed_entropies - self.lowest.entropy

    @classmethod
    def from_records(cls, records: list[:SeqRecord]):
        size = len(records)
        summed_kfreqs, summed_entropies = _summed_stats(records)
        total_jsd = _jsd(summed_kfreqs, summed_entropies, size)
        records = sorted(_delta_jsd(summed_kfreqs, summed_entropies, records))
        return cls(records, summed_kfreqs, summed_entropies, total_jsd)

    def __add__(self, other: SeqRecord):
        summed_kfreqs = self.total_kfreqs + self.lowest.kfreqs + other.kfreqs
        summed_entropies = self.total_entropies + self.lowest.entropy + other.entropy
        size = self.size + 1
        total_jsd = _jsd(summed_kfreqs, summed_entropies, size)
        records = sorted(
            _delta_jsd(
                summed_kfreqs,
                summed_entropies,
                [self.lowest, other] + list(self.records),
            )
        )
        return self.__class__(records, summed_kfreqs, summed_entropies, total_jsd)

    def __sub__(self, other: SeqRecord):
        if other is self.lowest:
            records = self.records
        else:
            records = [self.lowest] + [r for r in self.records if r is not other]

        summed_kfreqs = self.total_kfreqs + self.lowest.kfreqs - other.kfreqs
        summed_entropies = self.total_entropies + self.lowest.entropy - other.entropy
        size = self.size - 1
        total_jsd = _jsd(summed_kfreqs, summed_entropies, size)
        records = sorted(
            _delta_jsd(
                summed_kfreqs,
                summed_entropies,
                records,
            )
        )
        return self.__class__(records, summed_kfreqs, summed_entropies, total_jsd)

    def iter_record_names(self):
        for record in [self.lowest] + self.records:
            yield record.name

    def increases_jsd(self, record: SeqRecord) -> bool:
        # whether total JSD increases when record is used
        return self.total_jsd < _jsd(
            self.kfreqs + record.kfreqs,
        )
        raise NotImplementedError
