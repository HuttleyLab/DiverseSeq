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
from numpy import isnan, ndarray
from rich.progress import track

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
        if isnan(record.delta_jsd):
            print(f"{record.name!r} had a nan")
            exit()
        result.append(record)
    return result


def _check_integrity(instance, attribute, records: list[SeqRecord]):
    last = records[0]
    for r in records[1:]:
        if r.delta_jsd < last.delta_jsd:
            raise RuntimeError


@define
class SummedRecords:
    """use the from_records clas method to construct the instance"""

    # we need the records to have been sorted by delta_jsd
    # following check is in place for now until fully tested
    # todo delete when convinced no longer required

    records: list[SeqRecord] = field(validator=_check_integrity)
    summed_kfreqs: sparse_vector
    summed_entropies: float
    total_jsd: float
    size: int = field(init=False)
    record_names: set = field(init=False)
    lowest: SeqRecord = field(init=False)

    def __init__(
        self,
        records: list[SeqRecord],
        summed_kfreqs: sparse_vector,
        summed_entropies: float,
        total_jsd: float,
    ):
        self.record_names = {r.name for r in records}
        self.total_jsd = total_jsd
        self.lowest = records[0]
        self.records = records[1:]
        self.size = len(records)
        # NOTE we exclude lowest record from freqs and entropy
        self.summed_kfreqs = summed_kfreqs - self.lowest.kfreqs
        self.summed_entropies = summed_entropies - self.lowest.entropy

    @classmethod
    def from_records(cls, records: list[:SeqRecord]):
        size = len(records)
        summed_kfreqs, summed_entropies = _summed_stats(records)
        total_jsd = _jsd(summed_kfreqs, summed_entropies, size)
        records = sorted(
            _delta_jsd(summed_kfreqs, summed_entropies, records),
            key=lambda x: x.delta_jsd,
        )
        return cls(records, summed_kfreqs, summed_entropies, total_jsd)

    def _make_new(
        self,
        records: list[:SeqRecord],
        summed_kfreqs: sparse_vector,
        summed_entropies: float,
    ):
        """summed are totals from all records"""
        size = len(records)
        total_jsd = _jsd(summed_kfreqs, summed_entropies, size)
        records = sorted(
            _delta_jsd(
                summed_kfreqs,
                summed_entropies,
                records,
            ),
            key=lambda x: x.delta_jsd,
        )
        return self.__class__(records, summed_kfreqs, summed_entropies, total_jsd)

    def __contains__(self, item: SeqRecord):
        return item.name in self.record_names

    def __add__(self, other: SeqRecord):
        assert other not in self
        summed_kfreqs = self.summed_kfreqs + self.lowest.kfreqs + other.kfreqs
        summed_entropies = self.summed_entropies + self.lowest.entropy + other.entropy
        return self._make_new(
            [self.lowest, other] + list(self.records),
            summed_kfreqs,
            summed_entropies,
        )

    def __sub__(self, other: SeqRecord):
        assert other in self
        records = [r for r in self.records + [self.lowest] if r is not other]
        if len(records) != len(self.records):
            raise ValueError(
                f"cannot subtract record {other.name!r}, not present in self"
            )

        summed_kfreqs = self.summed_kfreqs + self.lowest.kfreqs - other.kfreqs
        summed_entropies = self.summed_entropies + self.lowest.entropy - other.entropy
        return self._make_new(records, summed_kfreqs, summed_entropies)

    def iter_record_names(self):
        for name in self.record_names:
            yield name

    def increases_jsd(self, record: SeqRecord) -> bool:
        # whether total JSD increases when record is used
        j = _jsd(
            self.summed_kfreqs + record.kfreqs,
            self.summed_entropies + record.entropy,
            self.size,
        )
        return self.total_jsd < j

    @property
    def mean_jsd(self):
        return self.total_jsd / self.size

    @property
    def mean_delta_jsd(self):
        total = fsum([fsum(r.delta_jsd for r in self.records), self.lowest.delta_jsd])
        return total / (len(self.records) + 1)

    def replaced_lowest(self, other: SeqRecord):
        """returns new SummedRecords with other instead of lowest"""
        summed_kfreqs = self.summed_kfreqs + other.kfreqs
        summed_entropies = self.summed_entropies + other.entropy
        return self._make_new([other] + self.records, summed_kfreqs, summed_entropies)


def max_divergent(
    records: list[SeqRecord], size: int, verbose: bool = False
) -> SummedRecords:
    """returns SummedRecords that maximises mean delta_jsd

    Parameters
    ----------
    records
        list of SeqRecord instances
    size
        starting size of SummedRecords

    Notes
    -----
    This is sensitive to the order of records.
    """
    size = size or 2
    sr = SummedRecords.from_records(records[:size])

    if len(records) <= size:
        return sr

    for r in track(records):
        if r in sr:
            continue

        if not sr.increases_jsd(r):
            continue

        nsr = sr + r
        sr = nsr if nsr.mean_delta_jsd > sr.mean_delta_jsd else sr.replaced_lowest(r)

    size = sr.size
    num_neg = sum(r.delta_jsd < 0 for r in [sr.lowest] + sr.records)
    while sr.size > 2 and sr.lowest.delta_jsd < 0:
        sr = SummedRecords.from_records(sr.records)

    if verbose:
        completed_size = sr.size
        print(
            f"Pruned {size - completed_size} records with delta_jsd < 0; original had {num_neg} negative"
        )
    return sr
