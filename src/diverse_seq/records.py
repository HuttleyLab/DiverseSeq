"""summed records contain multiple SeqRecord instances. This container class
efficiently computes jsd

Convert a sequence collection into a SummedRecords instance, which supports
the following applications

Find N
-------
identify the n-most diverse sequences

Most divergent
--------------
identify the set of sequences that maximise delta-JSD

SummedRecords is the container that simplifies these applications
"""

import functools
import itertools
import pathlib
import sys
import typing
from math import fsum

import numpy
from attrs import define, field
from cogent3 import make_table
from cogent3.app import typing as c3_types
from cogent3.app.composable import NotCompleted, define_app
from numpy import isclose as np_isclose
from rich import progress as rich_progress

from diverse_seq import data_store as dvs_data_store
from diverse_seq import util as dvs_util
from diverse_seq.record import (
    KmerSeq,
    member_to_kmerseq,
    seq_to_seqarray,
    seqarray_to_kmerseq,
    vector,
)

# needs a jsd method for a new sequence
# needs summed entropy scores
# needs combined array of summed freqs to allow efficient calculation of entropy
# on creation, computes total JSD, then delta_jsd (contribution of each sequence)
# seq records sorted by the latter; stored summed quantities for n and n-1 (with
# the least contributor omitted)


@functools.singledispatch
def _jsd(summed_freqs: vector | numpy.ndarray, summed_entropy: float, n: int) -> float:
    raise NotImplementedError


@_jsd.register
def _(summed_freqs: vector, summed_entropy: float, n: int):
    kfreqs = summed_freqs / n
    entropy = summed_entropy / n
    return kfreqs.entropy - entropy


@_jsd.register
def _(summed_freqs: numpy.ndarray, summed_entropy: float, n: int):
    kfreqs = summed_freqs / n
    entropy = summed_entropy / n
    kfreqs = kfreqs[~np_isclose(kfreqs, 0)]
    ke = -(kfreqs * numpy.log2(kfreqs)).sum()
    return ke - entropy


def _summed_stats(records: list[KmerSeq]) -> tuple[vector, float]:
    # takes series of records and sums quantitative parts
    sv = records[0].kfreqs
    vec = numpy.zeros(len(sv), dtype=float)
    entropies = []
    for record in records:
        vec += record.kfreqs
        entropies.append(record.entropy)

    vec = vector(data=vec, vector_length=len(vec), dtype=float)
    return vec, fsum(entropies)


def _delta_jsd(
    total_kfreqs: vector,
    total_entropies: float,
    records: list[KmerSeq],
) -> list[KmerSeq]:
    """measures contribution of each record to the total JSD"""
    n = len(records)
    total_jsd = _jsd(total_kfreqs, total_entropies, n)
    total_kfreqs = total_kfreqs.data
    result = []
    for record in records:
        summed_kfreqs = total_kfreqs - record.kfreqs.data
        summed_entropies = total_entropies - record.entropy
        jsd = _jsd(summed_kfreqs, summed_entropies, n - 1)
        record.delta_jsd = total_jsd - jsd
        if numpy.isnan(record.delta_jsd):
            print(f"{record.name!r} had a nan")
            sys.exit(1)
        result.append(record)
    return result


def _check_integrity(instance, attribute, records: list[KmerSeq]):
    last = records[0]
    for r in records[1:]:
        if r.delta_jsd < last.delta_jsd:
            raise RuntimeError


@define
class SummedRecords:
    """use the from_records clas method to construct the instance"""

    # we need the records to have been sorted by delta_jsd
    # following check is in place for now until fully tested
    # TODO delete when convinced no longer required

    records: list[KmerSeq] = field(validator=_check_integrity)
    summed_kfreqs: vector
    summed_entropies: float
    total_jsd: float
    size: int = field(init=False)
    record_names: set = field(init=False)
    lowest: KmerSeq = field(init=False)
    _stats: dvs_util.summary_stats = field(init=False)

    def __init__(
        self,
        records: list[KmerSeq],
        summed_kfreqs: vector,
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
        self._stats = dvs_util.summary_stats(
            numpy.array([r.delta_jsd for r in records]),
        )

    @classmethod
    def from_records(cls, records: list[:KmerSeq]):
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
        records: list[:KmerSeq],
        summed_kfreqs: vector,
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

    def __contains__(self, item: KmerSeq):
        return item.name in self.record_names

    def __add__(self, other: KmerSeq):
        assert other not in self
        summed_kfreqs = self.summed_kfreqs + self.lowest.kfreqs + other.kfreqs
        summed_entropies = self.summed_entropies + self.lowest.entropy + other.entropy
        return self._make_new(
            [self.lowest, other] + list(self.records),
            summed_kfreqs,
            summed_entropies,
        )

    def __sub__(self, other: KmerSeq):
        if other not in self:
            raise ValueError(
                f"cannot subtract record {other.name!r}, not present in self",
            )
        records = [r for r in self.records + [self.lowest] if r is not other]

        summed_kfreqs = self.summed_kfreqs + self.lowest.kfreqs - other.kfreqs
        summed_entropies = self.summed_entropies + self.lowest.entropy - other.entropy
        return self._make_new(records, summed_kfreqs, summed_entropies)

    def iter_record_names(self):
        yield from self.record_names

    def increases_jsd(self, record: KmerSeq) -> bool:
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
        """mean of delta_jsd"""
        return self._stats.mean

    @property
    def std_delta_jsd(self):
        """unbiased standard deviation of delta_jsd"""
        return self._stats.std

    @property
    def cov_delta_jsd(self):
        """coeff of variation of delta_jsd"""
        return self._stats.cov

    def replaced_lowest(self, other: KmerSeq):
        """returns new instance with other instead of lowest"""
        summed_kfreqs = self.summed_kfreqs + other.kfreqs
        summed_entropies = self.summed_entropies + other.entropy
        return self._make_new([other] + self.records, summed_kfreqs, summed_entropies)

    def to_table(self):
        names, deltas = list(
            zip(
                *[(r.name, r.delta_jsd) for r in [self.lowest] + self.records],
                strict=False,
            ),
        )
        return make_table(data={"names": names, "delta_jsd": deltas})

    def all_records(self):
        """returns all records in order of delta_jsd"""
        return [self.lowest] + self.records


def _get_stat_attribute(stat: str) -> str:
    if stat not in ("stdev", "cov"):
        raise ValueError(f"unknown value of stat {stat!r}")

    return {"stdev": "std_delta_jsd", "cov": "cov_delta_jsd"}[stat]


def max_divergent(
    records: list[KmerSeq],
    min_size: int = 2,
    max_size: int | None = None,
    stat: str = "stdev",
    max_set: bool = True,
    verbose: bool = False,
) -> SummedRecords:
    """returns SummedRecords that maximises stat

    Parameters
    ----------
    records
        list of SeqRecord instances
    min_size
        starting size of SummedRecords
    max_size
        defines upper limit of SummedRecords size
    stat
        either stdev or cov, which represent the statistics
        std(delta_jsd) and cov(delta_jsd) respectively
    max_set
        postprocess to identify subset that maximises stat

    Notes
    -----
    This is sensitive to the order of records.
    """
    max_size = max_size or len(records)
    sr = SummedRecords.from_records(records[:min_size])

    attr = _get_stat_attribute(stat)

    if len(records) <= min_size:
        return sr

    series = rich_progress.track(records, transient=True) if verbose else records
    for r in series:
        if r in sr:
            # already a member of the divergent set
            continue

        if not sr.increases_jsd(r):
            # does not increase total JSD
            continue

        nsr = sr + r
        sr = nsr if getattr(nsr, attr) > getattr(sr, attr) else sr.replaced_lowest(r)
        if sr.size > max_size:
            sr = SummedRecords.from_records(sr.records)

    if max_set:
        app = dvs_final_max(stat=stat, min_size=min_size, verbose=verbose)  # pylint: disable=no-value-for-parameter
        sr = app([sr])
    elif verbose:
        num_neg = sum(r.delta_jsd < 0 for r in [sr.lowest] + sr.records)
        print(f"Records with delta_jsd < 0 is {num_neg}")
    return sr


@define_app
def dvs_final_max(
    summed: list[SummedRecords],
    *,
    stat: str,
    min_size: int,
    verbose: bool,
) -> SummedRecords:
    """returns the set that maximises stat

    Parameters
    ----------
    sr
        SummedRecords instance
    stat
        name of a SummedRecords attribute that returns the statistic
    min_size
        the minimum size of the set
    verbose
        display extra information
    """
    if len(summed) > 1:
        records = list(itertools.chain.from_iterable(sr.all_records() for sr in summed))
        numpy.random.shuffle(records)
        sr = SummedRecords.from_records(records)
    else:
        sr = summed[0]

    if sr.size == min_size:
        return sr

    stat = _get_stat_attribute(stat)

    results = {getattr(sr, stat): sr}
    orig_size = sr.size
    orig_stat = getattr(sr, stat)

    while sr.size > min_size:
        sr = sr.from_records(sr.records)
        results[getattr(sr, stat)] = sr

    sr = results[max(results)]

    if verbose:
        print(
            f"Reduced size from {orig_size} to {sr.size}",
            f" Increased {stat} from {orig_stat:.3f} to {getattr(sr, stat):.3f}",
            sep="\n",
        )
    return sr


def most_divergent(
    records: list[KmerSeq],
    size: int,
    verbose: bool = False,
    show_progress: bool = False,
) -> SummedRecords:
    """returns size most divergent records

    Parameters
    ----------
    records
        list of SeqRecord instances
    size
        starting size of SummedRecords
    show_progress
        display progress bar
    """
    size = size or 2
    sr = SummedRecords.from_records(records[:size])

    if len(records) <= size:
        return sr

    series = rich_progress.track(records) if show_progress else records
    for r in series:
        if r in sr:
            continue

        if not sr.increases_jsd(r):
            continue

        sr = sr.replaced_lowest(r)

    if verbose:
        num_neg = sum(r.delta_jsd < 0 for r in [sr.lowest] + sr.records)
        print(f"number negative is {num_neg}")
    return sr


@define_app
class dvs_max:
    """return the maximally divergent sequences"""

    def __init__(
        self,
        *,
        seq_store: str | pathlib.Path,
        k: int = 3,
        min_size: int = 7,
        max_size: int = None,
        stat: str = "stdev",
        limit: int = None,
        verbose=0,
    ) -> None:
        """

        Parameters
        ----------
        seq_store
            path to divergent sequence store
        k
            k-mer size
        min_size
            the minimum number of sequences to be included in the divergent set
        max_size
            the maximum number of sequences to be included in the divergent set
        stat
            the stat to use for selecting whether to include a sequence
        limit
            limit number of sequence records to process
        stat
            the statistic to use for optimising, by default "cov_delta_jsd"
        verbose
            extra info display
        """
        self._seq_store = seq_store
        self._k = k
        self._limit = limit
        self._max_size = max_size
        self._min_size = min_size
        self._stat = stat
        self._verbose = verbose

    def main(self, seq_names: list[str]) -> SummedRecords:
        records = records_from_seq_store(
            seq_store=self._seq_store,
            seq_names=seq_names,
            limit=self._limit,
            k=self._k,
        )
        # TODO: add ability to set random number seed
        numpy.random.shuffle(records)
        return max_divergent(
            records=records,
            min_size=self._min_size,
            max_size=self._max_size,
            stat=self._stat,
            verbose=self._verbose > 0,
            max_set=False,
        )


def records_from_seq_store(
    *,
    seq_store: str | pathlib.Path,
    seq_names: list[str],
    k: int,
    moltype: str = "dna",
    limit: int | None,
) -> list[KmerSeq]:
    """converts sequences in seq_store into SeqRecord instances

    Parameters
    ----------
    seq_store
        path to divergent sequence store
    seq_names
        list of names that are members of the seq_store
    moltype
        the expected molecular type
    limit
        limit number of sequence records to process
    k
        k-mer size

    Returns
    -------
        sequences are converted into vector of k-mer counts
    """
    dstore = dvs_data_store.HDF5DataStore(seq_store, mode="r")
    make_record = member_to_kmerseq(k=k, moltype=moltype)
    records = [make_record(m) for m in dstore.completed if m.unique_id in seq_names]  # pylint: disable=not-callable
    records = records[:limit] if limit else records
    for record in records:
        if not record:
            print(record)
            sys.exit(1)
    return records


@define_app
class dvs_nmost:
    """return the N most divergent sequences"""

    def __init__(
        self,
        *,
        seq_store: str | pathlib.Path,
        k: int = 3,
        limit: int = None,
        n: int,
        verbose=0,
    ) -> None:
        """

        Parameters
        ----------
        seq_store
            path to divergent sequence store
        n
            the number of divergent sequences
        k
            k-mer size
        limit
            limit number of sequence records to process
        verbose
            extra info display
        """
        self._seq_store = seq_store
        self._k = k
        self._limit = limit
        self._max_size = n
        self._verbose = verbose

    def main(self, seq_names: list[str]) -> SummedRecords:
        records = records_from_seq_store(
            seq_store=self._seq_store,
            seq_names=seq_names,
            limit=self._limit,
            k=self._k,
        )
        # TODO: add ability to set random number seed
        numpy.random.shuffle(records)
        return most_divergent(records, size=self._max_size, verbose=self._verbose > 0)


@define_app
def dvs_final_nmost(summed: list[SummedRecords]) -> SummedRecords:
    """selects the best n records from a list of SummedRecords

    Notes
    -----
    Useful for aggregating results from multiple runs.
    """
    size = max(sr.size for sr in summed)
    records = list(itertools.chain.from_iterable(sr.all_records() for sr in summed))
    numpy.random.shuffle(records)
    return most_divergent(records, size=size, verbose=False, show_progress=False)


def apply_app(
    *,
    app: dvs_max,
    seqids: list[str],
    numprocs: int,
    verbose: bool,
    hide_progress: bool = False,
    finalise: typing.Callable[[list[SummedRecords]], SummedRecords],
) -> SummedRecords:
    """applies the app to seqids, polishing the selected set with finalise"""
    app_name = app.__class__.__name__
    with dvs_util.keep_running():
        if numprocs > 1 and len(seqids) > numprocs:
            seqids = list(dvs_util.chunked(seqids, numprocs, verbose=verbose))
        else:
            seqids = [seqids]

        with rich_progress.Progress(
            rich_progress.TextColumn("{task.description}"),
            rich_progress.BarColumn(),
            rich_progress.TaskProgressColumn(),
            rich_progress.TimeRemainingColumn(),
            rich_progress.TimeElapsedColumn(),
            disable=hide_progress,
        ) as progress:
            select = progress.add_task(
                description=f"[blue]Selection with {app_name!r}",
                total=len(seqids),
            )
            result = []
            for r in app.as_completed(
                seqids,
                parallel=numprocs > 1,
                par_kw=dict(max_workers=numprocs),
                show_progress=True,
            ):
                if not r:
                    dvs_util.print_colour(r, style="red")
                result.append(r.obj)
                progress.update(select, advance=1, refresh=True)

        dvs_util.print_colour(f"Merging results from {len(seqids)} runs...", "blue")

        result = finalise(result)

        if isinstance(result, NotCompleted):
            dvs_util.print_colour(f"{result.type}: {result.message}", "red")
            sys.exit(1)

    return result


# single apps that combine multiple apps to work off a sequence collection
# app convert seq coll to SequenceArray and converts those into KmerSeq instances


@define_app
class dvs_select_max:
    """selects the maximally divergent seqs from a sequence collection"""

    def __init__(
        self,
        min_size: int = 5,
        max_size: int = 30,
        stat: str = "stdev",
        moltype: str = "dna",
        include: list[str] | str | None = None,
        k: int = 6,
        seed: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        min_size
            minimum size of the divergent set
        max_size
            the maximum size of the divergent set
        stat
            either stdev or cov, which represent the statistics
            std(delta_jsd) and cov(delta_jsd) respectively
        moltype
            molecular type of the sequences
        include
            sequence names to include in the final result
        k
            k-mer size
        seed
            random number seed

        Notes
        -----
        If called with an alignment, the ungapped sequences are used.
        The order of the sequences is randomised. If include is not None, the
        named sequences are added to the final result.
        """
        self._s2k = seq_to_seqarray(moltype=moltype) + seqarray_to_kmerseq(
            k=k,
            moltype=moltype,
        )
        self._min_size = min_size
        self._max_size = max_size
        self._stat = stat
        self._rng = numpy.random.default_rng(seed)
        self._include = [include] if isinstance(include, str) else include

    def main(self, seqs: c3_types.SeqsCollectionType) -> c3_types.SeqsCollectionType:
        records = [self._s2k(seqs.get_seq(name)) for name in seqs.names]
        self._rng.shuffle(records)
        for record in records:
            if not record:
                print(record)
                return None
        result = max_divergent(
            records=records,
            min_size=self._min_size,
            max_size=self._max_size,
            stat=self._stat,
        )
        selected = set(result.record_names) | set(self._include or [])
        return seqs.take_seqs(selected)


@define_app
class dvs_select_nmost:
    """selects the n-most diverse seqs from a sequence collection"""

    def __init__(
        self,
        n: int = 10,
        moltype: str = "dna",
        include: list[str] | str | None = None,
        k: int = 6,
        seed: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        n
            the number of divergent sequences
        moltype
            molecular type of the sequences
        k
            k-mer size
        include
            sequence names to include in the final result
        seed
            random number seed

        Notes
        -----
        If called with an alignment, the ungapped sequences are used.
        The order of the sequences is randomised. If include is not None, the
        named sequences are added to the final result.
        """
        self._s2k = seq_to_seqarray(moltype=moltype) + seqarray_to_kmerseq(
            k=k,
            moltype=moltype,
        )
        self._n = n
        self._moltype = moltype
        self._rng = numpy.random.default_rng(seed)
        self._include = [include] if isinstance(include, str) else include

    def main(self, seqs: c3_types.SeqsCollectionType) -> c3_types.SeqsCollectionType:
        records = [self._s2k(seqs.get_seq(name)) for name in seqs.names]
        self._rng.shuffle(records)
        result = most_divergent(
            records=records,
            size=self._n,
        )
        selected = set(result.record_names) | set(self._include or [])
        return seqs.take_seqs(selected)
