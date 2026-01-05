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

import itertools
import pathlib
import sys
import typing

import numpy
from cogent3 import get_moltype
from cogent3.app import typing as c3_types
from cogent3.app.composable import NotCompleted, define_app
from rich import progress as rich_progress

from diverse_seq import _dvs as dvs
from diverse_seq import util as dvs_util

# needs a jsd method for a new sequence
# needs summed entropy scores
# needs combined array of summed freqs to allow efficient calculation of entropy
# on creation, computes total JSD, then delta_jsd (contribution of each sequence)
# seq records sorted by the latter; stored summed quantities for n and n-1 (with
# the least contributor omitted)

AppOutType = typing.Union[c3_types.SeqsCollectionType, c3_types.SerialisableType]


@define_app
def select_final_max(
    summed: list[dvs.SummedRecordsResult],
    *,
    seq_store: str | pathlib.Path,
    stat: str,
    min_size: int,
    max_size: int,
    k: int,
    num_states: int,
) -> dvs.SummedRecordsResult:
    """returns the set that maximises stat

    Parameters
    ----------
    summed
        list of SummedRecordsResult instances
    stat
        name of the statistic to be maximised
    min_size
        the minimum size of the set
    max_size
        the maximum size of the set
    k
        k-mer size
    num_states
        number of canonical characters in the moltype
    """
    if len(summed) > 1:
        records = list(itertools.chain.from_iterable(sr.record_names for sr in summed))
    else:
        records = summed[0].record_names

    max_size = max_size or len(records)
    numpy.random.shuffle(records)
    seq_store = dvs.make_zarr_store(str(seq_store))
    return dvs.max_divergent(
        seq_store,
        min_size=min_size,
        max_size=max_size,
        k=k,
        num_states=num_states,
        seqids=records,
        stat=stat,
    )


@define_app
class select_max:
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
        num_states: int = 4,
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
        num_states
            number of canonical characters in the moltype
        """
        self._seq_store = dvs.make_zarr_store(str(seq_store))
        self._k = k
        self._limit = limit
        self._max_size = max_size
        self._min_size = min_size
        self._stat = stat
        self._num_states = num_states

    def main(self, seq_names: list[str]) -> dvs.SummedRecordsResult:
        if self._limit:
            seq_names = seq_names[: self._limit]
        max_size = self._max_size or len(seq_names)
        return dvs.max_divergent(
            self._seq_store,
            min_size=self._min_size,
            max_size=max_size,
            k=self._k,
            num_states=self._num_states,
            seqids=seq_names,
            stat=self._stat,
        )


@define_app
class select_nmost:
    """return the N most divergent sequences"""

    def __init__(
        self,
        *,
        seq_store: str | pathlib.Path,
        k: int = 3,
        limit: int | None = None,
        n: int,
        num_states: int = 4,
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
        num_states
            number of canonical characters in the moltype
        """
        self._seq_store = dvs.make_zarr_store(str(seq_store))
        self._k = k
        self._limit = limit
        self._max_size = n
        self._num_states = num_states

    def main(self, seq_names: list[str]) -> dvs.SummedRecordsResult:
        if self._limit:
            seq_names = seq_names[: self._limit]

        return dvs.nmost_divergent(
            self._seq_store,
            n=self._max_size,
            k=self._k,
            num_states=self._num_states,
            seqids=seq_names,
        )


@define_app
def dvs_final_nmost(
    summed: list[dvs.SummedRecordsResult], seq_store: str | pathlib.Path
) -> dvs.SummedRecordsResult:
    """selects the best n records from a list of SummedRecords

    Notes
    -----
    Useful for aggregating results from multiple runs.
    """
    if not summed:
        return NotCompleted(
            "ERROR",
            origin="dvs_final_nmost",
            message="no SummedRecords instances were provided",
            source="Unknown",
        )
    seq_store = dvs.make_zarr_store(str(seq_store))
    size = max(sr.size for sr in summed)
    k = summed[0].k
    num_states = summed[0].num_states
    seqids = list(itertools.chain.from_iterable(sr.record_names for sr in summed))
    numpy.random.shuffle(seqids)  # noqa: NPY002
    return dvs.nmost_divergent(
        seq_store, n=size, k=k, num_states=num_states, seqids=seqids
    )


def apply_app(
    *,
    app: select_max,
    seqids: list[str],
    numprocs: int,
    verbose: bool,
    hide_progress: bool = False,
    finalise: typing.Callable[[list[dvs.SummedRecordsResult]], dvs.SummedRecordsResult],
) -> dvs.SummedRecordsResult:
    """applies the app to seqids, polishing the selected set with finalise"""
    if verbose and not hide_progress:
        dvs_util.print_colour(
            "Cannot show progress bar and verbose. "
            "Either hide_progress or disable verbose.",
            "red",
        )
        sys.exit(1)

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
                par_kw={"max_workers": numprocs},
                show_progress=False,
            ):
                if not r:
                    dvs_util.print_colour(str(r), "red")
                result.append(r.obj)
                progress.update(select, advance=1, refresh=True)

        if len(seqids) > 1:
            dvs_util.print_colour(f"Merging results from {len(seqids)} runs...", "blue")

        result = finalise(result)

        if isinstance(result, NotCompleted):
            dvs_util.print_colour(f"{result.type}: {result.message}", "red")
            sys.exit(1)

    return result


@define_app
class dvs_max:  # done
    """select the maximally divergent seqs from a sequence collection"""

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

        Returns
        -------
        The same type as the input sequence collection.
        """
        self._k = k
        self._moltype = moltype
        self._num_states = len(get_moltype(moltype).alphabet)
        self._min_size = min_size
        self._max_size = max_size
        self._stat = stat
        self._rng = numpy.random.default_rng(seed)
        self._include = [include] if isinstance(include, str) else include

    def main(self, seqs: c3_types.SeqsCollectionType) -> AppOutType:
        # make an in-memory ZarrStoreWrapper
        zstore = dvs_util.populate_inmem_zstore(seqs)
        seqids = list(zstore.unique_seqids)
        self._rng.shuffle(seqids)
        result = dvs.max_divergent(
            zstore,
            min_size=self._min_size,
            max_size=self._max_size,
            k=self._k,
            num_states=self._num_states,
            seqids=seqids,
            stat=self._stat,
        )
        selected = set(result.record_names) | set(self._include or [])
        return seqs.take_seqs(selected)


@define_app
class dvs_nmost:  # done
    """select the n-most diverse seqs from a sequence collection"""

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

        Returns
        -------
        The same type as the input sequence collection.
        """
        self._k = k
        self._n = n
        self._moltype = moltype
        self._rng = numpy.random.default_rng(seed)
        self._include = [include] if isinstance(include, str) else include

    def main(self, seqs: c3_types.SeqsCollectionType) -> AppOutType:
        # make an in-memory ZarrStoreWrapper
        zstore = dvs_util.populate_inmem_zstore(seqs)
        seqids = list(zstore.unique_seqids)
        self._rng.shuffle(seqids)
        result = dvs.nmost_divergent(zstore, n=self._n, k=self._k, seqids=seqids)
        selected = set(result.record_names) | set(self._include or [])
        return seqs.take_seqs(selected)


@define_app
class dvs_delta_jsd:  # done
    """returns delta_jsd for a sequence"""

    def __init__(
        self,
        seqs: c3_types.SeqsCollectionType,
        moltype: str = "dna",
        k: int = 6,
    ) -> None:
        """
        Parameters
        ----------
        seqs
            the sequence collection to use as the reference set
        moltype
            molecular type of the sequences
        k
            k-mer size

        Notes
        -----
        If called with an alignment, the ungapped sequences are used.
        The order of the sequences is randomised. If include is not None, the
        named sequences are added to the final result.

        If the query sequence has zero length, returns nan.

        Returns
        -------
        (sequence name, delta JSD)
        """
        degapped = seqs.degap()
        if (lengths := degapped.get_lengths()).array.min() == 0:
            zero_len = ", ".join([k for k, c in lengths.items() if c == 0])
            msg = f"cannot compute delta_jsd with zero-length sequences: {zero_len}"
            raise ValueError(msg)

        self.moltype = moltype
        records = [(s.name, numpy.array(s).tobytes()) for s in degapped.seqs]
        mtype = get_moltype(moltype)
        num_states = len(mtype.alphabet)
        self._sr = dvs.get_delta_jsd_calculator(records, k, num_states)

    def main(self, seq: c3_types.SeqType) -> tuple[str, float]:
        if seq.moltype.name != self.moltype:
            seq = seq.to_moltype(self.moltype)

        seq = seq.degap()
        if len(seq) == 0:
            return seq.name, numpy.nan

        delta = self._sr.delta_jsd(seq.name, numpy.array(seq).tobytes())
        return seq.name, delta
