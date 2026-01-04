"""Apps and methods used to compute kmer cluster trees for sequences."""

import multiprocessing
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import cogent3.app.typing as c3_types
import numpy
from cogent3 import PhyloNode, make_tree
from cogent3.app.composable import AppType, define_app
from loky import as_completed, get_reusable_executor
from rich.progress import Progress
from scipy.sparse import dok_matrix
from sklearn.cluster import AgglomerativeClustering

from diverse_seq import data_store as dvs_data_store
from diverse_seq.distance import (
    BottomSketch,
    euclidean_distance,
    euclidean_distances,
    mash_distance,
    mash_distances,
    mash_sketch,
)
from diverse_seq.record import (
    KmerSeq,
    SeqArray,
    _get_canonical_states,
    make_kmerseq,
    seq_to_seqarray,
)


class ClusterTreeBase:
    def __init__(
        self,
        *,
        k: int = 16,
        sketch_size: int | None = None,
        moltype: str = "dna",
        distance_mode: Literal["mash", "euclidean"] = "mash",
        mash_canonical_kmers: bool | None = None,
        show_progress: bool = False,
    ) -> None:
        """Initialise parameters for generating a kmer cluster tree.

        Parameters
        ----------
        k
            kmer size
        sketch_size
            size of sketches
        moltype
            seq collection molecular type
        distance_mode
            mash distance or euclidean distance between kmer freqs
        mash_canonical_kmers
            whether to use mash canonical kmers for mash distance
        show_progress
            whether to show progress bars

        Notes
        -----
        If mash_canonical_kmers is enabled when using the mash distance,
        kmers are considered identical to their reverse complement.

        References
        ----------
        .. [1] Ondov, B. D., Treangen, T. J., Melsted, P., Mallonee, A. B.,
           Bergman, N. H., Koren, S., & Phillippy, A. M. (2016).
           Mash: fast genome and metagenome distance estimation using MinHash.
           Genome biology, 17, 1-14.
        """
        if mash_canonical_kmers is None:
            mash_canonical_kmers = False

        if distance_mode not in ("mash", "euclidean"):
            msg = f"Unexpected distance {distance_mode!r}."
            raise ValueError(msg)

        if moltype not in ("dna", "rna") and mash_canonical_kmers:
            msg = "Canonical kmers only supported for dna/rna sequences."
            raise ValueError(msg)

        if distance_mode == "mash" and sketch_size is None:
            msg = "Expected sketch size for mash distance measure."
            raise ValueError(msg)

        self._moltype = moltype
        self._k = k
        self._num_states = len(_get_canonical_states(self._moltype))
        self._sketch_size = sketch_size
        self._distance_mode = distance_mode
        self._mash_canonical = mash_canonical_kmers
        self._progress = Progress(disable=not show_progress)

        self._s2a = seq_to_seqarray(moltype=moltype)


@define_app
class dvs_ctree(ClusterTreeBase):
    """Create a cluster tree from kmer distances."""

    def __init__(
        self,
        *,
        k: int = 12,
        sketch_size: int | None = 3_000,
        moltype: str = "dna",
        distance_mode: Literal["mash", "euclidean"] = "mash",
        mash_canonical_kmers: bool | None = None,
        show_progress: bool = False,
    ) -> None:
        """Initialise parameters for generating a kmer cluster tree.

        Parameters
        ----------
        k
            kmer size
        sketch_size
            size of sketches, only applies to mash distance
        moltype
            seq collection molecular type
        distance_mode
            mash distance or euclidean distance between kmer freqs
        mash_canonical_kmers
            whether to use mash canonical kmers for mash distance
        show_progress
            whether to show progress bars

        Notes
        -----
        This app is composable.

        If mash_canonical_kmers is enabled when using the mash distance,
        kmers are considered identical to their reverse complement.

        References
        ----------
        .. [1] Ondov, B. D., Treangen, T. J., Melsted, P., Mallonee, A. B.,
           Bergman, N. H., Koren, S., & Phillippy, A. M. (2016).
           Mash: fast genome and metagenome distance estimation using MinHash.
           Genome biology, 17, 1-14.
        """
        super().__init__(
            k=k,
            sketch_size=sketch_size,
            moltype=moltype,
            distance_mode=distance_mode,
            mash_canonical_kmers=mash_canonical_kmers,
            show_progress=show_progress,
        )

    def main(self, seqs: c3_types.SeqsCollectionType) -> PhyloNode:
        """Construct a cluster tree for a collection of sequences.

        Parameters
        ----------
        seqs
            Sequence collection to form cluster tree for.

        Returns
        -------
        PhyloNode
            a cluster tree.
        """
        seqs = seqs.degap()
        seq_names = seqs.names
        seq_arrays = [self._s2a(seqs.get_seq(name)) for name in seq_names]  # pylint: disable=not-callable

        with self._progress:
            if self._distance_mode == "mash":
                distances = mash_distances(
                    seq_arrays,
                    self._k,
                    self._sketch_size,
                    self._num_states,
                    mash_canonical=self._mash_canonical,
                    progress=self._progress,
                )
            elif self._distance_mode == "euclidean":
                distances = euclidean_distances(
                    seq_arrays,
                    self._k,
                    self._moltype,
                    progress=self._progress,
                )
            return make_cluster_tree(seq_names, distances, progress=self._progress)


def make_cluster_tree(
    seq_names: Sequence[str],
    pairwise_distances: numpy.ndarray,
    *,
    progress: Progress | None = None,
) -> PhyloNode:
    """Given pairwise distances between sequences, construct a cluster tree.

    Parameters
    ----------
    seq_names
        Names of sequences to cluster.
    pairwise_distances
        Pairwise distances between clusters.
    progress
        Progress bar, by default None.

    Returns
    -------
    PhyloNode
        The cluster tree.
    """
    if progress is None:
        progress = Progress(disable=True)

    clustering = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
    )

    tree_task = progress.add_task(
        "[green]Computing Tree",
        total=None,
    )

    clustering.fit(pairwise_distances)

    tree_dict = {i: seq_names[i] for i in range(len(seq_names))}
    node_index = len(seq_names)
    for left_index, right_index in clustering.children_:
        tree_dict[node_index] = (
            tree_dict.pop(left_index),
            tree_dict.pop(right_index),
        )
        node_index += 1
    # use string representation and then remove quotes
    treestring = str(tree_dict[node_index - 1]).replace("'", "")
    tree = make_tree(treestring=treestring, underscore_unmunge=True)
    progress.update(tree_task, completed=1, total=1)
    return tree


class DvsParCtreeMixin:
    def _mash_dist(self, seq_arrays: Sequence[SeqArray]) -> numpy.ndarray:
        """Calculates pairwise mash distances between sequences in parallel.

        Uses the number of processes specified in the constructor. If one process was
        specified runs in serial.

        Parameters
        ----------
        seq_arrays
            Sequence arrays.

        Returns
        -------
        numpy.ndarray
            Pairwise mash distances between sequences.
        """
        if self._numprocs == 1:
            return mash_distances(
                seq_arrays,
                self._k,
                self._sketch_size,
                self._num_states,
                mash_canonical=self._mash_canonical,
                progress=self._progress,
            )

        sketches = self.mash_sketches_parallel(seq_arrays)

        distance_task = self._progress.add_task(
            "[green]Computing Pairwise Distances",
            total=None,
        )

        # Compute distances in parallel
        num_jobs = self._numprocs
        futures = [
            self._executor.submit(
                compute_mash_chunk_distances,
                start,
                num_jobs,
                sketches,
                self._k,
                self._sketch_size,
            )
            for start in range(num_jobs)
        ]

        distances = numpy.zeros((len(sketches), len(sketches)))

        for future in futures:
            start_idx, distances_chunk = future.result()
            distances[start_idx::num_jobs] = distances_chunk[
                start_idx::num_jobs
            ].toarray()

        # Make lower triangular matrix symmetric
        distances = distances + distances.T - numpy.diag(distances.diagonal())

        self._progress.update(distance_task, completed=1, total=1)

        return distances

    def _euclidean_dist(
        self,
        seq_arrays: Sequence[SeqArray],
    ) -> numpy.ndarray:
        """Calculates pairwise euclidean distances between sequences in parallel.

        Uses the number of processes specified in the constructor. If one process was
        specified runs in serial.

        Parameters
        ----------
        seq_arrays
            Sequence arrays.

        Returns
        -------
        numpy.ndarray
            Pairwise euclidean distances between sequences.
        """
        if self._numprocs == 1:
            return euclidean_distances(
                seq_arrays,
                self._k,
                self._moltype,
                progress=self._progress,
            )

        distance_task = self._progress.add_task(
            "[green]Computing Pairwise Distances",
            total=None,
        )

        kmer_seqs = [
            make_kmerseq(
                seq,
                dtype=numpy.min_scalar_type(
                    len(_get_canonical_states(self._moltype)) ** self._k,
                ),
                k=self._k,
                moltype=self._moltype,
            )
            for seq in seq_arrays
        ]

        # Compute distances in parallel
        num_jobs = self._numprocs
        futures = [
            self._executor.submit(
                compute_euclidean_chunk_distances,
                start,
                num_jobs,
                kmer_seqs,
            )
            for start in range(num_jobs)
        ]

        distances = numpy.zeros((len(kmer_seqs), len(kmer_seqs)))

        for future in futures:
            start_idx, distances_chunk = future.result()
            distances[start_idx::num_jobs] = distances_chunk[
                start_idx::num_jobs
            ].toarray()

        # Make lower triangular matrix symmetric
        distances = distances + distances.T - numpy.diag(distances.diagonal())

        self._progress.update(distance_task, completed=1, total=1)
        return distances

    def mash_sketches_parallel(
        self,
        seq_arrays: Sequence[SeqArray],
    ) -> list[BottomSketch]:
        """Create sketch representations for a collection of sequence records in parallel.

        Parameters
        ----------
        seq_arrays
            Sequence arrays.

        Returns
        -------
        list[BottomSketch]
            Sketches for each sequence.
        """
        sketch_task = self._progress.add_task(
            "[green]Generating Sketches",
            total=len(seq_arrays),
        )

        bottom_sketches = [None for _ in range(len(seq_arrays))]

        # Compute sketches in parallel
        futures_to_idx = {
            self._executor.submit(
                mash_sketch,
                seq_array.data,
                self._k,
                self._sketch_size,
                self._num_states,
                mash_canonical=self._mash_canonical,
            ): i
            for i, seq_array in enumerate(seq_arrays)
        }

        for future in as_completed(list(futures_to_idx)):
            idx = futures_to_idx[future]
            bottom_sketches[idx] = future.result()

            self._progress.update(sketch_task, advance=1)

        return bottom_sketches


@define_app(app_type=AppType.NON_COMPOSABLE)
class dvs_par_ctree(ClusterTreeBase, DvsParCtreeMixin):
    """Create a cluster tree from kmer distances in parallel."""

    def __init__(
        self,
        *,
        k: int = 12,
        sketch_size: int | None = 3000,
        moltype: str = "dna",
        distance_mode: Literal["mash", "euclidean"] = "mash",
        mash_canonical_kmers: bool | None = None,
        show_progress: bool = False,
        max_workers: int | None = None,
        parallel: bool = True,
    ) -> None:
        """Initialise parameters for generating a kmer cluster tree.

        Parameters
        ----------
        k
            kmer size
        sketch_size
            size of sketches, only applies to mash distance
        moltype
            seq collection molecular type
        distance_mode
            mash distance or euclidean distance between kmer freqs
        mash_canonical_kmers
            whether to use mash canonical kmers for mash distance
        show_progress
            whether to show progress bars
        numprocs
            number of workers, defaults to running serial

        Notes
        -----
        This is app is not composable but can run in parallel. It is
        best suited to a single large sequence collection.

        If mash_canonical_kmers is enabled when using the mash distance,
        kmers are considered identical to their reverse complement.

        References
        ----------
        .. [1] Ondov, B. D., Treangen, T. J., Melsted, P., Mallonee, A. B.,
           Bergman, N. H., Koren, S., & Phillippy, A. M. (2016).
           Mash: fast genome and metagenome distance estimation using MinHash.
           Genome biology, 17, 1-14.
        """
        super().__init__(
            k=k,
            sketch_size=sketch_size,
            moltype=moltype,
            distance_mode=distance_mode,
            mash_canonical_kmers=mash_canonical_kmers,
            show_progress=show_progress,
        )

        if parallel:
            if max_workers is None:
                max_workers = multiprocessing.cpu_count()
            self._numprocs = min(max_workers, multiprocessing.cpu_count())
        else:
            self._numprocs = 1

        self._calc_dist = (
            self._mash_dist if distance_mode == "mash" else self._euclidean_dist
        )

    def main(self, seqs: c3_types.SeqsCollectionType) -> PhyloNode:
        """Construct a cluster tree for a collection of sequences.

        Parameters
        ----------
        seqs
            Sequence collection to form cluster tree for.

        Returns
        -------
        PhyloNode
            a cluster tree.
        """
        seqs = seqs.degap()
        self._executor = (
            get_reusable_executor(max_workers=self._numprocs)
            if self._numprocs != 1
            else nullcontext()
        )

        seq_names = seqs.names
        seq_arrays = [self._s2a(seqs.get_seq(name)) for name in seq_names]  # pylint: disable=not-callable

        with self._progress, self._executor:
            distances = self._calc_dist(seq_arrays)
            return make_cluster_tree(seq_names, distances, progress=self._progress)


@define_app(app_type=AppType.NON_COMPOSABLE)
class dvs_cli_par_ctree(ClusterTreeBase, DvsParCtreeMixin):
    """Create a cluster tree from kmer distances in parallel."""

    def __init__(
        self,
        *,
        seq_store: str | Path,
        limit: int | None = None,
        k: int = 12,
        sketch_size: int | None = 3000,
        moltype: str = "dna",
        distance_mode: Literal["mash", "euclidean"] = "mash",
        mash_canonical_kmers: bool | None = None,
        show_progress: bool = False,
        max_workers: int | None = None,
        parallel: bool = True,
    ) -> None:
        """Initialise parameters for generating a kmer cluster tree.

        Parameters
        ----------
        k
            kmer size
        sketch_size
            size of sketches, only applies to mash distance
        moltype
            seq collection molecular type
        distance_mode
            mash distance or euclidean distance between kmer freqs
        mash_canonical_kmers
            whether to use mash canonical kmers for mash distance
        show_progress
            whether to show progress bars
        numprocs
            number of workers, defaults to running serial

        Notes
        -----
        This is app is not composable but can run in parallel. It is
        best suited to a single large sequence collection.

        If mash_canonical_kmers is enabled when using the mash distance,
        kmers are considered identical to their reverse complement.

        References
        ----------
        .. [1] Ondov, B. D., Treangen, T. J., Melsted, P., Mallonee, A. B.,
           Bergman, N. H., Koren, S., & Phillippy, A. M. (2016).
           Mash: fast genome and metagenome distance estimation using MinHash.
           Genome biology, 17, 1-14.
        """
        super().__init__(
            k=k,
            sketch_size=sketch_size,
            moltype=moltype,
            distance_mode=distance_mode,
            mash_canonical_kmers=mash_canonical_kmers,
            show_progress=show_progress,
        )

        self._seq_store = seq_store
        self._limit = limit

        if parallel:
            if max_workers is None:
                max_workers = multiprocessing.cpu_count()
            self._numprocs = min(max_workers, multiprocessing.cpu_count())
        else:
            self._numprocs = 1

        self._calc_dist = (
            self._mash_dist if distance_mode == "mash" else self._euclidean_dist
        )

    def main(self, seq_names: list[str]) -> PhyloNode:
        """Construct a cluster tree for a collection of sequences.

        Parameters
        ----------
        seqs
            Sequence collection to form cluster tree for.

        Returns
        -------
        PhyloNode
            a cluster tree.
        """
        dstore = dvs_data_store.HDF5DataStore(self._seq_store, mode="r")

        self._executor = (
            get_reusable_executor(max_workers=self._numprocs)
            if self._numprocs != 1
            else nullcontext()
        )

        seq_arrays = {
            m.unique_id: m  # pylint: disable=not-callable
            for m in dstore.completed
            if m.unique_id in seq_names
        }
        seq_arrays = [seq_arrays[name] for name in seq_names]
        seq_arrays = seq_arrays[: self._limit] if self._limit else seq_arrays
        seq_arrays = [
            SeqArray(
                member.unique_id,
                member.read(),
                self._moltype,
                member.data_store.source,
            )
            for member in seq_arrays
        ]

        with self._progress, self._executor:
            distances = self._calc_dist(seq_arrays)
            return make_cluster_tree(seq_names, distances, progress=self._progress)


def compute_mash_chunk_distances(
    start_idx: int,
    stride: int,
    sketches: list[BottomSketch],
    k: int,
    sketch_size: int,
) -> tuple[int, dok_matrix]:
    """Find a subset of pairwise distances between sketches.

    Finds pairwise distances for the row with the given start index
    and each stride after that.

    Only finds lower triangular portion.

    Parameters
    ----------
    start_idx
        Start index for distance calculation.
    stride
        Index increment.
    sketches
        Sketches for pairwise distances.
    k
        kmer size.
    sketch_size
        Size of the sketches.

    Returns
    -------
    tuple[int, dok_matrix]
        The start index, and the calculated pairwise distances.
    """
    distances_chunk = dok_matrix((len(sketches), len(sketches)))
    for i in range(start_idx, len(sketches), stride):
        for j in range(i):
            distance = mash_distance(sketches[i], sketches[j], k, sketch_size)
            distances_chunk[i, j] = distance
    return start_idx, distances_chunk


def compute_euclidean_chunk_distances(
    start_idx: int,
    stride: int,
    kmer_seqs: list[KmerSeq],
) -> tuple[int, dok_matrix]:
    """Find a subset of pairwise distances between sketches.

    Finds pairwise distances for the row with the given start index
    and each stride after that.

    Only finds lower triangular portion.

    Parameters
    ----------
    start_idx
        Start index for distance calculation.
    stride
        Index increment.
    kmer_seqs
        kmer sequences.

    Returns
    -------
    tuple[int, dok_matrix]
        The start index, and the calculated pairwise distances.
    """
    distances_chunk = dok_matrix((len(kmer_seqs), len(kmer_seqs)))
    for i in range(start_idx, len(kmer_seqs), stride):
        freq_i = numpy.array(kmer_seqs[i].kfreqs)
        for j in range(i):
            freq_j = numpy.array(kmer_seqs[j].kfreqs)
            distance = euclidean_distance(freq_i, freq_j)
            distances_chunk[i, j] = distance
    return start_idx, distances_chunk
