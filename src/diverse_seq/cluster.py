"""Apps and methods used to compute kmer cluster trees for sequences."""

import warnings
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Literal

import cogent3.app.typing as c3_types
import numpy
from cogent3 import PhyloNode, make_tree
from cogent3.app.composable import AppType, define_app
from loky import as_completed, get_reusable_executor
from rich.progress import Progress
from scipy.sparse import dok_matrix
from sklearn.cluster import AgglomerativeClustering

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
        k : int, optional
            kmer size, by default 16
        sketch_size : int | None, optional
            size of sketches, by default None
        moltype : str, optional
            moltype, by default "dna"
        distance_mode : Literal[&quot;mash&quot;, &quot;euclidean&quot;], optional
            mash distance or euclidean distance between kmer freqs, by default "mash"
        mash_canonical_kmers : bool | None, optional
            whether to use mash canonical kmers for mash distance, by default False
        show_progress : bool, optional
            whether to show progress bars, by default False

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

        if distance_mode != "mash" and sketch_size is not None:
            warnings.warn(
                'Sketch size only applies when distance is "mash".',
                stacklevel=2,
            )

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
        k : int, optional
            kmer size, by default 16
        sketch_size : int | None, optional
            size of sketches, by default None
        moltype : str, optional
            moltype, by default "dna"
        distance_mode : Literal[&quot;mash&quot;, &quot;euclidean&quot;], optional
            mash distance or euclidean distance between kmer freqs, by default "mash"
        mash_canonical_kmers : bool | None, optional
            whether to use mash canonical kmers for mash distance, by default False
        show_progress : bool, optional
            whether to show progress bars, by default False

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
        seqs : list[c3_types.SeqsCollectionType]
            Sequence collection to form cluster tree for.

        Returns
        -------
        PhyloNode
            a cluster tree.
        """
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
    seq_names : Sequence[str]
        Names of sequences to cluster.
    pairwise_distances : numpy.ndarray
        Pairwise distances between clusters.
    progress : Progress | None, optional
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

    tree = make_tree(str(tree_dict[node_index - 1]))

    progress.update(tree_task, completed=1, total=1)

    return tree


@define_app(app_type=AppType.NON_COMPOSABLE)
class dvs_par_ctree(ClusterTreeBase):
    """Create a cluster tree from kmer distances.

    If numprocs>1, computations are performed in parallel.
    """

    def __init__(
        self,
        *,
        k: int = 16,
        sketch_size: int | None = None,
        moltype: str = "dna",
        distance_mode: Literal["mash", "euclidean"] = "mash",
        mash_canonical_kmers: bool | None = None,
        show_progress: bool = False,
        numprocs: int = 1,
    ) -> None:
        """Initialise parameters for generating a kmer cluster tree.

        Parameters
        ----------
        seq_store : str | pathlib.Path
            path to sequence store
        k : int, optional
            kmer size, by default 16
        sketch_size : int | None, optional
            size of sketches, by default None
        moltype : str, optional
            moltype, by default "dna"
        distance_mode : Literal["mash", "euclidean"], optional
            mash distance or euclidean distance between kmer freqs, by default "mash"
        mash_canonical_kmers : bool | None, optional
            whether to use mash canonical kmers for mash distance, by default False
        show_progress : bool, optional
            whether to show progress bars, by default False
        numprocs : int, optional
            number of workers, by default 1

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
        super().__init__(
            k=k,
            sketch_size=sketch_size,
            moltype=moltype,
            distance_mode=distance_mode,
            mash_canonical_kmers=mash_canonical_kmers,
            show_progress=show_progress,
        )

        self._numprocs = numprocs
        self._calc_dist = (
            self._mash_dist if distance_mode == "mash" else self._euclidean_dist
        )

    def main(self, seqs: c3_types.SeqsCollectionType) -> PhyloNode:
        """Construct a cluster tree for a collection of sequences.

        Parameters
        ----------
        seqs : list[c3_types.SeqsCollectionType]
            Sequence collection to form cluster tree for.

        Returns
        -------
        PhyloNode
            a cluster tree.
        """
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

    def _mash_dist(self, seq_arrays: Sequence[SeqArray]) -> numpy.ndarray:
        """Calculates pairwise mash distances between sequences in parallel.

        Uses the number of processes specified in the constructor. If one process was
        specified runs in serial.

        Parameters
        ----------
        seq_arrays : Sequence[SeqArray]
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
        seq_arrays : Sequence[SeqArray]
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
        seq_arrays : Sequence[SeqArray]
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
    start_idx : int
        Start index for distance calculation.
    stride : int
        Index increment.
    sketches : list[BottomSketch]
        Sketches for pairwise distances.
    k : int
        kmer size.
    sketch_size : int
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
    start_idx : int
        Start index for distance calculation.
    stride : int
        Index increment.
    kmer_seqs : list[KmerSeq]
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
