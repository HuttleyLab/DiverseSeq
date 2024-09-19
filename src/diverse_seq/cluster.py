"""Apps and methods used to compute kmer cluster trees for sequences."""

import pathlib
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Literal

import numpy
from cogent3 import PhyloNode, make_tree
from cogent3.app.composable import define_app
from cogent3.app.data_store import DataMember
from loky import as_completed, get_reusable_executor
from rich.progress import Progress
from scipy.sparse import dok_matrix
from sklearn.cluster import AgglomerativeClustering

from diverse_seq.data_store import HDF5DataStore, get_ordered_records
from diverse_seq.distance import BottomSketch, compute_mash_distance, mash_sketch
from diverse_seq.record import KmerSeq, _get_canonical_states
from diverse_seq.records import records_from_seq_store


@define_app
class dvs_ctree:
    """return the N most divergent sequences"""

    def __init__(
        self,
        *,
        seq_store: str | pathlib.Path,
        k: int = 16,
        sketch_size: int | None = None,
        moltype: str = "dna",
        distance_mode: Literal["mash", "euclidean"] = "mash",
        mash_canonical_kmers: bool | None = None,
        with_progress: bool = True,
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
        distance_mode : Literal[&quot;mash&quot;, &quot;euclidean&quot;], optional
            mash distance or euclidean distance between kmer freqs, by default "mash"
        mash_canonical_kmers : bool | None, optional
            whether to use mash canonical kmers for mash distance, by default False
        with_progress : bool, optional
            whether to show progress bars, by default True
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
        if mash_canonical_kmers is None:
            mash_canonical_kmers = False

        if moltype not in ("dna", "rna") and mash_canonical_kmers:
            msg = "Canonical kmers only supported for dna sequences."
            raise ValueError(msg)

        if distance_mode == "mash" and sketch_size is None:
            msg = "Expected sketch size for mash distance measure."
            raise ValueError(msg)

        if distance_mode != "mash" and sketch_size is not None:
            msg = "Sketch size should only be specified for the mash distance."
            raise ValueError(msg)
        if numprocs < 1:
            msg = "Expect numprocs>=1."
            raise ValueError(msg)

        self._seq_store = seq_store
        self._moltype = moltype
        self._k = k
        self._num_states = len(_get_canonical_states(self._moltype))
        self._sketch_size = sketch_size
        self._distance_mode = distance_mode
        self._mash_canonical = mash_canonical_kmers
        self._clustering = AgglomerativeClustering(
            metric="precomputed",
            linkage="average",
        )
        self._progress = Progress(disable=not with_progress)
        self._numprocs = numprocs

    def main(self, seq_names: list[str]) -> PhyloNode:
        """Construct a cluster tree for a collection of sequences.

        Parameters
        ----------
        seq_names : list[str]
            sequence names to cluster.

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
        with self._progress, self._executor:
            if self._distance_mode == "mash":
                distances = self.mash_distances(seq_names)
            elif self._distance_mode == "euclidean":
                distances = self.euclidean_distances(seq_names)
            else:
                msg = f"Unexpected distance {self._distance_mode}."
                raise ValueError(msg)
            return self.make_cluster_tree(seq_names, distances)

    def make_cluster_tree(
        self,
        seq_names: Sequence[str],
        pairwise_distances: numpy.ndarray,
    ) -> PhyloNode:
        """Given pairwise distances between sequences, construct a cluster tree.

        Parameters
        ----------
        seq_names : Sequence[str]
            Names of sequences to cluster.
        pairwise_distances : numpy.ndarray
            Pairwise distances between clusters.

        Returns
        -------
        PhyloNode
            The cluster tree.
        """
        tree_task = self._progress.add_task(
            "[green]Computing Tree",
            total=None,
        )

        self._clustering.fit(pairwise_distances)

        tree_dict = {i: seq_names[i] for i in range(len(seq_names))}
        node_index = len(seq_names)
        for left_index, right_index in self._clustering.children_:
            tree_dict[node_index] = (
                tree_dict.pop(left_index),
                tree_dict.pop(right_index),
            )
            node_index += 1

        tree = make_tree(str(tree_dict[node_index - 1]))

        self._progress.update(tree_task, completed=1, total=1)

        return tree

    def euclidean_distances(self, seq_names: list[str]) -> numpy.ndarray:
        """Calculates pairwise euclidean distances between sequences.

        Parameters
        ----------
        seq_names : list[str]
            Names for pairwise distance calculation.

        Returns
        -------
        numpy.ndarray
            Pairwise euclidean distances between sequences.
        """
        records = records_from_seq_store(
            seq_store=self._seq_store,
            seq_names=seq_names,
            limit=None,
            k=self._k,
        )
        return compute_euclidean_distances(records)

    def mash_distances(self, seq_names: list[str]) -> numpy.ndarray:
        """Calculates pairwise mash distances between sequences.

        Parameters
        ----------
        seq_names : list[str]
            Names for pairwise distance calculation.

        Returns
        -------
        numpy.ndarray
            Pairwise mash distances between sequences.
        """
        dstore = HDF5DataStore(self._seq_store)
        records = get_ordered_records(dstore, seq_names)
        return self.compute_mash_distances(records)

    def compute_mash_distances(self, records: list[DataMember]) -> numpy.ndarray:
        """Calculates pairwise mash distances between sequences.

        Parameters
        ----------
        records : list[DataMember]
            Sequence records.

        Returns
        -------
        numpy.ndarray
            Pairwise mash distances between sequences.
        """
        sketches = self.mash_sketches(records)

        distance_task = self._progress.add_task(
            "[green]Computing Pairwise Distances",
            total=None,
        )

        distances = numpy.zeros((len(sketches), len(sketches)))

        # Compute distances in serial mode
        if self._numprocs == 1:
            self._progress.update(
                distance_task,
                total=len(sketches) * (len(sketches) - 1) // 2,
            )

            for i in range(1, len(sketches)):
                for j in range(i):
                    distance = compute_mash_distance(
                        sketches[i],
                        sketches[j],
                        self._k,
                        self._sketch_size,
                    )
                    distances[i, j] = distance
                    distances[j, i] = distance

                    self._progress.update(distance_task, advance=1)

            return distances

        # Compute distances in parallel
        num_jobs = self._numprocs
        futures = [
            self._executor.submit(
                compute_chunk_distances,
                start,
                num_jobs,
                sketches,
                self._k,
                self._sketch_size,
            )
            for start in range(num_jobs)
        ]

        for future in futures:
            start_idx, distances_chunk = future.result()
            distances[start_idx::num_jobs] = distances_chunk[
                start_idx::num_jobs
            ].toarray()

        # Make lower triangular matrix symmetric
        distances = distances + distances.T - numpy.diag(distances.diagonal())

        self._progress.update(distance_task, completed=1, total=1)

        return distances

    def mash_sketches(self, records: list[DataMember]) -> list[BottomSketch]:
        """Create sketch representations for a collection of sequence records.

        Parameters
        ----------
        records : list[DataMember]
            Sequence records.

        Returns
        -------
        list[BottomSketch]
            Sketches for each sequence.
        """
        sketch_task = self._progress.add_task(
            "[green]Generating Sketches",
            total=len(records),
        )

        bottom_sketches = [None for _ in range(len(records))]

        # Compute sketches in serial
        if self._numprocs == 1:
            for i, record in enumerate(records):
                bottom_sketches[i] = mash_sketch(
                    record.read(),
                    self._k,
                    self._sketch_size,
                    self._num_states,
                    mash_canonical=self._mash_canonical,
                )

                self._progress.update(sketch_task, advance=1)

            return bottom_sketches

        # Compute sketches in parallel
        futures_to_idx = {
            self._executor.submit(
                mash_sketch,
                record.read(),
                self._k,
                self._sketch_size,
                self._num_states,
                mash_canonical=self._mash_canonical,
            ): i
            for i, record in enumerate(records)
        }

        for future in as_completed(list(futures_to_idx)):
            idx = futures_to_idx[future]
            bottom_sketches[idx] = future.result()

            self._progress.update(sketch_task, advance=1)

        return bottom_sketches


def compute_chunk_distances(
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
            distance = compute_mash_distance(sketches[i], sketches[j], k, sketch_size)
            distances_chunk[i, j] = distance
    return start_idx, distances_chunk


def compute_euclidean_distances(records: list[KmerSeq]) -> numpy.ndarray:
    """Compute pairwise euclidean distances between kmer frequencies of sequences.

    Parameters
    ----------
    records : list[KmerSeq]
        kmer sequences to calculate distances between.

    Returns
    -------
    numpy.ndarray
        Pairwise euclidean distances.
    """
    distances = numpy.zeros((len(records), len(records)))

    for i, record_i in enumerate(records):
        freq_i = numpy.array(record_i.kfreqs)
        for j in range(i + 1, len(records)):
            freq_j = numpy.array(records[j].kfreqs)
            distance = numpy.sqrt(((freq_i - freq_j) ** 2).sum())
            distances[i, j] = distance
            distances[j, i] = distance

    return distances
