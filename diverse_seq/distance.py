import math
from collections.abc import Sequence
from typing import Literal, TypeAlias

import cogent3.app.typing as c3_types
import numpy as np
from cogent3.app.composable import define_app
from cogent3.evolve.fast_distance import DistanceMatrix
from rich.progress import Progress

from diverse_seq import _dvs as dvs
from diverse_seq.util import (
    _get_canonical_states,
    populate_inmem_zstore,
)

BottomSketch: TypeAlias = list[int]


@define_app
class dvs_dist:
    """Calculate pairwise kmer-based distances between sequences.
    Supported distances include mash distance, and euclidean distance
    based on kmer frequencies.
    """

    def __init__(
        self,
        distance_mode: Literal["mash", "euclidean"] = "mash",
        *,
        k: int = 12,
        sketch_size: int | None = 3000,
        moltype: str = "dna",
        mash_canonical_kmers: bool | None = None,
        show_progress: bool = False,
    ) -> None:
        """Initialise parameters for kmer distance calculation.

        Parameters
        ----------
        distance_mode
            mash distance or euclidean distance between kmer freqs
        k
            kmer size
        sketch_size
            size of sketches, by default None
        moltype
            moltype
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
            msg = "Canonical kmers only supported for dna sequences."
            raise ValueError(msg)

        if distance_mode == "mash" and sketch_size is None:
            msg = "Expected sketch size for mash distance measure."
            raise ValueError(msg)

        self._moltype = moltype
        self._k = k
        self._show_progress = show_progress
        self._num_states = len(_get_canonical_states(self._moltype))

        if distance_mode == "mash":
            self._func = mash_distances
            kwargs = {
                "k": self._k,
                "sketch_size": sketch_size,
                "num_states": self._num_states,
                "mash_canonical": mash_canonical_kmers,
            }
        else:
            self._func = euclidean_distances
            kwargs = {"k": self._k, "moltype": self._moltype}
        self._func_kwargs = kwargs

    def main(
        self,
        seqs: c3_types.SeqsCollectionType,
    ) -> c3_types.PairwiseDistanceType:
        seqs = seqs.to_moltype(self._moltype).degap()
        zstore = populate_inmem_zstore(seqs)

        seq_arrays = zstore.get_lazyseqs(num_states=self._num_states)

        with Progress(disable=not self._show_progress) as progress:
            distances = self._func(
                seq_arrays,
                progress=progress,
                **self._func_kwargs,
            )

        return distances


def mash_distances(
    seq_arrays: list[dvs.LazySeq],
    k: int,
    sketch_size: int,
    num_states: int,
    *,
    mash_canonical: bool = False,
    progress: Progress | None = None,
) -> np.ndarray:
    """Calculates pairwise mash distances between sequences.

    Parameters
    ----------
    seq_arrays
        lazy sequence objects.
    k
        kmer size.
    sketch_size
        sketch size.
    num_states
        number of states for each position.
    mash_canonical
        whether to use mash canonical representation of kmers,
        by default False
    progress
        progress bar, by default None

    Returns
    -------
    numpy.ndarray
        Pairwise mash distances between sequences.
    """
    if progress is None:
        progress = Progress(disable=True)

    sketches = mash_sketches(
        seq_arrays,
        k,
        sketch_size,
        num_states,
        mash_canonical=mash_canonical,
        progress=progress,
    )

    distance_task = progress.add_task(
        "[green]Computing Pairwise Distances",
        total=len(sketches) * (len(sketches) - 1) // 2,
    )

    seqids = [sarr.seqid for sarr in seq_arrays]
    distances = np.zeros((len(sketches), len(sketches)))

    for i in range(1, len(sketches)):
        for j in range(i):
            distance = mash_distance(
                sketches[i],
                sketches[j],
                k,
                sketch_size,
            )
            distances[i, j] = distance
            distances[j, i] = distance

            progress.update(distance_task, advance=1)

    return DistanceMatrix.from_array_names(matrix=distances, names=seqids)


def mash_sketches(
    seq_arrays: Sequence[dvs.LazySeq],
    k: int,
    sketch_size: int,
    num_states: int,
    *,
    mash_canonical: bool = False,
    progress: Progress | None = None,
) -> list[BottomSketch]:
    """Create sketch representations for a collection of sequence sequence arrays.

    Parameters
    ----------
    seq_arrays
        Sequence arrays.
    k
        kmer size.
    sketch_size
        sketch size.
    num_states
        number of states.
    mash_canonical
        whether to use mash canonical kmer representation, by default False
    progress
        progress bar, by default None
    Returns
    -------
    list[BottomSketch]
        Sketches for each sequence.
    """
    if progress is None:
        progress = Progress(disable=True)

    sketch_task = progress.add_task(
        "[green]Generating Sketches",
        total=len(seq_arrays),
    )

    bottom_sketches = [None for _ in range(len(seq_arrays))]

    # Compute sketches in serial
    for i, seq_array in enumerate(seq_arrays):
        bottom_sketches[i] = dvs.mash_sketch(
            seq_array.get_seq(),
            k,
            int(sketch_size),
            num_states,
            mash_canonical,
        )

        progress.update(sketch_task, advance=1)

    return bottom_sketches


def mash_distance(
    left_sketch: BottomSketch,
    right_sketch: BottomSketch,
    k: int,
    sketch_size: int,
) -> float:
    """Compute the mash distance between two sketches.

    Parameters
    ----------
    left_sketch
        A sketch for comparison.
    right_sketch
        A sketch for comparison.
    k
        kmer size.
    sketch_size
        Size of the sketches.

    Returns
    -------
    float
        The mash distance between two sketches.
    """
    # Following the source code implementation
    intersection_size = 0
    union_size = 0

    left_index = 0
    right_index = 0
    while (
        union_size < sketch_size
        and left_index < len(left_sketch)
        and right_index < len(right_sketch)
    ):
        left, right = left_sketch[left_index], right_sketch[right_index]
        if left < right:
            left_index += 1
        elif right < left:
            right_index += 1
        else:
            left_index += 1
            right_index += 1
            intersection_size += 1
        union_size += 1

    if union_size < sketch_size:
        if left_index < len(left_sketch):
            union_size += len(left_sketch) - left_index
        if right_index < len(right_sketch):
            union_size += len(right_sketch) - right_index
        union_size = min(union_size, sketch_size)

    jaccard = intersection_size / union_size
    if intersection_size == union_size:
        return 0.0
    if intersection_size == 0:
        return 1.0
    distance = -math.log(2 * jaccard / (1.0 + jaccard)) / k
    if distance > 1:
        distance = 1.0
    return distance


def euclidean_distances(
    seq_arrays: Sequence[dvs.LazySeq],
    k: int,
    moltype: str,
    *,
    progress: Progress | None = None,
) -> DistanceMatrix:
    """Calculates pairwise euclidean distances between sequences.

    Parameters
    ----------
    seqs
        Sequences for pairwise distance calculation.
    progress
        Progress bar, by default None

    Returns
    -------
    np.ndarray
        Pairwise euclidean distances between sequences.
    """
    if progress is None:
        progress = Progress(disable=True)

    distances = np.zeros((len(seq_arrays), len(seq_arrays)))

    distance_task = progress.add_task(
        "[green]Computing Pairwise Distances",
        total=len(seq_arrays) * (len(seq_arrays) - 1) // 2,
    )

    seqids = [sarr.seqid for sarr in seq_arrays]
    for i, sarr_i in enumerate(seq_arrays[1:], 1):
        freq_i = np.array(sarr_i.get_kfreqs(k))
        for j, sarr_j in enumerate(seq_arrays[:i]):
            freq_j = np.array(sarr_j.get_kfreqs(k))
            distance = euclidean_distance(freq_i, freq_j)
            distances[i, j] = distance
            distances[j, i] = distance

            progress.update(distance_task, advance=1)

    return DistanceMatrix.from_array_names(matrix=distances, names=seqids)


def euclidean_distance(freq_1: np.ndarray, freq_2: np.ndarray) -> np.ndarray:
    return np.linalg.norm(freq_1 - freq_2)
