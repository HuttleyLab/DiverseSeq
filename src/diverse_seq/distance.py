import heapq
import math
from collections.abc import Sequence
from typing import Literal, TypeAlias

import cogent3.app.typing as c3_types
import numba
import numpy as np
from cogent3.app.composable import define_app
from cogent3.evolve.fast_distance import DistanceMatrix
from rich.progress import Progress

from diverse_seq.record import (
    SeqArray,
    _get_canonical_states,
    make_kmerseq,
    seq_to_seqarray,
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

        self._s2a = seq_to_seqarray(moltype=moltype)
        if distance_mode == "mash":
            self._func = mash_distances
            kwargs = dict(
                k=self._k,
                sketch_size=sketch_size,
                num_states=len(_get_canonical_states(self._moltype)),
                mash_canonical=mash_canonical_kmers,
            )
        else:
            self._func = euclidean_distances
            kwargs = dict(k=self._k, moltype=self._moltype)
        self._func_kwargs = kwargs

    def main(
        self,
        seqs: c3_types.SeqsCollectionType,
    ) -> c3_types.PairwiseDistanceType:
        seqs = seqs.to_moltype(self._moltype)
        seq_arrays = [self._s2a(seqs.get_seq(name)) for name in seqs.names]  # pylint: disable=not-callable

        with Progress(disable=not self._show_progress) as progress:
            distances = self._func(
                seq_arrays,
                progress=progress,
                **self._func_kwargs,
            )

        return dists_to_distmatrix(distances, seqs.names)


def dists_to_distmatrix(
    distances: np.ndarray,
    names: Sequence[str],
) -> c3_types.PairwiseDistanceType:
    dist_dict = {}
    for i in range(1, len(distances)):
        for j in range(i):
            dist_dict[(names[i], names[j])] = distances[i, j]
    return DistanceMatrix(dist_dict)


def mash_distances(
    seq_arrays: list[SeqArray],
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
        Sequence arrays.
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

    seqs = [seq_array.data for seq_array in seq_arrays]

    sketches = mash_sketches(
        seqs,
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

    return distances


def mash_sketches(
    seq_arrays: Sequence[np.ndarray],
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
        bottom_sketches[i] = mash_sketch(
            seq_array,
            k,
            sketch_size,
            num_states,
            mash_canonical,
        )

        progress.update(sketch_task, advance=1)

    return bottom_sketches


@numba.njit
def mash_sketch(
    seq_array: np.ndarray,
    k: int,
    sketch_size: int,
    num_states: int,
    mash_canonical: bool,
) -> BottomSketch:  # pragma: no cover
    """Find the mash sketch for a sequence array.

    Parameters
    ----------
    seq_array
        The sequence array to find the sketch for.
    k
        kmer size.
    sketch_size
        Size of the sketch.
    num_states
        Number of possible states (e.g. GCAT gives 4 for DNA).
    mash_canonical
        Whether to use the mash canonical representation of kmers.

    Returns
    -------
    BottomSketch
        The bottom sketch for the given sequence seq_array.
    """
    kmer_hashes = np.unique(get_kmer_hashes(seq_array, k, num_states, mash_canonical))

    heap = [0]
    heap.pop()

    for kmer_hash in kmer_hashes:
        if len(heap) < sketch_size:
            heapq.heappush(heap, -kmer_hash)
        else:
            heapq.heappushpop(heap, -kmer_hash)
    kmer_hashes = [-kmer_hash for kmer_hash in heap]
    kmer_hashes.sort()
    return kmer_hashes


@numba.njit
def get_kmer_hashes(
    seq: np.ndarray,
    k: int,
    num_states: int,
    mash_canonical: bool,
) -> np.ndarray:  # pragma: no cover
    """Get the kmer hashes comprising a sequence.

    Parameters
    ----------
    seq
        A sequence.
    k
        kmer size.
    num_states
        Number of states allowed for sequence type.
    mash_canonical
        Whether to use the mash canonical representation of kmers.

    Returns
    -------
    set[int]
        kmer hashes for the sequence.
    """
    seq = seq.astype(np.int64)

    kmer_hashes = np.zeros(len(seq) - k + 1, dtype=np.int32)

    skip_until = 0
    for i in range(k):
        if seq[i] >= num_states:
            skip_until = i + 1

    kmer_index = 0
    for i in range(len(seq) - k + 1):
        if seq[i + k - 1] >= num_states:
            skip_until = i + k

        if i < skip_until:
            continue
        kmer_hashes[kmer_index] = hash_kmer(seq[i : i + k], mash_canonical)
        kmer_index += 1
    return kmer_hashes[:kmer_index]


@numba.njit
def murmurhash3_32(data: np.ndarray, seed: int = 0x9747B28C) -> int:  # pragma: no cover
    """MurmurHash3 32-bit implementation for an array of integers.

    Parameters
    ----------
    data : np.ndarray
        The input array to hash.
    seed : int
        A seed for the hash function.

    Returns
    -------
    int
        The computed hash value.
    """
    length = data.size
    h = seed ^ length

    for i in range(length):
        k = data[i]

        # Mix the hash
        k *= 0xCC9E2D51
        k = (k << 15) | (k >> (32 - 15))  # Rotate left
        k *= 0x1B873593

        h ^= k
        h = (h << 13) | (h >> (32 - 13))  # Rotate left
        h = h * 5 + 0xE6546B64

    h ^= h >> 16
    h *= 0x85EBCA6B
    h ^= h >> 13
    h *= 0xC2B2AE35
    h ^= h >> 16

    return h & 0xFFFFFFFF  # Return as 32-bit integer


@numba.njit
def hash_kmer(kmer: np.ndarray, mash_canonical: bool) -> int:  # pragma: no cover
    """Hash a kmer, optionally use the mash canonical representaiton.

    Parameters
    ----------
    kmer
        The kmer to hash.
    canonical
        Whether to use the mash canonical representation for a kmer.

    Returns
    -------
    int
        The has of a kmer.

    Notes
    -----
    Uses MurmurHash3 32-bit implementation.
    """
    smallest = kmer
    if mash_canonical:
        reverse = reverse_complement(kmer)
        for i in range(kmer.size):
            if kmer[i] < reverse[i]:
                break
            if kmer[i] > reverse[i]:
                smallest = reverse
                break

    return murmurhash3_32(smallest)


@numba.njit
def reverse_complement(kmer: np.ndarray) -> np.ndarray:  # pragma: no cover
    """Take the reverse complement of a kmer.

    Assumes cogent3 DNA/RNA sequences (numerical
    representation for complement offset by 2
    from original).

    Parameters
    ----------
    kmer
        The kmer to attain the reverse complement of

    Returns
    -------
    numpy.ndarray
        The reverse complement of a kmer.
    """
    # 0123 TCAG
    # 3->1, 1->3, 2->0, 0->2
    return ((kmer + 2) % 4)[::-1]


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
    seq_arrays: Sequence[SeqArray],
    k: int,
    moltype: str,
    *,
    progress: Progress | None = None,
) -> np.ndarray:
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

    kmer_seqs = [
        make_kmerseq(
            seq,
            dtype=np.min_scalar_type(len(_get_canonical_states(moltype)) ** k),
            k=k,
            moltype=moltype,
        )
        for seq in seq_arrays
    ]

    distances = np.zeros((len(kmer_seqs), len(kmer_seqs)))

    distance_task = progress.add_task(
        "[green]Computing Pairwise Distances",
        total=len(kmer_seqs) * (len(kmer_seqs) - 1) // 2,
    )

    for i, kmer_seq_i in enumerate(kmer_seqs):
        freq_i = np.array(kmer_seq_i.kfreqs)
        for j in range(i + 1, len(kmer_seqs)):
            freq_j = np.array(kmer_seqs[j].kfreqs)

            distance = euclidean_distance(freq_i, freq_j)
            distances[i, j] = distance
            distances[j, i] = distance

            progress.update(distance_task, advance=1)

    return distances


def euclidean_distance(freq_1: np.ndarray, freq_2: np.ndarray) -> np.ndarray:
    return np.linalg.norm(freq_1 - freq_2)
