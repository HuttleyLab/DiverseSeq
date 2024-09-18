from collections.abc import Sequence
from typing import Literal

import cogent3.app.typing as c3_types
import numpy as np
from cogent3.app.composable import define_app
from rich.progress import Progress

from diverse_seq.record import (
    KmerSeq,
    _get_canonical_states,
    make_kmerseq,
    seq_to_seqarray,
)


@define_app
class dvs_dist:
    """return the N most divergent sequences"""

    def __init__(
        self,
        distance_mode: Literal["mash", "euclidean"] = "mash",
        *,
        k: int = 16,
        sketch_size: int | None = None,
        moltype: str = "dna",
        mash_canonical_kmers: bool | None = None,
        with_progress: bool = True,
    ) -> None:
        """Initialise parameters for kmer distance calculation.

        Parameters
        ----------
        distance_mode : Literal[&quot;mash&quot;, &quot;euclidean&quot;], optional
            mash distance or euclidean distance between kmer freqs, by default "mash"
        k : int, optional
            kmer size, by default 16
        sketch_size : int | None, optional
            size of sketches, by default None
        moltype : str, optional
            moltype, by default "dna"
        mash_canonical_kmers : bool | None, optional
            whether to use mash canonical kmers for mash distance, by default False
        with_progress : bool, optional
            whether to show progress bars, by default True

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

        self._moltype = moltype
        self._k = k
        self._num_states = len(_get_canonical_states(self._moltype))
        self._sketch_size = sketch_size
        self._distance_mode = distance_mode
        self._mash_canonical = mash_canonical_kmers
        self._with_progress = with_progress

        self._s2a = seq_to_seqarray(moltype=moltype)
        self._dtype = np.min_scalar_type(self._num_states**self._k)

    def main(
        self,
        seqs: c3_types.SeqsCollectionType,
    ) -> c3_types.PairwiseDistanceType:
        seq_arrays = [self._s2a(seqs.get_seq(name)) for name in seqs.names]

        if self._distance_mode == "mash":
            distances = self.mash_distances(seq_arrays)
        elif self._distance_mode == "euclidean":
            kmer_seqs = [
                make_kmerseq(
                    seq,
                    dtype=self._dtype,
                    k=self._k,
                    moltype=self._moltype,
                )
                for seq in seq_arrays
            ]
            distances = euclidean_distances(
                kmer_seqs,
                with_progress=self._with_progress,
            )
        else:
            msg = f"Unexpected distance {self._distance_mode}."
            raise ValueError(msg)
        return distances


def euclidean_distances(
    kmer_seqs: Sequence[KmerSeq],
    *,
    with_progress: bool = False,
) -> np.ndarray:
    """Calculates pairwise euclidean distances between sequences.

    Parameters
    ----------
    seqs : Sequence[SeqArray]
        Sequences for pairwise distance calculation.
    with_progress : bool, optional
        Whether to show progress bar, by default False

    Returns
    -------
    np.ndarray
        Pairwise euclidean distances between sequences.
    """

    distances = np.zeros((len(kmer_seqs), len(kmer_seqs)))

    with Progress(disable=not with_progress) as progress:
        distance_task = progress.add_task(
            "[green]Computing Pairwise Distances",
            total=len(kmer_seqs) * (len(kmer_seqs) - 1) // 2,
        )

        for i, kmer_seq_i in enumerate(kmer_seqs):
            freq_i = np.array(kmer_seq_i.kfreqs)
            for j in range(i + 1, len(kmer_seqs)):
                freq_j = np.array(kmer_seqs[j].kfreqs)

                distance = np.sqrt(((freq_i - freq_j) ** 2).sum())
                distances[i, j] = distance
                distances[j, i] = distance

                progress.update(distance_task, advance=1)

    return distances
