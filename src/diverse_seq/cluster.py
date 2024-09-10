import heapq
import math
import pathlib
from collections.abc import Generator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from typing import Literal, TypeAlias

import numpy as np
from cogent3 import PhyloNode, make_tree
from cogent3.app.composable import define_app
from cogent3.app.data_store import DataMember
from rich.progress import Progress
from scipy.sparse import dok_matrix
from sklearn.cluster import AgglomerativeClustering

from diverse_seq.data_store import HDF5DataStore
from diverse_seq.record import KmerSeq, _get_canonical_states
from diverse_seq.records import records_from_seq_store

BottomSketch: TypeAlias = list[int]


@define_app
class dvs_cluster_tree:
    """return the N most divergent sequences"""

    def __init__(
        self,
        *,
        seq_store: str | pathlib.Path,
        k: int = 16,
        sketch_size: int | None = None,
        moltype: str = "dna",
        distance_mode: Literal["mash", "euclidean"] = "mash",
        canonical_kmers: bool | None = None,
        with_progress: bool = True,
        numprocs: int = 1,
    ) -> None:
        if canonical_kmers is None:
            canonical_kmers = False

        if moltype not in ("dna", "rna") and canonical_kmers:
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
        self._canonical = canonical_kmers
        self._clustering = AgglomerativeClustering(
            metric="precomputed",
            linkage="average",
        )
        self._with_progress = with_progress
        self._progress = Progress() if with_progress else nullcontext()
        self._executor = ProcessPoolExecutor(max_workers=numprocs)
        self._numprocs = numprocs

    def main(self, seq_names: list[str]) -> PhyloNode:
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
        pairwise_distances: np.ndarray,
    ) -> PhyloNode:
        if self._with_progress:
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

        if self._with_progress:
            self._progress.update(tree_task, completed=1, total=1)

        return tree

    def euclidean_distances(self, seq_names: list[str]) -> np.ndarray:
        records = records_from_seq_store(
            seq_store=self._seq_store,
            seq_names=seq_names,
            limit=None,
            k=self._k,
        )
        return compute_euclidean_distances(records)

    def mash_distances(self, seq_names: list[str]) -> np.ndarray:
        dstore = HDF5DataStore(self._seq_store)
        records = [m for m in dstore.completed if m.unique_id in seq_names]
        return self.compute_mash_distances(records)

    def compute_mash_distances(self, records: list[DataMember]) -> np.ndarray:
        sketches = self.mash_sketches(records)

        if self._with_progress:
            distance_task = self._progress.add_task(
                "[green]Computing Pairwise Distances",
                total=None,
            )

        distances = np.zeros((len(sketches), len(sketches)))

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
        distances = distances + distances.T - np.diag(distances.diagonal())

        if self._with_progress:
            self._progress.update(distance_task, completed=1, total=1)

        return distances

    def mash_sketches(self, records: list[DataMember]) -> list[BottomSketch]:
        if self._with_progress:
            sketch_task = self._progress.add_task(
                "[green]Generating Sketches",
                total=len(records),
            )

        bottom_sketches = [None for _ in range(len(records))]
        futures_to_idx = {
            self._executor.submit(
                mash_sketch,
                record,
                self._k,
                self._sketch_size,
                self._num_states,
                canonical=self._canonical,
            ): i
            for i, record in enumerate(records)
        }

        for future in as_completed(list(futures_to_idx)):
            idx = futures_to_idx[future]
            bottom_sketches[idx] = future.result()

            if self._with_progress:
                self._progress.update(sketch_task, advance=1)

        return bottom_sketches


def compute_chunk_distances(
    start_idx: int,
    stride: int,
    sketches: list[BottomSketch],
    k: int,
    sketch_size: int,
) -> np.ndarray:
    distances_chunk = dok_matrix((len(sketches), len(sketches)))
    for i in range(start_idx, len(sketches), stride):
        for j in range(i):
            distance = compute_mash_distance(sketches[i], sketches[j], k, sketch_size)
            distances_chunk[i, j] = distance
    return start_idx, distances_chunk


def mash_sketch(
    record: DataMember,
    k: int,
    sketch_size: int,
    num_states: int,
    *,
    canonical: bool,
) -> BottomSketch:
    seq = record.read()
    kmer_hashes = {
        hash_kmer(kmer, canonical=canonical) for kmer in iter_kmers(seq, k, num_states)
    }
    heap = []
    for kmer_hash in kmer_hashes:
        if len(heap) < sketch_size:
            heapq.heappush(heap, -kmer_hash)
        else:
            heapq.heappushpop(heap, -kmer_hash)
    return sorted(-kmer_hash for kmer_hash in heap)


def compute_mash_distance(
    left_sketch: BottomSketch,
    right_sketch: BottomSketch,
    k: int,
    sketch_size: int,
) -> float:
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


def iter_kmers(
    seq: np.ndarray,
    k: int,
    num_states: int,
) -> Generator[np.ndarray, None, None]:
    skip_until = 0
    for i in range(k):
        if seq[i] >= num_states:
            skip_until = i + 1

    for i in range(len(seq) - k + 1):
        if seq[i + k - 1] >= num_states:
            skip_until = i + k

        if i < skip_until:
            continue
        yield seq[i : i + k]


def hash_kmer(kmer: np.ndarray, *, canonical: bool) -> int:
    tuple_kmer = tuple(map(int, kmer))
    if canonical:
        reverse = map(int, reverse_complement(kmer))
        tuple_kmer = min(reverse, tuple_kmer)

    return hash(tuple_kmer)


def reverse_complement(kmer: np.ndarray) -> np.ndarray:
    # 0123 TCAG
    # 3->1, 1->3, 2->0, 0->2
    return ((kmer + 2) % 4)[::-1]


def compute_euclidean_distances(records: list[KmerSeq]) -> np.ndarray:
    distances = np.zeros((len(records), len(records)))

    for i, record_i in enumerate(records):
        freq_i = np.array(record_i.kfreqs)
        for j in range(i + 1, len(records)):
            freq_j = np.array(records[j].kfreqs)
            distance = np.sqrt(((freq_i - freq_j) ** 2).sum())
            distances[i, j] = distance
            distances[j, i] = distance

    return distances
