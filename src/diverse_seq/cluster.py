import pathlib

import numpy as np
from cogent3 import PhyloNode, make_tree
from cogent3.app.composable import define_app
from sklearn.cluster import AgglomerativeClustering

from diverse_seq.record import KmerSeq
from diverse_seq.records import records_from_seq_store


@define_app
class dvs_cluster:
    """return the N most divergent sequences"""

    def __init__(self, *, seq_store: str | pathlib.Path, k: int = 3) -> None:
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
        print("Initialising")
        self._seq_store = seq_store
        self._k = k
        self._clustering = AgglomerativeClustering(
            metric="precomputed",
            linkage="average",
        )

    def main(self, seq_names: list[str]) -> PhyloNode:
        records = records_from_seq_store(
            seq_store=self._seq_store,
            seq_names=seq_names,
            limit=None,
            k=self._k,
        )
        distances = compute_distances(records)
        self._clustering.fit(distances)

        tree_dict = {i: seq_names[i] for i in range(len(seq_names))}
        node_index = len(seq_names)
        for left_index, right_index in self._clustering.children_:
            tree_dict[node_index] = (
                tree_dict.pop(left_index),
                tree_dict.pop(right_index),
            )
            node_index += 1
        return make_tree(str(tree_dict[node_index - 1]))


def compute_distances(records: list[KmerSeq]) -> np.array:
    distances = np.zeros((len(records), len(records)))

    for i, record_i in enumerate(records):
        freq_i = np.array(record_i.kfreqs)
        for j in range(i + 1, len(records)):
            freq_j = np.array(records[j].kfreqs)
            distance = np.sqrt(((freq_i - freq_j) ** 2).sum())
            distances[i, j] = distance
            distances[j, i] = distance

    return distances
