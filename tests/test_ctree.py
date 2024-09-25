# pylint: disable=not-callable
import pytest
from cogent3 import SequenceCollection, make_tree

from diverse_seq.cluster import dvs_ctree, dvs_par_ctree


def check_ctree_app(app: dvs_ctree | dvs_par_ctree, seqs: SequenceCollection) -> None:
    tree = app(seqs.take_seqs(["Human", "Chimpanzee", "Rhesus", "Horse"]))
    expected = make_tree("(((Human, Chimpanzee), Rhesus), Horse);")
    assert tree.same_topology(expected)

    tree = app(seqs.take_seqs(["Human", "Chimpanzee", "Manatee", "Dugong"]))
    expected = make_tree("((Human, Chimpanzee), (Manatee, Dugong));")
    assert tree.same_topology(expected)

    tree = app(seqs.take_seqs(["Human", "Chimpanzee", "Manatee", "Dugong", "Rhesus"]))
    expected = make_tree("(((Human, Chimpanzee), Rhesus), (Manatee, Dugong));")
    assert tree.same_topology(expected)


@pytest.mark.parametrize("sketch_size", [400, 4e50])
def test_ctree_mash(unaligned_seqs: SequenceCollection, sketch_size: int) -> None:
    app = dvs_ctree(k=16, sketch_size=sketch_size, distance_mode="mash")
    check_ctree_app(app, unaligned_seqs)


def test_ctree_euclidean(unaligned_seqs: SequenceCollection) -> None:
    app = dvs_ctree(k=5, distance_mode="euclidean")
    check_ctree_app(app, unaligned_seqs)


@pytest.mark.parametrize("max_workers", [1, 4])
def test_ctree_mash_parallel(
    unaligned_seqs: SequenceCollection,
    max_workers: int,
) -> None:
    app = dvs_par_ctree(
        k=16,
        sketch_size=400,
        max_workers=max_workers,
        parallel=True,
        distance_mode="mash",
    )
    check_ctree_app(app, unaligned_seqs)


@pytest.mark.parametrize("max_workers", [1, 4])
def test_ctree_euclidean_parallel(
    unaligned_seqs: SequenceCollection,
    max_workers: int,
) -> None:
    app = dvs_par_ctree(
        k=5,
        max_workers=max_workers,
        parallel=True,
        distance_mode="euclidean",
    )
    check_ctree_app(app, unaligned_seqs)
