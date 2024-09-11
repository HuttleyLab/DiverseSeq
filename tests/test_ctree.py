import pytest
from cogent3 import make_tree

from diverse_seq.cluster import dvs_cluster_tree


@pytest.mark.parametrize("sketch_size", [400, 4e50])
def test_ctree_mash(DATA_DIR, sketch_size):
    dvseqs_path = DATA_DIR / "brca1.dvseqs"
    app = dvs_cluster_tree(seq_store=dvseqs_path, k=16, sketch_size=sketch_size)
    tree = app(["Human", "Chimpanzee", "Rhesus", "Horse"])
    expected = make_tree("(((Human, Chimpanzee), Rhesus), Horse);")
    assert tree.same_topology(expected)

    tree = app(["Human", "Chimpanzee", "Manatee", "Dugong"])
    expected = make_tree("((Human, Chimpanzee), (Manatee, Dugong));")
    assert tree.same_topology(expected)

    tree = app(["Human", "Chimpanzee", "Manatee", "Dugong", "Rhesus"])
    expected = make_tree("(((Human, Chimpanzee), Rhesus), (Manatee, Dugong));")
    assert tree.same_topology(expected)


def test_ctree_mash_parallel(DATA_DIR):
    dvseqs_path = DATA_DIR / "brca1.dvseqs"
    app = dvs_cluster_tree(
        seq_store=dvseqs_path,
        k=16,
        sketch_size=400,
        numprocs=4,
    )

    tree = app(["Human", "Chimpanzee", "Manatee", "Dugong", "Rhesus"])
    expected = make_tree("(((Human, Chimpanzee), Rhesus), (Manatee, Dugong));")
    assert tree.same_topology(expected)


def test_ctree_euclidean(DATA_DIR):
    dvseqs_path = DATA_DIR / "brca1.dvseqs"
    app = dvs_cluster_tree(seq_store=dvseqs_path, k=5, distance_mode="euclidean")
    tree = app(["Human", "Chimpanzee", "Rhesus", "Horse"])
    expected = make_tree("(((Human, Chimpanzee), Rhesus), Horse);")
    assert tree.same_topology(expected)

    tree = app(["Human", "Chimpanzee", "Manatee", "Dugong"])
    expected = make_tree("((Human, Chimpanzee), (Manatee, Dugong));")
    assert tree.same_topology(expected)

    tree = app(["Human", "Chimpanzee", "Manatee", "Dugong", "Rhesus"])
    expected = make_tree("(((Human, Chimpanzee), Rhesus), (Manatee, Dugong));")
    assert tree.same_topology(expected)
