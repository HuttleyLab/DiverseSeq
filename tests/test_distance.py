import pytest
from cogent3 import load_unaligned_seqs
from numpy.testing import assert_array_equal

from diverse_seq.distance import dvs_dist


@pytest.fixture(scope="session")
def seq_path(DATA_DIR):
    return DATA_DIR / "brca1.fasta"


def test_euclidean_distance(seq_path):
    app = dvs_dist(
        "euclidean",
        k=3,
        with_progress=True,
    )

    seqs = load_unaligned_seqs(seq_path)
    seqs = seqs.take_seqs(["Human", "Chimpanzee", "Manatee", "Dugong", "Rhesus"])
    dists = app(seqs)

    assert_array_equal(dists, dists.T)  # Symmetric

    human = dists[0]

    assert human[1] < human[4]  # chimpanzee closer than rhesus
    assert human[4] < human[2]  # rhesus closer than manatee
    assert human[4] < human[3]  # rhesus closer than dugong

    chimpanzee = dists[1]
    assert chimpanzee[4] < chimpanzee[2]  # rhesus closer than manatee
    assert chimpanzee[4] < chimpanzee[3]  # rhesus closer than dugong

    manatee = dists[2]
    assert manatee[3] < manatee[4]  # dugong closer than rhesus
