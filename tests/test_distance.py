from numpy.testing import assert_array_equal

from diverse_seq.distance import dvs_dist


def test_euclidean_distance(unaligned_seqs):
    app = dvs_dist(
        "euclidean",
        k=3,
        with_progress=True,
    )

    unaligned_seqs = unaligned_seqs.take_seqs(
        ["Human", "Chimpanzee", "Manatee", "Dugong", "Rhesus"],
    )
    dists = app(unaligned_seqs)

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


def test_mash_distance(unaligned_seqs):
    app = dvs_dist(
        "mash",
        k=16,
        sketch_size=400,
        mash_canonical_kmers=True,
        with_progress=True,
    )

    unaligned_seqs = unaligned_seqs.take_seqs(
        ["Human", "Chimpanzee", "Manatee", "Dugong", "Rhesus"],
    )
    dists = app(unaligned_seqs)

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
