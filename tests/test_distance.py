# pylint: disable=not-callable

from diverse_seq.distance import dvs_dist


def test_euclidean_distance(unaligned_seqs):
    app = dvs_dist(
        "euclidean",
        k=3,
    )

    unaligned_seqs = unaligned_seqs.take_seqs(
        ["Human", "Chimpanzee", "Manatee", "Dugong", "Rhesus"],
    )
    dists = app(unaligned_seqs)

    assert (
        dists["Human", "Chimpanzee"] < dists["Human", "Dugong"]
    )  # chimpanzee closer than rhesus
    assert (
        dists["Human", "Rhesus"] < dists["Human", "Manatee"]
    )  # rhesus closer than manatee
    assert (
        dists["Human", "Rhesus"] < dists["Human", "Dugong"]
    )  # rhesus closer than dugong

    assert (
        dists["Chimpanzee", "Rhesus"] < dists["Chimpanzee", "Manatee"]
    )  # rhesus closer than manatee
    assert (
        dists["Chimpanzee", "Rhesus"] < dists["Chimpanzee", "Dugong"]
    )  # rhesus closer than dugong

    assert (
        dists["Manatee", "Dugong"] < dists["Manatee", "Rhesus"]
    )  # dugong closer than rhesus


def test_mash_distance(unaligned_seqs):
    app = dvs_dist(
        "mash",
        k=16,
        sketch_size=400,
        mash_canonical_kmers=True,
    )

    unaligned_seqs = unaligned_seqs.take_seqs(
        ["Human", "Chimpanzee", "Manatee", "Dugong", "Rhesus"],
    )
    dists = app(unaligned_seqs)

    assert (
        dists["Human", "Chimpanzee"] < dists["Human", "Dugong"]
    )  # chimpanzee closer than rhesus
    assert (
        dists["Human", "Rhesus"] < dists["Human", "Manatee"]
    )  # rhesus closer than manatee
    assert (
        dists["Human", "Rhesus"] < dists["Human", "Dugong"]
    )  # rhesus closer than dugong

    assert (
        dists["Chimpanzee", "Rhesus"] < dists["Chimpanzee", "Manatee"]
    )  # rhesus closer than manatee
    assert (
        dists["Chimpanzee", "Rhesus"] < dists["Chimpanzee", "Dugong"]
    )  # rhesus closer than dugong

    assert (
        dists["Manatee", "Dugong"] < dists["Manatee", "Rhesus"]
    )  # dugong closer than rhesus
