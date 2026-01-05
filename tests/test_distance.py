# pylint: disable=not-callable
import cogent3 as c3
import numpy as np

from diverse_seq.distance import dvs_dist


def calc_expected_euclidean(seqs, name_order, k):
    from itertools import combinations

    kcounts = {name: seqs.seqs[name].count_kmers(k=k) for name in name_order}
    kfreqs = {n: c / float(c.sum()) for n, c in kcounts.items()}
    for n, f in kfreqs.items():
        cf = f
        if not np.allclose(f, cf):
            print(f"%% MISMATCH in freqs for {n}")
    mat = np.zeros((len(name_order), len(name_order)), dtype=float)
    for n1, n2 in combinations(name_order, 2):
        f1 = kfreqs[n1]
        f2 = kfreqs[n2]
        d = np.linalg.norm(f1 - f2)
        i = name_order.index(n1)
        j = name_order.index(n2)
        mat[i, j] = d
        mat[j, i] = d

    return c3.evolve.fast_distance.DistanceMatrix.from_array_names(mat, name_order)


def test_euclidean_distance(unaligned_seqs):
    k = 5
    app = dvs_dist(
        "euclidean",
        k=k,
    )
    names = ["Human", "Chimpanzee", "Manatee", "Dugong", "Rhesus"]
    unaligned_seqs = unaligned_seqs.take_seqs(names)
    dists = app(unaligned_seqs).take_dists(names)

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

    expect = calc_expected_euclidean(unaligned_seqs, names, k).take_dists(names)
    assert np.allclose(dists.array, expect.array, atol=1e-3)


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
    expect = c3.make_table(
        data={
            "names": ["Chimpanzee", "Dugong", "Human", "Manatee", "Rhesus"],
            "Chimpanzee": [
                0.0,
                0.1683538414781319,
                0.0088147336093394,
                0.16419257116053393,
                0.04415109484255263,
            ],
            "Dugong": [
                0.1683538414781319,
                0.0,
                0.1683538414781319,
                0.014324116610635772,
                0.14696095357271735,
            ],
            "Human": [
                0.0088147336093394,
                0.1683538414781319,
                0.0,
                0.16030933484134605,
                0.04415109484255263,
            ],
            "Manatee": [
                0.16419257116053393,
                0.014324116610635772,
                0.16030933484134605,
                0.0,
                0.1413009800073594,
            ],
            "Rhesus": [
                0.04415109484255263,
                0.14696095357271735,
                0.04415109484255263,
                0.1413009800073594,
                0.0,
            ],
        },
        index_name="names",
    )

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
