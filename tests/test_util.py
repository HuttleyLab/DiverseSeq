import pytest

from cogent3 import get_moltype

from divergent.util import str2arr


def test_str2arr():
    dna = get_moltype("dna")
    s = "ACGTT"
    expect = dna.alphabet.to_indices(s)
    app = str2arr()
    g = app(s)
    assert (g == expect).all()
    g = app("ACGNT")
    assert g[-2] > 3  # index for non-canonical character > num_states
