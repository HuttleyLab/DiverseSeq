import pytest
from numpy import array
from numpy.testing import assert_allclose

from diverse_seq.distance import _intersect_union, jaccard


def _make_sample(type_):
    return type_([0, 1, 3]), type_([1, 4, 5])


@pytest.mark.parametrize("rec1,rec2", (_make_sample(set), _make_sample(array)))
def test_intersect_union(rec1, rec2):
    i, u = _intersect_union(rec1, rec2)
    assert i == 1 and u == 5


@pytest.mark.parametrize(
    "rec1,rec2,exp",
    (
        [{1, 2}, {3, 4}, 1],
        [set(), {1}, 1],
        [{1}, set(), 1],
        [{1, 2}, {1, 2}, 0],
        [{1, 2}, {1, 3}, 2 / 3],
    ),
)
def test_jaccard(rec1, rec2, exp):
    assert_allclose(jaccard(rec1, rec2), exp)
