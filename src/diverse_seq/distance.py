from functools import singledispatch

from numpy import intersect1d, ndarray, union1d


@singledispatch
def _intersect_union(rec1, rec2) -> tuple[int, int]:
    raise NotImplementedError


@_intersect_union.register(set)
def _(rec1: set, rec2: set) -> tuple[int, int]:
    intersect = rec1 & rec2
    union = rec1 | rec2
    return len(intersect), len(union)


@_intersect_union.register(ndarray)
def _(rec1: ndarray, rec2: ndarray) -> tuple[int, int]:
    intersect = intersect1d(rec1, rec2, assume_unique=True)
    union = union1d(rec1.data, rec2.data)
    return len(intersect), len(union)


def jaccard(rec1, rec2) -> float:
    """returns the Jaccard distance between rec1 and rec2"""
    i, u = _intersect_union(rec1, rec2)
    return 1 - (i / u)
