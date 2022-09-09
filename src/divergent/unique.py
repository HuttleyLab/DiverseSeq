import numpy

from cogent3.app import composable
from cogent3.app import typing as c3_types

from divergent import util as dv_utils
from divergent.record import unique_kmers


@composable.define_app
class signature_kmers:
    def __init__(self, paths, verbose=False):
        self.paths = paths
        self.loader = (
            dv_utils.load_bytes()
            + dv_utils.blosc_decompress()
            + dv_utils.unpickle_data()
        )
        self.verbose = verbose

    def main(self, path: c3_types.IdentifierType) -> unique_kmers:
        query = self.loader(path)
        setdiff = numpy.setdiff1d

        if self.verbose:
            print(f"initial number k-mers = {len(query):,}")

        for p in self.paths:
            if str(p) == str(path):
                continue
            o = self.loader(p)
            query.data = setdiff(query.data, o.data, assume_unique=True)

        if self.verbose:
            print(f"final number k-mers = {len(query):,}")

        return query
