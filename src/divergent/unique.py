import re

from collections import defaultdict
from pathlib import Path

from cogent3 import get_moltype, make_table
from cogent3.app import composable
from cogent3.app import typing as c3_types
from cogent3.app.composable import NotCompleted
from cogent3.util import parallel as PAR
from numpy import setdiff1d, zeros
from rich.progress import track

from divergent import util as dv_utils
from divergent.record import indices_to_seqs, seq_to_unique_kmers, unique_kmers
from src.divergent.distance import jaccard


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
        if self.verbose:
            print(f"initial number k-mers = {len(query):,}")

        for p in self.paths:
            if str(p) == str(path):
                continue
            o = self.loader(p)
            d = setdiff1d(query.data, o.data, assume_unique=True)
            if len(d) == 0:
                return NotCompleted(
                    "FAIL",
                    self,
                    f"redundant with {p.stem}",
                    source=path,
                )
            query.data = d

        if self.verbose:
            print(f"final number k-mers = {len(query):,}")

        return query


def non_redundant(paths, seqdir):
    _name_parts = re.compile("[-.]")
    get_identifier = lambda x: _name_parts.split(x)[0]
    # paths are from the k-mers, but we need the fasta paths
    recid_to_path = {get_identifier(Path(p).stem): p for p in paths}
    seqfns = []
    for fn in seqdir.glob("*.fa*"):
        for identifier in recid_to_path:
            if identifier in fn.stem:
                seqfns.append(fn)
                break

    if not seqfns:
        raise ValueError("no matching sequence file names")

    k = 7
    app = dv_utils.seq_from_fasta() + seq_to_unique_kmers(k=k, moltype="dna")
    series = PAR.as_completed(app, seqfns, max_workers=6)
    records = list(
        track(
            series,
            total=len(seqfns),
            update_period=1,
            description=f"seq to {k}-mers...",
        )
    )

    # identify seqs with equal k-mers
    matches = defaultdict(list)
    matched = set()
    for i in track(
        range(len(records) - 1), total=len(records) - 1, description="calc distances..."
    ):
        rec1 = records[i]
        rec1_id = get_identifier(rec1.name)
        if rec1_id in matched:
            continue
        for j in range(i + 1, len(records)):
            rec2 = records[j]
            rec2_id = get_identifier(rec2.name)
            if jaccard(rec1, rec2) == 0:
                matches[rec1_id].append(rec2_id)
                matched.update({rec1_id, rec2_id})
                continue

    nr_paths = [recid_to_path[recid] for recid in matches]
    redundant_paths = [
        recid_to_path[recid] for recid in (recid_to_path.keys() - matches.keys())
    ]
    return matches, nr_paths, redundant_paths


def get_signature_kmers(app, parallel, paths):
    if parallel:
        series = PAR.as_completed(
            app,
            paths,
            max_workers=8,
        )
    else:
        series = map(app, paths)

    results = []
    # genomes with redundant signatures need to be grouped
    # a single member(s) should then be used and added
    # the skipped genomes should be logged as such
    redundant = []
    for r in track(
        series, total=len(paths), description="Finding unique", transient=True
    ):
        if not r:
            redundant.append(Path(r.source))
            continue
        results.append(r)
    return results, redundant


@composable.define_app
class matched_kmers:
    def __init__(self, signatures_path, moltype, k):
        table = dv_utils.load_tsv(signatures_path)
        patterns = {}
        for ref, kmers in table.tolist():
            patterns[ref] = set(kmers.split(","))
        self.patterns = patterns
        self.make_unique = seq_to_unique_kmers(k, moltype)
        self.k = k
        self.moltype = get_moltype("dna")

    def main(self, seq: c3_types.SeqType) -> tuple[str, set]:
        uniques = self.make_unique(seq)
        states = "".join(self.moltype)
        seqs = zeros(len(uniques), dtype=f"|U{self.k}")
        kmers = set(indices_to_seqs(uniques.data, seqs, states, self.k))
        return seq.name, {ref for ref, rk in self.patterns.items() if rk & kmers}


def make_signature_table(results, parallel):
    app = dv_utils.arr2str()
    if parallel:
        series = PAR.as_completed(
            app,
            results,
            max_workers=6,
        )
    else:
        series = map(app, results)

    rows = [
        r
        for r in track(
            series, total=len(results), transient=True, description="Convert to str"
        )
        if r
    ]
    return make_table(["name", "unique"], data=rows)
