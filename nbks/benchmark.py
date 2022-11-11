import time

from itertools import product
from pathlib import Path

import click

from cogent3 import make_seq, make_table
from cogent3.app.composable import define_app
from cogent3.util import parallel as PAR
from numpy.random import shuffle
from rich.progress import track

from divergent.record import seq_to_record
from divergent.records import max_divergent
from divergent.util import faster_load_fasta


try:
    from wakepy import set_keepawake, unset_keepawake
except (ImportError, NotImplementedError):
    # may not be installed, or on linux where this library doesn't work
    def _do_nothing_func(*args, **kwargs):
        ...

    set_keepawake, unset_keepawake = _do_nothing_func, _do_nothing_func


def _load_all_seqs(seqdir, limit):
    fasta_paths = list(seqdir.glob("*.fa*"))
    if limit:
        fasta_paths = fasta_paths[:limit]
    loader = faster_load_fasta()
    seqs = []
    for r in track(
        PAR.as_completed(loader, fasta_paths, max_workers=6),
        total=len(fasta_paths),
        description="Load seqs",
    ):
        name = list(r)[0]
        seqs.append(make_seq(r[name], name=name))
    return seqs


def time_seq2rec(seqs, k):
    # convert to records
    start = time.time()
    s2r = seq_to_record(k=k, moltype="dna")
    records = [s2r(s) for s in seqs]
    end = time.time()
    return end - start, records


def time_max_divergent(records):
    # convert to records
    start = time.time()
    result = max_divergent(records, min_size=7, max_size=12, stat="mean_jsd")
    end = time.time()
    return end - start


@define_app
class timed_run:
    def __init__(self, seqs):
        self.seqs = seqs

    def main(self, settings: tuple[int]) -> tuple:
        k, num_seqs, rep = settings
        shuffle(self.seqs)
        s2r_time, records = time_seq2rec(self.seqs[:num_seqs], k=k)
        md_time = time_max_divergent(records)
        return (k, num_seqs, rep, s2r_time, md_time)


_click_command_opts = dict(
    no_args_is_help=True, context_settings={"show_default": True}
)


@click.group(**_click_command_opts)
def main():
    """benchmarking exercises"""
    pass


@main.command(**_click_command_opts)
@click.argument("seqdir", type=Path)
@click.argument("outpath", type=Path)
@click.option("-L", "--limit", type=int)
@click.option("-P", "--parallel", is_flag=True)
def run(seqdir, outpath, limit, parallel):
    set_keepawake(keep_screen_awake=False)

    settings = list(product(range(3, 8, 2), range(50, 201, 50), range(5)))
    if limit:
        settings = settings[:5]
    seqs = _load_all_seqs(seqdir, limit)
    app = timed_run(seqs)
    series = (
        PAR.as_completed(app, settings, max_workers=6)
        if parallel
        else map(app, settings)
    )
    rows = list(track(series, total=len(settings)))
    table = make_table(
        header=["k", "num_seqs", "rep", "time(s2r)", "time(maxd)"], data=rows
    )
    table.write(outpath)

    unset_keepawake()


@main.command(**_click_command_opts)
@click.argument("seqdir", type=Path)
def profile(seqdir):

    settings = 7, 10, 1
    seqs = _load_all_seqs(seqdir, 100)
    app = timed_run(seqs)
    result = app(settings)
    print(result)


if __name__ == "__main__":
    main()
