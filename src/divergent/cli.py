import shutil

from pathlib import Path

import click

from cogent3 import get_moltype, make_table
from cogent3.util import parallel as PAR
from rich.progress import track
from scitrack import CachingLogger

import divergent.util as dv_utils

from divergent.record import seq_to_kmer_counts, seq_to_unique_kmers
from divergent.unique import (
    get_signature_kmers,
    make_signature_table,
    non_redundant,
    signature_kmers,
)


try:
    from wakepy import set_keepawake, unset_keepawake
except (ImportError, NotImplementedError):
    # may not be installed, or on linux where this library doesn't work
    def _do_nothing_func(*args, **kwargs):
        ...

    set_keepawake, unset_keepawake = _do_nothing_func, _do_nothing_func


__author__ = "Gavin Huttley"
__copyright__ = "Copyright 2022-, Gavin Huttley"
__credits__ = ["Gavin Huttley"]
__license__ = "BSD"
__version__ = "2022.8.4"  # A DATE BASED VERSION
__maintainer__ = "Gavin Huttley"
__email__ = "Gavin.Huttley@anu.edu.au"
__status__ = "alpha"


LOGGER = CachingLogger()


@click.group()
@click.version_option(__version__)  # add version option
def main():
    """dvgt -- alignment free measurement of divergent sequences"""
    pass


_verbose = click.option(
    "-v",
    "--verbose",
    count=True,
    help="is an integer indicating number of cl occurrences",
)

# you can define custom parsers / validators
def _parse_csv_arg(*args) -> list:
    return args[-1].split(",")


_names = click.option(
    "--names",
    callback=_parse_csv_arg,
    help="converts comma separated values",
)

_outpath = click.option(
    "-o", "--outpath", type=Path, help="the input string will be cast to Path instance"
)

_outdir = click.option(
    "-o",
    "--outdir",
    type=Path,
    help="directory to write compressed json",
)


def _make_outpath(outdir, path, k):
    return outdir / f"{path.stem.split('.')[0]}-{k}-mer.json.blosc2"


_seqdir = click.option(
    "-s",
    "--seqdir",
    required=True,
    type=Path,
    help="directory containing fasta formatted sequence files",
)


@main.command(no_args_is_help=True)
@_seqdir
@_outdir
@click.option("-k", type=int, default=7, help="k-mer size")
@click.option("-p", "--parallel", is_flag=True, help="run in parallel")
@click.option(
    "-U", "--unique", is_flag=True, help="unique kmers only, not their counts"
)
@click.option("-L", "--limit", type=int, help="number of records to process")
@click.option("-O", "--overwrite", is_flag=True, help="overwrite existing")
@_verbose
def seqs2kmers(seqdir, outdir, k, parallel, unique, limit, overwrite, verbose):
    """write kmer data for seqs in seqdir"""
    if overwrite and outdir.exists():
        shutil.rmtree(outdir, ignore_errors=True)

    outdir.mkdir(parents=True, exist_ok=True)

    paths = [str(p) for p in seqdir.glob("**/*.fa*")]
    if limit:
        paths = paths[:limit]

    kwargs = dict(k=k, moltype="dna")
    app = (
        dv_utils.seq_from_fasta()
        + (seq_to_unique_kmers(**kwargs) if unique else seq_to_kmer_counts(**kwargs))
        + dv_utils.bundle_data()
    )

    not_done = []
    for p in paths:
        op = _make_outpath(outdir, Path(p), k)
        if op.exists():
            continue
        not_done.append(p)

    paths = paths if overwrite else not_done

    if parallel:
        series = PAR.as_completed(app, paths, max_workers=6)
    else:
        series = map(app, paths)

    set_keepawake(keep_screen_awake=False)

    for result in track(series, total=len(paths), update_period=1):
        if not result:
            print(result)
            exit()
        b, source = result
        path = Path(source)
        outpath = outdir / f"{path.stem.split('.')[0]}-{k}-mer.pickle.blosc2"
        outpath.write_bytes(b)
        del b, source

    unset_keepawake()


@main.command(no_args_is_help=True)
@click.option(
    "-i",
    "--indir",
    required=True,
    type=Path,
    help="directory containing pickled k-mers",
)
@_seqdir
@_outdir
@click.option("-p", "--parallel", is_flag=True, help="run in parallel")
@click.option("-L", "--limit", type=int, help="number of records to process")
@_verbose
def sig_kmers(indir, seqdir, outdir, parallel, limit, verbose):
    """signature k-mers uniquely identify each sequence.

    Sequences that are redundant are excluded and written out as
    a tsv file to outdir.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{indir.stem}-unique.tsv"
    table_outpath = outdir / f"{indir.stem}-redundant.tsv"

    paths = list(indir.glob("**/*.blosc2"))

    set_keepawake(keep_screen_awake=False)

    limit = limit or len(paths)
    app = signature_kmers(paths[:limit], verbose=verbose)

    results, redundant = get_signature_kmers(app, parallel, paths[:limit])
    if redundant:
        nr, nr_paths, r_paths = non_redundant(redundant, seqdir)

        app = signature_kmers([p for p in paths if p not in r_paths])
        new_results, new_redundant = get_signature_kmers(app, parallel, nr_paths)
        results.extend(new_results)
        redund_map = [(k, ",".join(nr[k])) for k in nr]
        table = make_table(header=["representative", "matched to"], data=redund_map)
        table.write(table_outpath)

    unique = make_signature_table(results, parallel)
    unique.write(table_outpath)
    click.secho(f"Wrote {outpath}", fg="green")

    unset_keepawake()

    unset_keepawake()


if __name__ == "__main__":
    main()
