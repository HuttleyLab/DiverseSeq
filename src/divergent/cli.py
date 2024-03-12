from os.path import exists
from pathlib import Path

import click
import h5py

from cogent3 import make_table
from cogent3.util import parallel as PAR
from rich.progress import track
from scitrack import CachingLogger

import divergent.util as dv_utils

from divergent.record import seq_to_record
from divergent.records import max_divergent, most_divergent


def _do_nothing_func(*args, **kwargs):
    ...


try:
    from wakepy import set_keepawake, unset_keepawake
except (ImportError, NotImplementedError):
    # may not be installed, or on linux where this library doesn't work
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
    """dvgt -- alignment free detection of most divergent sequences using JSD"""
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

_click_command_opts = dict(
    no_args_is_help=True, context_settings={"show_default": True}
)


@main.command(**_click_command_opts)
@_seqdir
@_outpath
@click.option(
    "-z", "--min_size", default=7, type=int, help="minimum size of divergent set"
)
@click.option(
    "-zp", "--max_size", default=None, type=int, help="maximum size of divergent set"
)
@click.option(
    "-x", "--fixed_size", is_flag=True, help="result will have size number of seqs"
)
@click.option("-k", type=int, default=3, help="k-mer size")
@click.option(
    "-st",
    "--stat",
    type=click.Choice(["total_jsd", "mean_delta_jsd", "mean_jsd"]),
    default="mean_delta_jsd",
    help="statistic to maximise",
)
@click.option("-p", "--parallel", is_flag=True, help="run in parallel")
@click.option("-L", "--limit", type=int, help="number of sequences to process")
@click.option(
    "-T",
    "--test_run",
    is_flag=True,
    help="reduce number of paths and size of query seqs",
)
@_verbose
def max(
    seqdir,
    outpath,
    min_size,
    max_size,
    fixed_size,
    stat,
    k,
    parallel,
    limit,
    test_run,
    verbose,
):
    """identify the seqs that maximise average delta JSD"""
    from numpy.random import shuffle

    if max_size is not None and min_size > max_size:
        click.secho(f"{min_size=} is greater than {max_size}", fg="red")
        exit(1)

    set_keepawake(keep_screen_awake=False)

    limit = 2 if test_run else limit or len(paths)
    paths = paths[:limit]
    app = dv_utils.seq_from_fasta() + seq_to_record(k=k, moltype="dna")


    records = []
    for result in track(series, total=len(paths), update_period=1):
        if not result:
            print(result)
            exit()
        records.append(result)

    shuffle(records)
    if fixed_size:
        sr = most_divergent(records, size=min_size, verbose=verbose > 0)
    else:
        sr = max_divergent(
            records,
            min_size=min_size,
            max_size=max_size,
            stat=stat,
            verbose=verbose > 0,
        )

    names, deltas = list(
        zip(*[(r.name, r.delta_jsd) for r in [sr.lowest] + sr.records])
    )
    table = make_table(data={"names": names, stat: deltas})
    outpath.parent.mkdir(parents=True, exist_ok=True)
    table.write(outpath)

    unset_keepawake()


@main.command(**_click_command_opts)
@_seqdir
@click.option(
    "-o",
    "--outpath",
    type=Path,
    help="location to write processed seqs as HDF5",
)
@click.option("-p", "--parallel", is_flag=True, default=False, help="run in parallel")
@click.option(
    "-F",
    "--force_overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing file if it exists",
)
@click.option(
    "-m",
    "--moltype",
    type=click.Choice(["dna", "rna"]),
    default="dna",
    help="Molecular type of sequences, defaults to DNA",
)
def prep(seqdir, outpath, parallel, force_overwrite, moltype):
    """Writes processed sequences to an HDF5 file."""

    set_keepawake(keep_screen_awake=False)

    if seqdir.is_file():
        paths = dv_utils.get_seq_identifiers([seqdir])
    else:
        paths = list(seqdir.glob("**/*.fa*"))
        if not paths:
            click.secho(f"{seqdir} contains no fasta paths", fg="red")
            exit(1)

    outpath_h5 = f"{outpath}.h5"

    app = dv_utils.seq_from_fasta() + dv_utils._seqs_to_array()

    if parallel:
        workers = PAR.get_size() - 1
        series = PAR.as_completed(app, paths, max_workers=workers)
    else:
        series = map(app, paths)

    records = []
    for result in track(series, total=len(paths), update_period=1):
        if not result:
            print(result)
            exit()
        records.append(result)

    write_mode = "w" if force_overwrite else "w-"

    try:
        with h5py.File(outpath_h5, mode=write_mode) as f:
            for name, seq in records:
                dataset = f.create_dataset(
                    name=name,
                    data=seq,
                    dtype="u1",
                )
                dataset.attrs["moltype"] = moltype
            num_records = len(f.keys())
    except FileExistsError:
        click.secho(
            f"FileExistsError: Unable to create file at {outpath_h5} because a file "
            f"with the same name already exists. Please choose a different "
            f"name or use the -F flag to force overwrite the existing file.",
            fg="red",
        )
        exit(1)

    click.secho(
        f"Successfully processed {num_records} sequences and wrote to {outpath_h5}",
        fg="green",
    )

    unset_keepawake()


if __name__ == "__main__":
    main()
