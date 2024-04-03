import itertools

from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Optional

import click
import h5py
import hdf5plugin

from cogent3 import get_moltype, make_table
from cogent3.util import parallel as PAR
from numpy import empty, random, uint8
from rich.progress import track
from scitrack import CachingLogger

from divergent import loader as dv_loader
from divergent.record import SeqArray, seq_to_seqarray, seqarray_to_record
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


class OrderedGroup(click.Group):
    """custom group class to ensure help function returns commands in desired order.
    class is adapted from Максим Стукало's answer to
    https://stackoverflow.com/questions/47972638/how-can-i-define-the-order-of-click-sub-commands-in-help
    """

    def __init__(
        self,
        name: Optional[str] = None,
        commands: Optional[Mapping[str, click.Command]] = None,
        **kwargs,
    ):
        super().__init__(name, commands, **kwargs)
        #: the registered subcommands by their exported names.
        self.commands = commands or OrderedDict()

    def list_commands(self, ctx: click.Context) -> Mapping[str, click.Command]:
        return self.commands


@click.group(cls=OrderedGroup)
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


def make_task_iterator(func, tasks, parallel):
    if parallel:
        workers = PAR.get_size() - 1
        return PAR.as_completed(func, tasks, max_workers=workers)
    else:
        return map(func, tasks)


_click_command_opts = dict(
    no_args_is_help=True, context_settings={"show_default": True}
)


@main.command(**_click_command_opts)
@click.option(
    "-s",
    "--seqdir",
    required=True,
    type=Path,
    help="directory containing fasta formatted sequence files",
)
@click.option(
    "-o",
    "--outpath",
    required=True,
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
        paths = dv_loader.get_seq_identifiers([seqdir])
    else:
        paths = list(seqdir.glob("**/*.fa*"))
        if not paths:
            click.secho(f"{seqdir} contains no fasta paths", fg="red")
            exit(1)

    suffixed_outpath = outpath.with_suffix(".dvgtseqs")
    if suffixed_outpath.exists() and not force_overwrite:
        click.secho(
            f"FileExistsError: Unable to create file at {suffixed_outpath} because a file "
            f"with the same name already exists. Please choose a different "
            f"name or use the -F flag to force overwrite the existing file.",
            fg="red",
        )
        exit(1)

    fasta_to_seqarray = dv_loader.seq_from_fasta(moltype=moltype) + seq_to_seqarray()
    tasks = make_task_iterator(fasta_to_seqarray, paths, parallel)

    records = []
    for result in track(tasks, total=len(paths), update_period=1):
        if not result:
            print(result)
            exit()
        records.append(result)

    write_mode = "w" if force_overwrite else "w-"
    with h5py.File(suffixed_outpath, mode=write_mode) as f:
        # only support collection of seqs of the same moltype,
        # so moltype can be stored as a top level attribute
        f.attrs["moltype"] = moltype
        f.attrs["source"] = str(seqdir)

        for record in records:
            dset = f.create_dataset(
                name=record.seqid, data=record.data, dtype="u1", **hdf5plugin.Blosc2()
            )
            dset.attrs["source"] = str(record.source)
        num_records = len(f.keys())

    click.secho(
        f"Successfully processed {num_records} sequences and wrote to {suffixed_outpath}",
        fg="green",
    )

    unset_keepawake()


@main.command(**_click_command_opts)
@click.option(
    "-s",
    "--seqfile",
    required=True,
    type=Path,
    help="HDF5 file containing sequences, must have been processed by the 'prep' command",
)
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
    seqfile,
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
    """Identify the seqs that maximise average delta JSD"""

    if max_size is not None and min_size > max_size:
        click.secho(f"{min_size=} is greater than {max_size}", fg="red")
        exit(1)

    if seqfile.suffix != ".dvgtseqs":
        click.secho(
            f"Sequence data needs to be preprocessed, run 'dvgt prep -s "
            "<path_to_your_seqs.fasta> -o <path_to_write_processed_seqs.dvgtseqs>' "
            "to prepare the sequence data",
            fg="red",
        )
        exit(1)

    set_keepawake(keep_screen_awake=False)

    with h5py.File(seqfile, mode="r") as f:
        limit = 2 if test_run else limit or len(f.keys())
        orig_moltype = get_moltype(f.attrs["moltype"])

        seqs = []
        for name, dset in itertools.islice(f.items(), limit):
            out = empty(len(dset), dtype=uint8)
            dset.read_direct(out)
            seq = SeqArray(
                seqid=name, data=out, moltype=orig_moltype, source=f.attrs["source"]
            )
            seqs.append(seq)

    make_records = seqarray_to_record(k=k, moltype=orig_moltype)
    tasks = make_task_iterator(make_records, seqs, parallel)

    records = []
    for result in track(tasks, total=len(seqs), update_period=1):
        if not result:
            print(result)
            exit()
        records.append(result)

    random.shuffle(records)
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


if __name__ == "__main__":
    main()
