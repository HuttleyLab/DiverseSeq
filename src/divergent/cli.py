import os
import h5py 
import tempfile
import itertools

from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Optional

import click
from rich.progress import track

from cogent3 import make_table
from cogent3.app.data_store import DataStoreDirectory
from cogent3.util import parallel as PAR
from numpy import random, empty, uint8
from scitrack import CachingLogger

from divergent.data_store import HDF5DataStore
from divergent.loader import (

    dvgt_load_seqs,
    dvgt_seq_file_to_data_store,
)
from divergent.record import seqarray_to_record, SeqArray
from divergent.records import max_divergent, most_divergent
from divergent.writer import dvgt_write_prepped_seqs


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
@click.option("-L", "--limit", type=int, help="number of sequences to process")
def prep(seqdir, outpath, parallel, force_overwrite, moltype, limit):
    """Writes processed sequences to an HDF5 file."""

    set_keepawake(keep_screen_awake=False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dvgtseqs_path = outpath.with_suffix(".dvgtseqs")
        if dvgtseqs_path.exists() and not force_overwrite:
            click.secho(
                "A file with the same name already exists. Existing data members will be skipped. "
                "Use the -F flag if you want to overwrite the existing file.",
                fg="blue",
            )
        elif dvgtseqs_path.exists() and force_overwrite:
            os.remove(dvgtseqs_path)

        if seqdir.is_file():
            convert2dstore = dvgt_seq_file_to_data_store(dest=tmp_dir, limit=limit)
            in_dstore = convert2dstore(seqdir)
        else:
            seqfile_suffix = seqdir.suffix
            supported_suffixes = [".fa", ".fasta", ".fna", ".faa"]
            if seqfile_suffix not in supported_suffixes:
                click.secho(
                    f"Input file of type {seqfile_suffix} is not a supported types: {', '.join(supported_suffixes)}",
                    fg="red",
                )
                exit(1)

            in_dstore = DataStoreDirectory(source=seqdir, suffix=seqfile_suffix)

        out_dstore = HDF5DataStore(source=dvgtseqs_path, limit=limit)
        prep_pipeline = dvgt_load_seqs(moltype=moltype) + dvgt_write_prepped_seqs(
            out_dstore
        )
        result = prep_pipeline.apply_to(
            in_dstore, show_progress=True, parallel=parallel
        )

        click.secho(
            f"Successfully created {result}",
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
    "-m",
    "--moltype",
    type=click.Choice(["dna", "rna"]),
    default="dna",
    help="Molecular type of sequences, defaults to DNA",
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
    moltype,
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
        orig_moltype = "dna"
        seqs = []
        for name, dset in itertools.islice(f.items(), limit):
            if isinstance(dset, h5py.Group):
                continue
            # initalise array
            out = empty(len(dset), dtype=uint8)
            # read db content directly into initialised array
            dset.read_direct(out)
            seqs.append(SeqArray(seqid=name, data=out, moltype="dna", source=name))

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
