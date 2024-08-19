import sys
import tempfile
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import click
from cogent3.app import data_store as c3_data_store
from scitrack import CachingLogger

from divergent import __version__
from divergent import data_store as dvgt_data_store
from divergent import records as dvgt_records
from divergent import util as dvgt_util
from divergent.io import dvgt_file_to_dir, dvgt_load_seqs, dvgt_write_seqs

LOGGER = CachingLogger()


class OrderedGroup(click.Group):
    """custom group class to ensure help function returns commands in desired order.
    class is adapted from Максим Стукало's answer to
    https://stackoverflow.com/questions/47972638/how-can-i-define-the-order-of-click-sub-commands-in-help
    """

    def __init__(
        self,
        name: str | None = None,
        commands: Mapping[str, click.Command] | None = None,
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


_verbose = click.option(
    "-v",
    "--verbose",
    count=True,
    help="is an integer indicating number of cl occurrences",
)


_outpath = click.option(
    "-o",
    "--outpath",
    type=Path,
    help="the input string will be cast to Path instance",
)


_click_command_opts = dict(
    no_args_is_help=True,
    context_settings={"show_default": True},
)

_overwrite = click.option(
    "-F",
    "--force_overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing file if it exists",
)
_suffix = click.option("-sf", "--suffix", default="fa", help="sequence file suffix")
_num_procs = click.option("-np", "--numprocs", default=1, help="number of processes")
_parallel = click.option(
    "-p",
    "--parallel",
    is_flag=True,
    default=False,
    help="run in parallel",
)
_limit = click.option("-L", "--limit", type=int, help="number of sequences to process")
_seqfile = click.option(
    "-s",
    "--seqfile",
    required=True,
    type=Path,
    help="path to .dvtgseqs file",
)
_k = click.option("-k", type=int, default=3, help="k-mer size")


@main.command(**_click_command_opts)
@click.option(
    "-s",
    "--seqdir",
    required=True,
    type=Path,
    help="directory containing sequence files",
)
@_suffix
@click.option(
    "-o",
    "--outpath",
    required=True,
    type=Path,
    help="location to write processed seqs",
)
@_parallel
@_overwrite
@click.option(
    "-m",
    "--moltype",
    type=click.Choice(["dna", "rna"]),
    default="dna",
    help="Molecular type of sequences, defaults to DNA",
)
@_limit
def prep(seqdir, suffix, outpath, parallel, force_overwrite, moltype, limit):
    """Writes processed sequences to an outpath ending with .dvgtseqs file."""
    dvgtseqs_path = outpath.with_suffix(".dvgtseqs")
    if dvgtseqs_path.exists() and not force_overwrite:
        click.secho(
            "A file with the same name already exists. Existing data members will be skipped. "
            "Use the -F flag if you want to overwrite the existing file.",
            fg="blue",
        )
    elif dvgtseqs_path.exists() and force_overwrite:
        dvgtseqs_path.unlink()

    if suffix.startswith("."):
        suffix = suffix[1:]

    seq_format = dvgt_util.get_seq_file_format(suffix)
    if seq_format is None:
        click.secho(
            f"Unrecognised sequence file suffix '{suffix}'",
            fg="red",
        )
        sys.exit(1)

    with dvgt_util.keep_running(), tempfile.TemporaryDirectory() as tmp_dir:
        if seqdir.is_file():
            convert2dstore = dvgt_file_to_dir(dest=tmp_dir, limit=limit)
            in_dstore = convert2dstore(seqdir)
        else:
            in_dstore = c3_data_store.DataStoreDirectory(source=seqdir, suffix=suffix)
            if not len(in_dstore):
                click.secho(
                    f"{seqdir} contains no files matching '*.{suffix}'",
                    fg="red",
                )
                sys.exit(1)

        out_dstore = dvgt_data_store.HDF5DataStore(source=dvgtseqs_path, mode="w")

        prep_pipeline = dvgt_load_seqs(
            moltype=moltype,
            seq_format=seq_format,
        ) + dvgt_write_seqs(
            data_store=out_dstore,
        )
        result = prep_pipeline.apply_to(
            in_dstore,
            show_progress=True,
            parallel=parallel,
        )

    out_dstore.close()
    click.secho(
        f"Successfully created {result}",
        fg="green",
    )


@main.command(**_click_command_opts)
@_seqfile
@_outpath
@click.option(
    "-z",
    "--min_size",
    default=7,
    type=int,
    help="minimum size of divergent set",
)
@click.option(
    "-zp",
    "--max_size",
    default=None,
    type=int,
    help="maximum size of divergent set",
)
@_k
@click.option(
    "-st",
    "--stat",
    type=click.Choice(["total_jsd", "mean_delta_jsd", "mean_jsd"]),
    default="mean_delta_jsd",
    help="statistic to maximise",
)
@_num_procs
@_limit
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
    stat,
    k,
    numprocs,
    limit,
    test_run,
    verbose,
):
    """Identify the seqs that maximise average delta JSD"""

    if max_size is not None and min_size > max_size:
        click.secho(f"{min_size=} cannot be greater than {max_size=}", fg="red")
        sys.exit(1)

    if seqfile.suffix != ".dvgtseqs":
        click.secho(
            "Sequence data needs to be preprocessed, use 'dvgt prep'",
            fg="red",
        )
        sys.exit(1)

    seqids = dvgt_data_store.get_seqids_from_store(seqfile)
    limit = 2 if test_run else limit
    if limit is not None:
        seqids = seqids[:limit]

    app = dvgt_records.dvgt_max(
        seq_store=seqfile,
        k=k,
        min_size=min_size,
        max_size=max_size,
        stat=stat,
        limit=limit,
        verbose=verbose,
    )
    finalise = dvgt_records.dvgt_final_max(
        stat=stat,
        min_size=min_size,
        verbose=verbose,
    )
    result = dvgt_records.apply_app(
        app=app,
        seqids=seqids,
        numprocs=1,
        verbose=verbose,
        finalise=finalise,
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    table = result.to_table()
    table.write(outpath)


@main.command(**_click_command_opts)
@_seqfile
@_outpath
@click.option(
    "-n",
    "--number",
    type=int,
    help="number of seqs in divergent set",
    required=True,
)
@_k
@_num_procs
@_limit
@_verbose
def nmost(
    seqfile,
    outpath,
    number,
    k,
    numprocs,
    limit,
    verbose,
):
    """Identify n seqs that maximise average delta JSD"""

    if seqfile.suffix != ".dvgtseqs":
        click.secho(
            "Sequence data needs to be preprocessed, use 'dvgt prep'",
            fg="red",
        )
        sys.exit(1)

    seqids = dvgt_data_store.get_seqids_from_store(seqfile)
    if limit is not None:
        seqids = seqids[:limit]

    app = dvgt_records.dvgt_nmost(
        seq_store=seqfile,
        n=number,
        k=k,
        limit=limit,
        verbose=verbose,
    )
    result = dvgt_records.apply_app(
        app=app,
        seqids=seqids,
        numprocs=numprocs,
        verbose=verbose,
        finalise=dvgt_records.dvgt_final_nmost(),
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    table = result.to_table()
    table.write(outpath)


if __name__ == "__main__":
    main()
