import sys
import tempfile
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import click
from cogent3.app.composable import NotCompleted
from cogent3.app.data_store import DataStoreDirectory
from scitrack import CachingLogger

from divergent import __version__
from divergent import util as dvgt_util
from divergent.io import dvgt_load_seqs, dvgt_write_prepped_seqs, dvgt_write_seq_store
from divergent.records import dvgt_calc

try:
    from wakepy.keep import running as keep_running

    # trap flaky behaviour on linux
    with keep_running():
        ...

except (NotImplementedError, ImportError):
    from divergent.util import fake_wake as keep_running


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


@main.command(**_click_command_opts)
@click.option(
    "-s",
    "--seqdir",
    required=True,
    type=Path,
    help="directory containing sequence files",
)
@click.option("-sf", "--suffix", default="fa", help="sequence file suffix")
@click.option(
    "-o",
    "--outpath",
    required=True,
    type=Path,
    help="location to write processed seqs",
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

    with keep_running(), tempfile.TemporaryDirectory() as tmp_dir:
        if seqdir.is_file():
            convert2dstore = dvgt_write_seq_store(dest=tmp_dir, limit=limit)
            in_dstore = convert2dstore(seqdir)
        else:
            in_dstore = DataStoreDirectory(source=seqdir, suffix=suffix)
            if not len(in_dstore):
                click.secho(
                    f"{seqdir} contains no files matching '*.{suffix}'",
                    fg="red",
                )
                sys.exit(1)

        prep_pipeline = dvgt_load_seqs(
            moltype=moltype,
            seq_format=seq_format,
        ) + dvgt_write_prepped_seqs(
            dvgtseqs_path,
            limit=limit,
        )
        result = prep_pipeline.apply_to(
            in_dstore,
            show_progress=True,
            parallel=parallel,
        )

    click.secho(
        f"Successfully created {result}",
        fg="green",
    )


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
@click.option(
    "-x",
    "--fixed_size",
    is_flag=True,
    help="result will have number of seqs of `min_size`",
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
        sys.exit(1)

    if seqfile.suffix != ".dvgtseqs":
        click.secho(
            "Sequence data needs to be preprocessed, run 'dvgt prep -s "
            "<path_to_your_seqs.fasta> -o <path_to_write_processed_seqs.dvgtseqs>' "
            "to prepare the sequence data",
            fg="red",
        )
        sys.exit(1)

    with keep_running():
        limit = 2 if test_run else limit
        mode = "most" if fixed_size else "max"

        dvgt_app = dvgt_calc(
            mode=mode,
            k=k,
            parallel=parallel,
            limit=limit,
            min_size=min_size,
            max_size=max_size,
            stat=stat,
            verbose=verbose,
        )

        table = dvgt_app(seqfile)

        if isinstance(table, NotCompleted):
            click.secho(
                message=f"{table.type}: {table.message}",
                fg="red",
            )
            sys.exit(1)

        outpath.parent.mkdir(parents=True, exist_ok=True)
        table.write(outpath)


if __name__ == "__main__":
    main()
