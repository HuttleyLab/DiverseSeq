import random
import sys
import tempfile
import time
import typing
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import click
import numpy
import rich.progress as rich_progress
from cogent3.app import data_store as c3_data_store
from scitrack import CachingLogger

from diverse_seq import __version__
from diverse_seq import cluster as dvs_cluster
from diverse_seq import data_store as dvs_data_store
from diverse_seq import io as dvs_io
from diverse_seq import records as dvs_records
from diverse_seq import util as dvs_util

if typing.TYPE_CHECKING:
    from click.core import Context, Option

LOGGER = CachingLogger()


def _get_seed(ctx: "Context", param: "Option", value: str | None) -> int:
    value = time.time() if value is None else value
    return int(value)


# custom group class to ensure help function returns commands in desired order.
# class is adapted from Максим Стукало's answer to
# https://stackoverflow.com/questions/47972638/how-can-i-define-the-order-of-click-sub-commands-in-help


class OrderedGroup(click.Group):
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


_click_command_opts = {
    "no_args_is_help": True,
    "context_settings": {"show_default": True},
}


@click.group(cls=OrderedGroup)
@click.version_option(__version__)  # add version option
def main() -> None:
    """dvs -- alignment free detection of the most diverse sequences using JSD"""


_hide_progress = click.option(
    "-hp",
    "--hide_progress",
    is_flag=True,
    callback=dvs_util._hide_progress,
    help="hide progress bars",
)
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
    help="path to write output file",
    required=True,
)
_overwrite = click.option(
    "-F",
    "--force_overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing file if it exists",
)
_include = click.option(
    "-i",
    "--include",
    callback=dvs_util._comma_sep_or_file,  # noqa: SLF001
    help="seqnames to include in divergent set",
)
_suffix = click.option("-sf", "--suffix", default="fa", help="sequence file suffix")
_numprocs = click.option("-np", "--numprocs", default=1, help="number of processes")
_limit = click.option("-L", "--limit", type=int, help="number of sequences to process")
_moltype = click.option(
    "-m",
    "--moltype",
    type=click.Choice(["dna", "rna"]),
    default="dna",
    help="Molecular type of sequences",
)
_seqfile = click.option(
    "-s",
    "--seqfile",
    required=True,
    type=Path,
    callback=dvs_util._check_h5_dstore,
    help="path to .dvseqs file",
)
_k = click.option("-k", type=int, default=6, help="k-mer size")

_seed = click.option(
    "--seed",
    type=int,
    callback=_get_seed,
    help="seed for random number generator, defaults to system clock",
)

_click_opts = _click_command_opts.copy()
_click_opts.pop("no_args_is_help")


@main.command(**_click_opts)
@click.option(
    "-o",
    "--outpath",
    type=Path,
    default=Path("demo.fa"),
    help="write a demo fasta file",
)
def demo_data(outpath: Path) -> None:
    """Export a demo sequence file"""
    from diverse_seq import load_sample_data

    seqs = load_sample_data()
    seqs.write(outpath, file_format="fasta")
    dvs_util.print_colour(f"Wrote '{outpath!s}'", "green")


@main.command(**_click_command_opts)
@click.option(
    "-s",
    "--seqdir",
    required=True,
    type=Path,
    help="one sequence file, or a directory containing multiple sequence files",
)
@_suffix
@click.option(
    "-o",
    "--outpath",
    required=True,
    type=Path,
    help="write processed seqs to this filename",
)
@_numprocs
@_overwrite
@_moltype
@_limit
@_hide_progress
def prep(
    seqdir: Path,
    suffix: str,
    outpath: Path,
    numprocs: int,
    force_overwrite: bool,
    moltype: str,
    limit: int | None,
    hide_progress: bool,
) -> None:
    """Writes processed sequences to a <HDF5 file>.dvseqs."""
    dvseqs_path = outpath.with_suffix(".dvseqs")
    if dvseqs_path.exists() and not force_overwrite:
        dvs_util.print_colour(
            "A file with the same name already exists. Existing data members will be skipped. "
            "Use the -F flag if you want to overwrite the existing file.",
            "blue",
        )
        sys.exit(1)

    if dvseqs_path.exists() and force_overwrite:
        dvseqs_path.unlink()

    suffix = suffix.removeprefix(".")

    seq_format = dvs_util.get_seq_file_format(suffix)
    if seq_format is None:
        dvs_util.print_colour(
            f"Unrecognised sequence file suffix '{suffix}'",
            "red",
        )
        sys.exit(1)

    with dvs_util.keep_running(), tempfile.TemporaryDirectory() as tmp_dir:
        if seqdir.is_file():
            convert2dstore = dvs_io.dvs_file_to_dir(dest=tmp_dir)
            in_dstore = convert2dstore(seqdir)  # pylint: disable=not-callable
        else:
            in_dstore = c3_data_store.DataStoreDirectory(source=seqdir, suffix=suffix)

        if len(in_dstore) < 5:
            msg = f"Num files matching '{seqdir}/*.{suffix}' = {len(in_dstore)} < 5."
            if seqdir.is_dir():
                msg = f"{msg} Did you mean to pass a file path instead?"
            dvs_util.print_colour(msg, "red")
            sys.exit(1)

        if limit is not None:
            members = in_dstore.completed[:]
            random.shuffle(members)
            in_dstore = members[:limit]

        out_dstore = dvs_data_store.HDF5DataStore(source=dvseqs_path, mode="w")

        loader = dvs_io.dvs_load_seqs(
            moltype=moltype,
            seq_format=seq_format,
        )
        writer = dvs_io.dvs_write_seqs(
            data_store=out_dstore,
        )

        with rich_progress.Progress(
            rich_progress.TextColumn("[progress.description]{task.description}"),
            rich_progress.BarColumn(),
            rich_progress.TaskProgressColumn(),
            rich_progress.TimeRemainingColumn(),
            rich_progress.TimeElapsedColumn(),
            disable=hide_progress,
        ) as progress:
            convert = progress.add_task("Processing sequences", total=len(in_dstore))
            for r in loader.as_completed(
                in_dstore,
                show_progress=False,
                parallel=numprocs > 1,
                par_kw={"max_workers": numprocs},
            ):
                if not r:
                    dvs_util.print_colour(str(r), "red")
                    sys.exit(1)
                writer(r)  # pylint: disable=not-callable
                progress.update(convert, advance=1, refresh=True)
                del r

    out_dstore.close()
    dvs_util.print_colour(
        f"Successfully created '{out_dstore.source!s}'",
        "green",
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
    type=click.Choice(["stdev", "cov"]),
    default="stdev",
    help="statistic to maximise",
)
@_seed
@_include
@_numprocs
@_limit
@click.option(
    "-T",
    "--test_run",
    is_flag=True,
    help="reduce number of paths and size of query seqs",
)
@_verbose
@_hide_progress
def max(  # noqa: A001
    seqfile: Path,
    outpath: Path,
    min_size: int,
    max_size: int | None,
    stat: str,
    seed: int,
    include: list[str] | None,
    k: int,
    numprocs: int,
    limit: int | None,
    test_run: bool,
    verbose: int,
    hide_progress: bool,
) -> None:
    """Identify the seqs that maximise average delta JSD"""
    if max_size is not None and min_size > max_size:
        dvs_util.print_colour(f"{min_size=} cannot be greater than {max_size=}", "red")
        sys.exit(1)

    if seqfile.suffix != ".dvseqs":
        dvs_util.print_colour(
            "Sequence data needs to be preprocessed, use 'dvs prep'",
            "red",
        )
        sys.exit(1)

    seqids = dvs_data_store.get_seqids_from_store(seqfile)
    if len(seqids) < min_size:
        msg = f"Num seqs in {seqfile}={len(seqids)} < {min_size=}. Nothing to do!"
        dvs_util.print_colour(msg, "red")
        sys.exit(1)

    if include and not set(include) <= set(seqids):
        dvs_util.print_colour(
            f"provided {include=} not in the sequence data",
            "red",
        )
        sys.exit(1)

    if verbose:
        dvs_util.print_colour(f"Using random seed: {seed}", "blue")

    rng = numpy.random.default_rng(seed=seed)
    rng.shuffle(seqids)

    limit = min_size + 1 if test_run else limit
    if limit is not None:
        seqids = seqids[:limit]

    app = dvs_records.select_max(
        seq_store=seqfile,
        k=k,
        min_size=min_size,
        max_size=max_size,
        stat=stat,
        limit=limit,
        verbose=verbose,
    )
    # turn off pylint check, since the function is made into a class
    finalise = dvs_records.select_final_max(  # pylint: disable=no-value-for-parameter
        stat=stat,
        min_size=min_size,
        verbose=verbose,
    )
    result = dvs_records.apply_app(
        app=app,
        seqids=seqids,
        numprocs=numprocs,
        verbose=verbose,
        hide_progress=hide_progress,
        finalise=finalise,
    )

    # user requested inclusions are added to the selected divergent set
    if include:
        include_records = dvs_records.records_from_seq_store(
            seq_store=seqfile,
            seq_names=include,
            k=k,
            limit=None,
        )
        result = dvs_records.SummedRecords.from_records(
            result.all_records() + include_records,
        )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    table = result.to_table()
    table.write(outpath)
    dvs_util.print_colour(
        f"{table.shape[0]} divergent sequences IDs written to '{outpath!s}'",
        "green",
    )


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
@_seed
@_include
@_numprocs
@_limit
@_verbose
@_hide_progress
def nmost(
    seqfile: Path,
    outpath: Path,
    number: int,
    k: int,
    include: list[str] | None,
    seed: int,
    numprocs: int,
    limit: int | None,
    verbose: int,
    hide_progress: bool,
) -> None:
    """Identify n seqs that maximise average delta JSD"""

    if seqfile.suffix != ".dvseqs":
        dvs_util.print_colour(
            "Sequence data needs to be preprocessed, use 'dvs prep'",
            "red",
        )
        sys.exit(1)

    seqids = dvs_data_store.get_seqids_from_store(seqfile)
    if len(seqids) < number:
        msg = f"Num seqs in {seqfile}={len(seqids)} < {number=}. Nothing to do!"
        dvs_util.print_colour(msg, "red")
        sys.exit(1)

    if include and not set(include) <= set(seqids):
        dvs_util.print_colour(
            f"provided {include=} not in the sequence data",
            "red",
        )
        sys.exit(1)

    if verbose:
        dvs_util.print_colour(f"Using random seed: {seed}", "blue")

    rng = numpy.random.default_rng(seed=seed)
    rng.shuffle(seqids)
    if limit is not None:
        seqids = seqids[:limit]

    app = dvs_records.select_nmost(
        seq_store=seqfile,
        n=number,
        k=k,
        limit=limit,
        verbose=verbose,
    )
    result = dvs_records.apply_app(
        app=app,
        seqids=seqids,
        numprocs=numprocs,
        verbose=verbose,
        hide_progress=hide_progress,
        finalise=dvs_records.dvs_final_nmost(),  # pylint: disable=no-value-for-parameter
    )
    # user requested inclusions are added to the selected divergent set
    if include:
        include_records = dvs_records.records_from_seq_store(
            seq_store=seqfile,
            seq_names=include,
            k=k,
            limit=None,
        )
        result = dvs_records.SummedRecords.from_records(
            result.all_records() + include_records,
        )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    table = result.to_table()
    table.write(outpath)
    dvs_util.print_colour(
        f"{table.shape[0]} divergent sequences IDs written to '{outpath!s}'",
        "green",
    )


@main.command(**_click_command_opts)
@_seqfile
@_outpath
@_moltype
@_k
@click.option(
    "--sketch-size",
    type=int,
    default=None,
    help="sketch size for mash distance, e.g. 3000",
)
@click.option(
    "-d",
    "--distance",
    type=click.Choice(["mash", "euclidean"]),
    default="mash",
    help="distance measure for tree construction",
)
@click.option(
    "-c",
    "--canonical-kmers",
    is_flag=True,
    default=False,
    help="consider kmers identical to their reverse complement",
)
@_seed
@_limit
@_numprocs
@_hide_progress
@_verbose
def ctree(
    seqfile: Path,
    outpath: Path,
    moltype: str,
    k: int,
    sketch_size: int | None,
    distance: str,
    canonical_kmers: bool,
    seed: int,
    limit: int | None,
    numprocs: int,
    hide_progress: bool,
    verbose: int,
):
    """Quickly compute a cluster tree based on kmers for a collection of sequences."""
    if seqfile.suffix != ".dvseqs":
        dvs_util.print_colour(
            "Sequence data needs to be preprocessed, use 'dvs prep'",
            "red",
        )
        sys.exit(1)

    if outpath is None:
        dvs_util.print_colour(
            "Specify a file name to write the newick tree to.",
            "red",
        )
        sys.exit(1)

    if sketch_size is None and distance == "mash":
        dvs_util.print_colour(
            "Sketch size must be specified for mash distance.",
            "red",
        )
        sys.exit(1)

    seqids = dvs_data_store.get_seqids_from_store(seqfile)
    if verbose:
        dvs_util.print_colour(f"Using random seed: {seed}", "blue")

    rng = numpy.random.default_rng(seed=seed)
    rng.shuffle(seqids)
    if limit is not None:
        seqids = seqids[:limit]

    app = dvs_cluster.dvs_cli_par_ctree(
        seq_store=seqfile,
        limit=limit,
        k=k,
        sketch_size=sketch_size,
        moltype=moltype,
        distance_mode=distance,
        mash_canonical_kmers=canonical_kmers,
        max_workers=numprocs,
        parallel=numprocs > 1,
        show_progress=not hide_progress,
    )
    tree = app(seqids)  # pylint: disable=not-callable
    if not tree:
        dvs_util.print_colour(tree, "red")
        sys.exit(1)

    tree.write(outpath)
    dvs_util.print_colour(f"Newick tree written to '{outpath!s}'", "green")


if __name__ == "__main__":
    main()
