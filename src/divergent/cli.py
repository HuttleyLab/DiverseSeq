import json
import time

from pathlib import Path

import blosc2
import click

from cogent3 import make_seq, open_
from cogent3.app import composable
from cogent3.app import io as io_app
from cogent3.app import typing as c3_types
from cogent3.parse.fasta import MinimalFastaParser
from cogent3.util import parallel as PAR
from rich.progress import track
from scitrack import CachingLogger

from .record import seq_to_kmer_counts, sparse_vector


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


def _label_func(label):
    return label.split()[0]


@composable.define_app(app_type="loader")
class seq_from_fasta:
    def __init__(self, label_func=_label_func, max_length=None) -> None:
        self.label_func = label_func
        self.max_length = max_length

    def main(self, path: c3_types.IdentifierType) -> c3_types.SeqType:
        with open_(path) as infile:
            data = list(iter(MinimalFastaParser(infile.read().splitlines())))
            assert len(data) == 1
            name, seq = data[0]
            name = self.label_func(name)

        if self.max_length:
            seq = seq[: self.max_length]

        seq = make_seq(seq, name=name, moltype="dna")
        seq.source = str(path)
        return seq


@composable.define_app
class seq_to_kmers:
    def __init__(self, k: int):
        self.k = k

    def main(self, seq: c3_types.SeqType) -> tuple[bytes, str]:
        sv = seq_to_kmer_counts(seq=seq, moltype="dna", k=self.k)
        return (
            blosc2.compress(
                json.dumps(sv.to_rich_dict()).encode("utf8"),
                shuffle=blosc2.Filter.BITSHUFFLE,
            ),
            sv.source,
        )


def _make_outpath(outdir, path, k):
    return outdir / f"{path.stem.split('.')[0]}-{k}-mer.json.blosc2"


@main.command(no_args_is_help=True)
@click.option(
    "-i",
    "--indir",
    required=True,
    type=Path,
    help="directory containing fasta formatted sequence files",
)
@click.option(
    "-o",
    "--outdir",
    type=Path,
    help="directory to write compressed json",
)
@click.option("-k", type=int, default=7)
@click.option("-p", "--parallel", is_flag=True, help="run in parallel")
@click.option("-L", "--limit", type=int, help="number of records to process")
@_verbose
def oneoff(indir, outdir, k, parallel, limit, verbose):
    from wakepy import set_keepawake, unset_keepawake

    outdir.mkdir(parents=True, exist_ok=True)

    paths = [str(p) for p in indir.glob("**/*.fa*")]
    if limit:
        paths = paths[:limit]

    app = seq_from_fasta() + seq_to_kmers(k=k)

    not_done = []
    for p in paths:
        op = _make_outpath(outdir, Path(p), k)
        if op.exists():
            continue
        not_done.append(p)

    paths = not_done

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
        outpath = outdir / f"{path.stem.split('.')[0]}-{k}-mer.json.blosc2"
        outpath.write_bytes(b)

    unset_keepawake()
    # LOGGER.log_args()
    # LOGGER.log_versions("numpy")
    # LOGGER.input_file(infile)
    #
    # LOGGER.log_file_path = outpath / "dvg-n.log"


if __name__ == "__main__":
    main()
