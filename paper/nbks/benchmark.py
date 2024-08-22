import os
import shutil
import tempfile
import time
from pathlib import Path

import click
from click.testing import CliRunner
from cogent3 import make_table
from rich import progress as rich_progress

from divergent import cli as dvgt_cli
from divergent import util as dvgt_util

RUNNER = CliRunner()


class TempWorkingDir:
    def __enter__(self):
        self.original_directory = os.getcwd()
        self.temp_directory = tempfile.mkdtemp()
        os.chdir(self.temp_directory)
        return Path(self.temp_directory)

    def __exit__(self, exc_type, exc_value, traceback):
        os.chdir(self.original_directory)
        shutil.rmtree(self.temp_directory)


class TimeIt:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time

    def get_elapsed_time(self):
        return self.elapsed_time


_click_command_opts = dict(
    no_args_is_help=True,
    context_settings={"show_default": True},
)


def run_prep(temp_dir, seqdir, dvgt_file, suffix, num_seqs):
    # run the prep command
    args = f"-s {seqdir} -o {dvgt_file} -sf {suffix} -L {num_seqs} -np 1".split()
    with TimeIt() as timer:
        r = RUNNER.invoke(dvgt_cli.prep, args, catch_exceptions=False)
        assert r.exit_code == 0, r.output
    return timer.get_elapsed_time()


def run_max(seqfile, outpath, k):
    args = f"-s {seqfile} -o {outpath} --min_size 5 --max_size 10 -k {k} -np 1".split()
    with TimeIt() as timer:
        r = RUNNER.invoke(dvgt_cli.max, args, catch_exceptions=False)
        assert r.exit_code == 0, r.output
    return timer.get_elapsed_time()


@click.command(**_click_command_opts)
@click.argument("seqdir", type=Path)
@click.argument("outpath", type=Path)
@click.option("-s", "--suffix", type=str, default="gb")
def run(seqdir, suffix, outpath):
    seqdir = seqdir.absolute()
    reps = [1, 2, 3]
    kmer_sizes = [2, 3, 4, 5, 6, 7, 8]
    num_seqs = [50, 100, 150, 200]
    results = []
    with dvgt_util.keep_running():
        with rich_progress.Progress(
            rich_progress.TextColumn("[progress.description]{task.description}"),
            rich_progress.BarColumn(),
            rich_progress.TaskProgressColumn(),
            rich_progress.TimeRemainingColumn(),
            rich_progress.TimeElapsedColumn(),
        ) as progress:
            repeats = progress.add_task("Doing reps", total=len(reps))
            for _ in reps:
                seqnum = progress.add_task("Doing num seqs", total=len(num_seqs))
                for num in num_seqs:
                    with TempWorkingDir() as temp_dir:
                        dvgt_file = temp_dir / f"dvgt_L{num}.dvgtseqs"
                        elapsed_time = run_prep(
                            temp_dir,
                            seqdir,
                            dvgt_file,
                            suffix,
                            num,
                        )
                        results.append(("prep", num, None, elapsed_time))

                        kmers = progress.add_task(
                            "Doing kmers",
                            total=len(kmer_sizes),
                            transient=True,
                        )
                        for k in kmer_sizes:
                            kout = temp_dir / f"selected-{k}.tsv"
                            elapsed_time = run_max(dvgt_file, kout, k)
                            results.append(("max", num, k, elapsed_time))
                            progress.update(kmers, advance=1, refresh=True)
                        progress.remove_task(kmers)

                        progress.update(seqnum, advance=1, refresh=True)
                progress.remove_task(seqnum)
                progress.update(repeats, advance=1, refresh=True)

    table = make_table(header=("command", "numseqs", "k", "time(s)"), data=results)
    print(table)
    table.write(outpath)


if __name__ == "__main__":
    run()
