from pathlib import Path
import cogent3
import itertools
from diverse_seq import cli as dvs_cli
from diverse_seq import util as dvs_util
from click.testing import CliRunner
from benchmark import TempWorkingDir, TimeIt
from rich import progress as rich_progress

RUNNER = CliRunner()


def main():
    in_file = Path("../data/soil.dvseqs").absolute()
    assert in_file.exists()
    reps = [1, 2, 3]
    kmer_sizes = list(range(10, 17, 2))
    num_seqs = [50, 100, 150, 200]

    results = {"k": [], "numseqs": [], "time(s)": []}
    with dvs_util.keep_running():
        with rich_progress.Progress(
            rich_progress.TextColumn("[progress.description]{task.description}"),
            rich_progress.BarColumn(),
            rich_progress.TaskProgressColumn(),
            rich_progress.TimeRemainingColumn(),
            rich_progress.TimeElapsedColumn(),
        ) as progress:
            repeats = progress.add_task("Doing reps", total=len(reps))
            for _ in reps:
                combo = progress.add_task(
                    "Doing k x num_seq", total=len(num_seqs) * len(kmer_sizes)
                )
                for k, num_seq in itertools.product(kmer_sizes, num_seqs):
                    with TempWorkingDir() as temp_dir:
                        out_tree_file = temp_dir / f"{in_file.name}.tre"
                        args = f"-s {in_file} -o {out_tree_file} -k {k} --distance mash --sketch-size 3000 -hp --limit {num_seq}".split()
                        with TimeIt() as timer:
                            r = RUNNER.invoke(
                                dvs_cli.ctree, args, catch_exceptions=False
                            )
                            assert r.exit_code == 0, r.output
                        results["time(s)"].append(timer.get_elapsed_time())
                        results["k"].append(k)
                        results["numseqs"].append(num_seq)
                    progress.update(combo, advance=1, refresh=True)

                progress.remove_task(combo)
                progress.update(repeats, advance=1, refresh=True)

    table = cogent3.make_table(data=results)
    outpath = "benchmark-ctree.tsv"
    table.write(outpath)
    dvs_util.print_colour(f"Wrote {outpath}!", "green")


if __name__ == "__main__":
    main()
