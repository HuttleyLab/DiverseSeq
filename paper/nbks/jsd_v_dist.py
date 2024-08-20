from itertools import product
from pathlib import Path
from statistics import mean, stdev

import click
from cogent3 import make_table
from cogent3.app import io
from cogent3.app import typing as c3_types
from cogent3.app.composable import define_app
from numpy import array, isnan
from numpy.random import choice
from rich.progress import track
from scipy.special import binom

from divergent import util as dvgt_utils
from divergent.records import dvgt_select_max


def get_combinations(all_vals, choose, number):
    """generates random combinations of indices that will be used to
    obtain sets of names"""
    indices = set()
    interval = range(len(all_vals))
    max_size = binom(len(all_vals), choose)
    number = min(number, max_size)
    while len(indices) < number:
        candidate = tuple(sorted(choice(interval, size=choose, replace=False)))
        if candidate in indices:
            continue
        indices.add(candidate)
    return [[all_vals[i] for i in combo] for combo in indices]


class min_dist:
    """returns minimum pairwise genetic distance from a set of names"""

    def __init__(self, dists):
        self.dists = dists
        self.num = len(dists.names)

    def __call__(self, names: list[str]) -> float:
        dists = self.dists.take_dists(names)
        values = array([v for v in dists.to_dict().values() if not isnan(v)])
        return values.min()


@define_app
class compare_sets:
    """compares the minimum genetic distance from a sets of sequences identified
    by dvgt_select_max against randomly drawn sets of the same size"""

    def __init__(
        self,
        *,
        k: int,
        min_size: int,
        max_size: int,
        stat: str,
        dist_size: int = 1000,
    ) -> None:
        self.dist_size = dist_size
        self.dvgt = dvgt_select_max(
            k=k,
            min_size=min_size,
            max_size=max_size,
            stat=stat,
        )

    def main(self, aln: c3_types.AlignedSeqsType) -> dict:
        calc_min = min_dist(aln.distance_matrix(calc="paralinear", drop_invalid=False))
        divergent = self.dvgt(aln.degap())
        num = divergent.num_seqs
        combinations = get_combinations(aln.names, num, self.dist_size)

        stats = {"observed": [True], "mean(dist)": [calc_min(divergent.names)]}
        for names in combinations:
            stats["observed"].append(False)
            stats["mean(dist)"].append(calc_min(names))

        dists = stats["mean(dist)"]
        obs = dists[0]
        # we estimate the probability the minimum genetic distance from
        # the num sequences identified by dvgt_select_max equals that of a random
        # draw of num sequences by counting the number of times the min distance from
        # the latter is greater than or equal to the diverged set.
        gt = sum(v >= obs for v in dists[1:])
        return {
            "stats": stats,
            "divergent": len(divergent.names),  # size oif the divergent set
            "dist_size": len(combinations),
            "num_gt": gt,
            "source": aln.info.source,
        }


def run(
    store_path: Path,
    stat,
    k: int = 4,
    min_size: int = 7,
    max_size: int = 10,
    dist_size: int = 1000,
    limit: int | None = None,
    serial: bool = False,
):
    """runs a specific combination of parameters"""
    with dvgt_utils.keep_running():
        loader = io.load_aligned(moltype="dna")
        make_distn = compare_sets(
            dist_size=dist_size,
            k=k,
            min_size=min_size,
            max_size=max_size,
            stat=stat,
        )

        app = loader + make_distn
        dstore = io.open_data_store(store_path, suffix="fa", limit=limit)

        results = []
        for result in app.as_completed(dstore, parallel=not serial):
            if not result:
                continue
            result = result.obj  # what get's returned is a source proxy object
            result["pval"] = result["num_gt"] / result["dist_size"]
            result["num_divergent"] = result.pop("divergent")
            result.pop("stats")
            results.append(result)

    return results


_click_command_opts = dict(
    no_args_is_help=True,
    context_settings={"show_default": True},
)


@click.command(**_click_command_opts)
@click.argument("data_path", type=Path)
@click.option(
    "-o",
    "--outpath",
    type=Path,
    default=Path("jsd_v_dist.tsv"),
    help="writes output to this file",
)
@click.option(
    "-x",
    "--max_k",
    type=int,
    default=11,
    help="upper bound for k-mer size, not inclusive",
)
@click.option("-r", "--repeats", type=int, default=3, help="repeats per condition")
@click.option("-D", "--dry_run", is_flag=True, help="don't write output")
@click.option("-S", "--serial", is_flag=True, help="run on 1 CPU")
def main(data_path, outpath, dry_run, max_k, serial, repeats):
    """measures performance of divergent using a directory containg fasta
    formatted alignment files of DNA sequences"""
    settings = (
        list(
            product(
                range(2, max_k),  # k values
                (5, 10),  # min_size values
                (10,),  # max_size value
                ("stdev", "cov"),  # stat values
            ),
        )
        * repeats
    )

    order = "k", "min_size", "max_size", "stat"
    rows = []
    for config in track(settings, description="Working on setting..."):
        setting = dict(zip(order, config, strict=False))
        result = run(data_path, dist_size=1000, serial=serial, **setting)
        sizes = [r["num_divergent"] for r in result]
        rows.append(
            list(config)
            + [mean(sizes), stdev(sizes)]
            + [100 * sum(r["pval"] < 0.05 for r in result) / len(result)],
        )

    table = make_table(
        header=list(order) + ["mean(size)", "stdev(size)", "(p-value<0.05)%"],
        data=rows,
    )
    print(table)
    if not dry_run:
        table.write(outpath)


if __name__ == "__main__":
    main()
