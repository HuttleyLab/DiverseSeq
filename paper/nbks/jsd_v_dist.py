from itertools import product
from pathlib import Path
from statistics import mean

import click
from cogent3 import make_table
from cogent3.app import io
from cogent3.app import typing as c3_types
from cogent3.app.composable import define_app
from cogent3.util import parallel as PAR
from numpy.random import choice
from rich.progress import track
from scipy.special import binom

from divergent.record import seq_to_record
from divergent.records import max_divergent

try:
    from wakepy import set_keepawake, unset_keepawake
except (ImportError, NotImplementedError):
    # may not be installed, or on linux where this library doesn't work
    def _do_nothing_func(*args, **kwargs): ...

    set_keepawake, unset_keepawake = _do_nothing_func, _do_nothing_func


def get_combinations(all_vals, choose, number):
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


@define_app
class divergent_seqs:
    """the divergent seqs"""

    def __init__(
        self,
        k: int = 3,
        min_size: int = 2,
        max_size: int = 10,
        stat: str = "total_jsd",
        max_set: bool = True,
    ) -> None:
        self.min_size = min_size
        self.max_size = max_size
        self.stat = stat
        self.seqs_to_records = seq_to_record(k=k, moltype="dna")
        self.max_set = max_set

    def main(self, aln: c3_types.AlignedSeqsType) -> list[str]:
        records = [self.seqs_to_records(s) for s in aln.degap().seqs]
        result = max_divergent(
            records,
            min_size=self.min_size,
            max_size=self.max_size,
            stat=self.stat,
            verbose=False,
            max_set=self.max_set,
        )
        return result.record_names


class mean_dist:
    def __init__(self, aln):
        self.dists = aln.distance_matrix(calc="paralinear", drop_invalid=True)

    def __call__(self, names):
        dists = self.dists.take_dists(names)
        return mean(dists.to_dict().values())


@define_app
class assess_distances:
    def __init__(self, dvgnt, dist_size: int = 1000) -> None:
        self.dvgnt = dvgnt
        self.dist_size = dist_size

    def main(self, aln: c3_types.AlignedSeqsType) -> dict:
        get_mean = mean_dist(aln)
        divergent = self.dvgnt(aln)
        num = len(divergent)
        combinations = get_combinations(aln.names, num, self.dist_size)

        stats = {"observed": [True], "mean(dist)": [get_mean(divergent)]}
        for names in combinations:
            stats["observed"].append(False)
            stats["mean(dist)"].append(get_mean(names))

        dists = stats["mean(dist)"]
        obs = dists[0]
        gt = sum(v >= obs for v in dists[1:])
        return {
            "stats": stats,
            "divergent": list(divergent),
            "dist_size": len(combinations),
            "num_gt": gt,
            "source": aln.info.source,
        }


def run(
    aln_dir: Path,
    stat,
    max_set,
    k: int = 4,
    min_size: int = 7,
    dist_size: int = 1000,
    limit=None,
):
    set_keepawake(keep_screen_awake=False)

    loader = io.load_aligned(moltype="dna")
    dvgnt = divergent_seqs(k=k, min_size=min_size, stat=stat, max_set=max_set)
    make_distn = assess_distances(dvgnt, dist_size=dist_size)

    app = loader + make_distn
    dstore = io.get_data_store(aln_dir, suffix="fa", limit=limit)

    results = []
    pvals = []
    series = PAR.as_completed(app, dstore, max_workers=6)
    for result in series:
        if not result:
            continue
        results.append(result)
        pvals.append(result["num_gt"] / result["dist_size"])

    unset_keepawake()

    return results, pvals


_click_command_opts = dict(
    no_args_is_help=True,
    context_settings={"show_default": True},
)


@click.command(**_click_command_opts)
@click.argument("seqdir", type=Path)
def main(seqdir):
    settings = list(
        product(
            range(2, 8),
            ("mean_jsd", "mean_delta_jsd", "total_jsd"),
            (True, False),
        ),
    )

    order = "k", "stat", "max_set"
    rows = []
    for config in track(settings, description="Working on setting..."):
        setting = dict(zip(order, config, strict=False))
        result, pvals = run(seqdir, min_size=5, dist_size=1000, **setting)
        size = len(result["divergent"])
        rows.append(list(config) + [size] + [sum(p < 0.1 for p in pvals), len(pvals)])

    table = make_table(header=list(order) + ["size", "p-value<0.1", "num"], data=rows)
    print(table)
    table.write("jsd_v_dist.tsv")


if __name__ == "__main__":
    main()
