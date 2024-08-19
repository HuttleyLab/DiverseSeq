from itertools import product
from pathlib import Path
from statistics import mean, stdev

import click
from cogent3 import make_table
from cogent3.app import io
from cogent3.app import typing as c3_types
from cogent3.app.composable import define_app
from cogent3.util import parallel as PAR
from numpy import isnan, array
from numpy.random import choice
from rich.progress import track
from scipy.special import binom

from divergent import records as dvgt_records
from divergent import util as dvgt_utils
from divergent.records import max_divergent


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


class calc_mean:
    def __init__(self, aln):
        self.dists = aln.distance_matrix(calc="paralinear", drop_invalid=False)
        self.num = len(aln.names)

    def __call__(self, names):
        dists = self.dists.take_dists(names)
        values = [v for v in dists.to_dict().values() if not isnan(v)]
        stats = dvgt_utils.summary_stats(array(values))
        return stats.mean


@define_app
class assess_distances:
    def __init__(
        self,
        *,
        k,
        min_size,
        max_size,
        stat,
        max_set,
        dist_size: int = 1000,
    ) -> None:
        self._s2k = dvgt_records.seq_to_seqarray(
            moltype="dna",
        ) + dvgt_records.seqarray_to_kmerseq(
            k=k,
            moltype="dna",
        )
        self.dist_size = dist_size
        self.min_size = min_size
        self.max_size = max_size
        self.stat = stat
        self.max_set = max_set

    def main(self, aln: c3_types.AlignedSeqsType) -> dict:
        mean_dist = calc_mean(aln)
        records = [self._s2k(s) for s in aln.degap().seqs]
        divergent = max_divergent(
            records,
            min_size=self.min_size,
            max_size=self.max_size,
            stat=self.stat,
            max_set=self.max_set,
        )
        num = divergent.size
        combinations = get_combinations(aln.names, num, self.dist_size)

        stats = {"observed": [True], "mean(dist)": [mean_dist(divergent.record_names)]}
        for names in combinations:
            stats["observed"].append(False)
            stats["mean(dist)"].append(mean_dist(names))

        dists = stats["mean(dist)"]
        obs = dists[0]
        gt = sum(v >= obs for v in dists[1:])
        return {
            "stats": stats,
            "divergent": list(divergent.record_names),
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
    with dvgt_utils.keep_running():
        loader = io.load_aligned(moltype="dna")
        make_distn = assess_distances(
            dist_size=dist_size,
            k=k,
            min_size=min_size,
            max_size=10,
            stat=stat,
            max_set=max_set,
        )

        app = loader + make_distn
        dstore = io.open_data_store(aln_dir, suffix="fa", limit=limit)

        results = []
        series = PAR.as_completed(app, dstore, max_workers=6)
        for result in series:
            if not result:
                continue
            result["pval"] = result["num_gt"] / result["dist_size"]
            result["num_divergent"] = len(result.pop("divergent"))
            result.pop("stats")
            results.append(result)

    return results


_click_command_opts = dict(
    no_args_is_help=True,
    context_settings={"show_default": True},
)


@click.command(**_click_command_opts)
@click.argument("seqdir", type=Path)
def main(seqdir):
    settings = list(
        product(
            range(2, 11),
            ("stdev", "cov"),
            (True, False),
        ),
    )

    order = "k", "stat", "max_set"
    rows = []
    for config in track(settings, description="Working on setting..."):
        setting = dict(zip(order, config, strict=False))
        result = run(seqdir, min_size=5, dist_size=1000, **setting)
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
    table.write("jsd_v_dist.tsv")


if __name__ == "__main__":
    main()
