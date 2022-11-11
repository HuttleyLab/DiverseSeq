from itertools import product

from cogent3 import make_table, make_unaligned_seqs
from cogent3.app import typing as c3_types
from cogent3.app.composable import NotCompleted, define_app
from cogent3.util import parallel as PAR
from numpy import array
from numpy.random import choice, shuffle
from rich.progress import track
from tqdm.notebook import tqdm

from divergent.record import SeqRecord, seq_to_record
from divergent.records import max_divergent


try:
    from wakepy import set_keepawake, unset_keepawake
except (ImportError, NotImplementedError):
    # may not be installed, or on linux where this library doesn't work
    def _do_nothing_func(*args, **kwargs):
        ...

    set_keepawake, unset_keepawake = _do_nothing_func, _do_nothing_func


POOL = {"a": "ACGGGGGT", "b": "ACCCCCGT", "c": "AAAAACGT", "d": "ACGTTTTT"}


@define_app
class make_sample:
    def __init__(self, pool_sizes: dict, seq_len: int):
        self.pool_sizes = pool_sizes
        self.seq_len = seq_len

    def main(self, num: int) -> c3_types.UnalignedSeqsType:
        seqs = {}
        for pool, num_seqs in self.pool_sizes.items():
            seqs |= seqs_from_pool(pool, num_seqs, self.seq_len)
        return make_unaligned_seqs(seqs, moltype="dna", source=f"rep-{num}")


@define_app
class seqcoll_to_records:
    def __init__(self, k: int):
        self.s2r = seq_to_record(k=k, moltype="dna")

    def main(self, seqs: c3_types.UnalignedSeqsType) -> list[SeqRecord]:
        records = [self.s2r(s) for s in seqs.seqs]
        shuffle(records)
        return records


@define_app
class true_positive:
    def __init__(
        self,
        expected: set[str],
        stat: str,
        label_2_pool: callable = lambda x: x.split("-")[0],
        size: int = 2,
    ) -> None:
        self.expected = expected
        self.label2pool = label_2_pool
        self.size = size
        self.stat = stat

    def main(self, records: list[SeqRecord]) -> bool:
        result = max_divergent(
            records, min_size=self.size, stat=self.stat, verbose=False, max_set=True
        )
        if len(result.record_names) != len(self.expected):
            return NotCompleted(
                "FAIL",
                self,
                f"number of records {len(result.record_names)} != {len(self.expected)}",
            )

        found_pools = {self.label2pool(n) for n in result.record_names}
        if self.expected != found_pools:
            return NotCompleted(
                "FAIL", self, f"found pool {found_pools} != {self.expected}"
            )
        return True


def seqs_from_pool(pool: str, num_seqs: int, seq_len: int) -> dict:
    return {
        f"{pool}-{i}": "".join(choice(list(POOL[pool]), size=seq_len))
        for i in range(num_seqs)
    }


def do_run(pool, num_reps, seq_len, k=3, stat="mean_jsd"):
    make_seqs = make_sample(pool, seq_len=seq_len)
    s2r = seqcoll_to_records(k=k)
    finds_true = true_positive(set(POOL), stat=stat, size=2)
    app = make_seqs + s2r + finds_true
    return list(map(app, range(num_reps)))


@define_app
class eval_condition:
    def __init__(self, seq_len, num_reps, k, repeats, pools) -> None:
        self.num_reps = num_reps
        self.k = k
        self.repeats = repeats
        self.pools = pools
        self.seq_len = seq_len

    def main(self, settings: tuple[str]) -> c3_types.TabularType:
        title, stat = settings
        pool = self.pools[title]
        num_correct = []
        for _ in range(self.repeats):
            r = do_run(
                pool=pool,
                num_reps=self.num_reps,
                seq_len=self.seq_len,
                k=self.k,
                stat=stat,
            )
            num_correct.append(self.num_reps - sum(r))

        num_correct = array(num_correct)
        mean = num_correct.mean()
        stdev = num_correct.std(ddof=1)
        return make_table(
            header=[
                "pop_size",
                "repeats",
                "seq_len",
                "stat",
                "mean(correct)",
                "stdv(correct)",
            ],
            data=[[self.num_reps, self.repeats, self.seq_len, stat, mean, stdev]],
            title=title,
        )


def main():
    set_keepawake(keep_screen_awake=False)

    BALANCED = dict(a=25, b=25, c=25, d=25)
    IMBALANCED = dict(a=1, b=49, c=25, d=25)
    config = dict(
        seq_len=200,
        num_reps=50,
        k=3,
        repeats=3,
        pools=dict(balanced=BALANCED, imbalanced=IMBALANCED),
    )
    app = eval_condition(**config)
    stats = "mean_delta_jsd", "total_jsd", "mean_jsd"
    pools = "balanced", "imbalanced"

    result_tables = []
    settings = list(product(pools, stats))
    series = PAR.as_completed(app, settings, max_workers=6)
    for t in track(series, total=len(settings), description="200bp sim"):
        if not t:
            print(t)
        result_tables.append(t)

    config["seq_len"] = 1000
    app = eval_condition(**config)
    stats = "mean_delta_jsd", "mean_jsd"
    pools = "balanced", "imbalanced"
    settings = list(product(pools, stats))
    series = PAR.as_completed(app, settings, max_workers=6)
    for t in track(series, total=len(settings), description=f"{config['seq_len']}bp sim"):
        if not t:
            print(t)
        result_tables.append(t)

    app = eval_condition(**config)
    table = result_tables[0].appended("Pool", result_tables[1:])
    table.write("synthetic_knowns_summary.tsv")
    print(table)

    unset_keepawake()


if __name__ == "__main__":
    main()
