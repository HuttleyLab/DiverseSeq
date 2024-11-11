import multiprocessing
import os
import time
from pathlib import Path

from cogent3 import load_aligned_seqs, make_aligned_seqs
from diverse_seq.cluster import dvs_par_ctree

MAMMALS_PATH = Path("data/mammals-aligned")
OUT_FILE = Path("out/results.tsv")


KS = list(range(4, 21))
SKETCH_SIZES = [
    100,
    250,
    500,
    750,
    1000,
    2500,
    5000,
    7500,
    10000,
    25000,
    50000,
    75000,
    100000,
]


def load_alignment(directory: Path):
    fasta_files = filter(
        lambda file_name: file_name.endswith(".fa"), os.listdir(directory)
    )

    seqs = {}

    for file in fasta_files:
        aln = load_aligned_seqs(directory / file, moltype="dna")
        aln_dict = aln.to_dict()
        for seq_name in aln_dict:
            if seq_name in seqs:
                seqs[seq_name] = seqs[seq_name] + "-" + aln_dict[seq_name]
            else:
                seqs[seq_name] = aln_dict[seq_name]

    aln = make_aligned_seqs(seqs, moltype="dna")
    return aln


def get_completed() -> set[tuple[int, int]]:
    completed = set()
    with OUT_FILE.open() as f:
        for line in f:
            k, ss = line.strip().split("\t")[:2]
            k = int(k)
            ss = int(ss)
            completed.add((k, ss))
    return completed


def run_experiment(aln, k, sketch_size):
    print(f"Doing {k=} {sketch_size=}")
    ctree_app = dvs_par_ctree(
        k=k,
        sketch_size=sketch_size,
        parallel=True,
    )
    start_time = time.time()
    ctree = ctree_app(aln)
    total_time = time.time() - start_time

    to_write = [
        str(k),
        str(sketch_size),
        str(multiprocessing.cpu_count()),
        str(total_time),
        str(ctree),
    ]
    with OUT_FILE.open("a") as f:
        f.write("\t".join(to_write) + "\n")


def run_experiments():
    completed = get_completed()
    aln = load_alignment(MAMMALS_PATH)
    for k in KS:
        for sketch_size in SKETCH_SIZES:
            if (k, sketch_size) in completed:
                print(f"Skipping {k=} {sketch_size=}")
                continue
            run_experiment(aln, k, sketch_size)


if __name__ == "__main__":
    run_experiments()
