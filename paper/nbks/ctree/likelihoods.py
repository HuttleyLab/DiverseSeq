import os
from pathlib import Path
from piqtree2 import fit_tree

from cogent3 import load_aligned_seqs, make_aligned_seqs, make_tree

MAMMALS_PATH = Path("data/mammals-aligned")

CONCAT_PATH = Path("data/concat/concat.fasta")
RESULTS_FILE = Path("out/results.tsv")
IQ_FILE = Path("out/iqtree.tsv")
OUT_FILE = Path("out/results_with_ls.tsv")
OUT_IQ_FILE = Path("out/iqtree_with_ls.tsv")


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


def do_likelihoods():
    if OUT_FILE.exists() or OUT_IQ_FILE.exists():
        raise FileExistsError("Might overwrite file")
    with OUT_FILE.open("w") as f:
        f.write("")
    with OUT_IQ_FILE.open("w") as f:
        f.write("")

    print("Loading")
    aln = load_alignment(MAMMALS_PATH)

    with RESULTS_FILE.open() as f:
        for line in f:
            k, ss, cpus, time, tree = line.strip().split("\t")
            print(f"Doing {k} {ss}")
            tree = make_tree(tree).unrooted()
            tree = fit_tree(aln, tree, "GTR", rand_seed=1)
            likelihood = tree.params["lnL"]
            with OUT_FILE.open("a") as f:
                f.write(
                    "\t".join([k, ss, cpus, time, str(likelihood), str(tree)]) + "\n"
                )
    with IQ_FILE.open() as f:
        for line in f:
            time, tree = line.strip().split("\t")
            print("Doing iq")
            tree = make_tree(tree).unrooted()
            tree = fit_tree(aln, tree, "GTR", rand_seed=1)
            likelihood = tree.params["lnL"]
            with OUT_IQ_FILE.open("a") as f:
                f.write("\t".join([time, str(likelihood), str(tree)]) + "\n")


if __name__ == "__main__":
    do_likelihoods()
