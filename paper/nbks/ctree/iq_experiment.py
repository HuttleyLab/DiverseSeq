import os
from pathlib import Path
import time
from piqtree2 import build_tree

from cogent3 import load_aligned_seqs, make_aligned_seqs

MAMMALS_PATH = Path("data/mammals-aligned")

CONCAT_PATH = Path("data/concat/concat.fasta")
OUT_FILE = Path("out/iqtree.tsv")


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


def do_iqtree():
    if OUT_FILE.exists():
        raise FileExistsError("Might overwrite file")

    print("Loading")
    aln = load_alignment(MAMMALS_PATH)
    aln.write(CONCAT_PATH)

    print("Running IQ-TREE")
    start = time.time()
    tree = build_tree(aln, "GTR", rand_seed=1)
    end = time.time()

    print("Saving results")
    with OUT_FILE.open("w") as f:
        f.write("\t".join([str(end - start), str(tree)]))


if __name__ == "__main__":
    do_iqtree()
