from warnings import filterwarnings

# silence an annoying warning from loky
filterwarnings("ignore", message="A worker stopped while some jobs.*")


import cogent3

in_dstore = cogent3.open_data_store("../data/mammal_orths_31", suffix="fa.gz", mode="r")
out_dstore = cogent3.open_data_store(
    "../data/mammal_orths_31_aligned", suffix="fa", mode="w"
)
loader = cogent3.get_app("load_unaligned", moltype="dna")
translatable = cogent3.get_app("select_translatable", frame=1)
mlength = cogent3.get_app("min_length", length=600)
aligner = cogent3.get_app("progressive_align", "codon")
writer = cogent3.get_app("write_seqs", data_store=out_dstore)
app = loader + translatable + mlength + aligner + writer

_ = app.apply_to(in_dstore, parallel=True, show_progress=True)

print(out_dstore.describe)
print(out_dstore.summary_not_completed)


print(f"Wrote {out_dstore.source}")
