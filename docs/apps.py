# %% [markdown]
#
# We provide programmatic access to `diverse_seq` functions as [cogent3 apps](https://cogent3.org/doc/app/index.html). The `dvs` apps mirror the capabilities of their command line counterparts with two key differences: the input and outputs. There is no data transformation step. Just use `cogent3` to [load the sequence collection](https://cogent3.org/doc/cookbook/loading_sequences.html) (aligned or otherwise) and pass it to an app instance. The `dvs_nmost` and `dvs_max` will identify the sequences to keep and return the same input data type that contains just those sequences.

# ## What apps are available?
# We use the `cogent3` capabilities for displaying the installed apps and getting help on them.

# %%
import cogent3

cogent3.available_apps("dvs")

# %% [markdown]
# ## Getting help on an app
# We do this for the `dvs_nmost` app only.

# %%
cogent3.app_help("dvs_nmost")

# %% [markdown]
# ## Using dvs_nmost
# Load the sample data as a `cogent3` `SequenceCollection` of unaligned sequences.
# > **Note**
# > The apps can use either an alignment or a `cogent3` sequence collection (of unaligned sequences).

# %%
import diverse_seq

seqs = diverse_seq.load_sample_data()
seqs

# %% [markdown]
# When we create an app, we can see all the parameter settings (including defaults) as follows.
# %%
nmost = cogent3.get_app("dvs_nmost", n=10, seed=123)
nmost

# %%
result = nmost(seqs)
result

# %% [markdown]
# ## Using dvs_max

# %%
dvs_max = cogent3.get_app("dvs_max", max_size=10, seed=123)
dvs_max

# %%
result = dvs_max(seqs)
result

# %% [markdown]
# ## Using dvs_delta_jsd
# The `dvs_delta_jsd` app computes the delta JSD values for a single sequence against a reference set of sequences. It returns a tuple of sequence name, delta JSD value.
#
# > **Note**
# > There is no command line interface for this app.
#
# Say we have a reference group of sequences, `ref_seqs`. We want to evaluate each sequence in a set of query sequences to see what their delta JSD values are against the reference set. These values allow us, for example, to select a sequence that is highly diverged from all sequences in the reference set, or one which is very similar to a sequence in the reference set.
#
# For this example, we define `ref_seqs` as the first 10 sequences in our sample data.

# %%
ref_seqs = seqs.take_seqs(seqs.names[:10])
ref_seqs

# %% [markdown]
# We define our query group as the remaining sequences.

# %%
query_seqs = seqs.take_seqs(seqs.names[:10], negate=True)
query_seqs

# %%
dvs_djsd = cogent3.get_app("dvs_delta_jsd", seqs=ref_seqs, moltype="dna", k=8)
dvs_djsd

# %% [markdown]
# We now compute the delta JSD values for each sequence in `query_seqs` against `ref_seqs` and make a table from the results. We just display the top few records.

# %%
name_deltas = [dvs_djsd(seq) for seq in query_seqs.seqs]
table = cogent3.make_table(header=["seqname", "delta_jsd"], data=name_deltas, index_name="seqname")
table = table.sorted(reverse="delta_jsd")
table.head()

# %% [markdown]
# And the conclusion is that the `Bandicoot` (a marsupial) sequence is the most divergent from the reference set (which are all Eutherian, or "placental", mammals).

# %% [markdown]
# ## Using dvs_ctree

# %%
dvs_ctree = cogent3.get_app("dvs_ctree")
dvs_ctree

# %%
result = dvs_ctree(seqs)
result

# %%
dnd = result.get_figure()
dnd.show(renderer="svg", height=700, width=400)

# %% [markdown]
# ## dvs_par_ctree is for parallel processing
# We don't use it here, but it is best suited to a sequence collection with a lot of sequences.
