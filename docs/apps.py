# %% [markdown]
#
# We provide programmatic access to `diverse_seq` functions as [cogent3 apps](https://cogent3.org/doc/app/index.html). The `dvs` apps mirror the capabilities of their command line counterparts with two key differences: the input and outputs. There is no data transformation step. Just use `cogent3` to [load the sequence collection](https://cogent3.org/doc/cookbook/loading_sequences.html) (aligned or otherwise) and pass it to an app instance. The `dvs_nmost` and `dvs_max` will identify the sequences to keep and return the same input data type that contains just those sequences.
# > **Note**
# We now enforce using `new_type` cogent3 objects so users need to do the same (see the cogent3 ["Migration to new type core objects"](https://cogent3.org/#features-announcements) announcement).

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
