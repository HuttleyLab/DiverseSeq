# `diverse-seq` provides alignment-free algorithms to facilitate phylogenetic workflows

`diverse-seq` implements computationally efficient alignment-free algorithms that enable efficient prototyping for phylogenetic workflows. It can accelerate parameter selection searches for sequence alignment and phylogeny estimation by identifying a subset of sequences that are representative of the diversity in a collection.

You can read more about the methods implemented in `diverse-seq` in the [published article](https://joss.theoj.org/papers/10.21105/joss.07765).

??? warning
    We have rewritten a substantial part of the project in Rust. We now also use the Zarr storage format instead of HDF5. As a result, the output file from `dvs prep` now has the suffix `.dvseqsz` instead of `.dvseq`. **Old-format files are not compatible with this version.**

### Installation

`diverse-seq` can be installed from PyPI as follows

```
pip install diverse-seq
```

??? note
    If you experience any errors during installation, we recommend using [uv pip](https://docs.astral.sh/uv/). This command provides much better error messages than the standard `pip` command. If you cannot resolve the installation problem, please [open an issue](https://github.com/HuttleyLab/DiverseSeq/issues).

#### The `extra` install option

To get the additional components for working within Jupyter notebooks, or just to reproduce all the examples in the documentation, do the install as follows

```
pip install "diverse-seq[extra]"
```

#### Installation and usage with `uv`

Speaking of `uv`, it provides a simplified approach to install `dvs` as a command-line only tool as

```
uv tool install diverse-seq
```

For the examples in this documentation, you can access the `dvs` command as

```
uvx --from diverse-seq dvs
```

without having to activate a virtual environment.
