# `diverse-seq` provides alignment-free algorithms to facilitate phylogenetic workflows

`diverse-seq` implements computationally efficient alignment-free algorithms that enable efficient prototyping for phylogenetic workflows. It can accelerate parameter selection searches for sequence alignment and phylogeny estimation by identifying a subset of sequences that are representative of the diversity in a collection.

You can read more about the methods implemented in `diverse-seq` in the preprint [here](https://biorxiv.org/cgi/content/short/2024.11.10.622877v1).

### Installation

`diverse-seq` can be installed from PyPI as follows

```
pip install diverse-seq
```

> **NOTE**
> If you experience any errors during installation, we recommend using [uv pip](https://docs.astral.sh/uv/). This command provides much better error messages than the standard `pip` command. If you cannot resolve the installation problem, please [open an issue](https://github.com/HuttleyLab/DiverseSeq/issues).

#### The `extra` install option

To get the additional components for working within Jupyter notebooks, or just to reproduce all the examples in the documentation, do the install as follows

```
pip install "diverse-seq[extra]"
```

> **Note**
> The first usage of `diverse-seq` is slow because both `cogent3` and `diverse-seq` are compiling some functions. This is a one-time cost.

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
