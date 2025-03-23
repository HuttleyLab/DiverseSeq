![PyPI - Python Version](https://img.shields.io/pypi/pyversions/diverse-seq)
[![CI](https://github.com/HuttleyLab/DiverseSeq/actions/workflows/ci.yml/badge.svg)](https://github.com/HuttleyLab/DiverseSeq/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/HuttleyLab/DiverseSeq/badge.svg?branch=main)](https://coveralls.io/github/HuttleyLab/DiverseSeq?branch=main)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/ef3010ea162f47a2a5a44e0f3f6ed1f0)](https://app.codacy.com/gh/HuttleyLab/DiverseSeq/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![CodeQL](https://github.com/HuttleyLab/DiverseSeq/actions/workflows/codeql.yml/badge.svg)](https://github.com/HuttleyLab/DiverseSeq/actions/workflows/codeql.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# `diverse-seq` provides alignment-free algorithms to facilitate phylogenetic workflows

`diverse-seq` implements computationally efficient alignment-free algorithms that enable efficient prototyping for phylogenetic workflows. It can accelerate parameter selection searches for sequence alignment and phylogeny estimation by identifying a subset of sequences that are representative of the diversity in a collection. We show that selecting representative sequences with an entropy measure of *k*-mer frequencies correspond well to sampling via conventional genetic distances. The computational performance is linear with respect to the number of sequences and can be run in parallel. Applied to a collection of 10.5k whole microbial genomes on a laptop took ~8 minutes to prepare the data and 4 minutes to select 100 representatives. `diverse-seq` can further boost the performance of phylogenetic estimation by providing a seed phylogeny that can be further refined by a more sophisticated algorithm. For ~1k whole microbial genomes on a laptop, it takes ~1.8 minutes to estimate a bifurcating tree from mash distances.

You can read more about the methods implemented in `diverse-seq` in the preprint [here](https://biorxiv.org/cgi/content/short/2024.11.10.622877v1).

The user documentation [is here](https://diverse-seq.readthedocs.io).

### Installation

We recommend installing `diverse-seq` from PyPI as follows

```
pip install "diverse-seq[extra]"
```

for the full jupyter experience.

For command line only usage, install as follows

```
pip install diverse-seq
```

> **NOTE**
> If you experience any errors during installation, we recommend using [uv pip](https://docs.astral.sh/uv/). This command provides much better error messages than the standard `pip` command. If you cannot resolve the installation problem, please open an issue on the [GitHub repository](https://github.com/HuttleyLab/DiverseSeq/issues).

#### Using `uv`

Speaking of `uv`, it provides a simplified approach to install `dvs` as a command-line only tool as

```
uv tool install diverse-seq
```

Usage in this case is then

```
uvx --from diverse-seq dvs
```

#### Dependencies

For a full listing of dependencies, see the [pyproject.toml](./pyproject.toml) file.

### The command line interface

`dvs` is the command line interface for `diverse-seq`.

<details>
    <summary>The `dvs` subcommands</summary>

<!-- [[[cog
import cog
from diverse_seq.cli import main
from click.testing import CliRunner
runner = CliRunner()
result = runner.invoke(main, [])
help = result.output.replace("Usage: main", "Usage: dvs")
cog.out(
    "```\n{}\n```".format(help)
)
]]] -->
```
Usage: dvs [OPTIONS] COMMAND [ARGS]...

  dvs -- alignment free detection of the most diverse sequences using JSD

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  demo-data  Export a demo sequence file
  prep       Writes processed sequences to a <HDF5 file>.dvseqs.
  max        Identify the seqs that maximise average delta JSD
  nmost      Identify n seqs that maximise average delta JSD
  ctree      Quickly compute a cluster tree based on kmers for a collection...

```
<!-- [[[end]]] -->

</details>

### The Python API

We make comparable capabilities available as [cogent3 apps](https://cogent3.org/doc/app/index.html). The main difference is the app instances directly operate on, and return, `cogent3` sequence collections. See [the docs](https://diverse-seq.readthedocs.io/en/latest/apps/) for demonstrations of how to use the apps.

## Project Information 

`diverse-seq` is released under the BSD-3 license. If you want to contribute to the `diverse-seq` project (and we hope you do! :innocent:) the code of conduct and other useful developer information is available on the [wiki](https://github.com/HuttleyLab/DiverseSeq/wiki).