[![CI](https://github.com/HuttleyLab/DiverseSeq/actions/workflows/ci.yml/badge.svg)](https://github.com/HuttleyLab/DiverseSeq/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/HuttleyLab/DiverseSeq/badge.svg?branch=main)](https://coveralls.io/github/HuttleyLab/DiverseSeq?branch=main)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/ef3010ea162f47a2a5a44e0f3f6ed1f0)](https://app.codacy.com/gh/HuttleyLab/DiverseSeq/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![CodeQL](https://github.com/HuttleyLab/DiverseSeq/actions/workflows/codeql.yml/badge.svg)](https://github.com/HuttleyLab/DiverseSeq/actions/workflows/codeql.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# `diverse_seq` identifies the most diverse biological sequences from a collection

`diverse_seq` provides tools for selecting a representative subset of sequences from a larger collection. It is an alignment-free method which scales linearly with the number of sequences. It identifies the subset of sequences that maximize diversity as measured using Jensen-Shannon divergence. `diverse_seq` provides a command-line tool (`dvs`) and plugins to the Cogent3 app system (prefixed by `dvs_`) allowing users to embed code in their own scripts. The command-line tools can be run in parallel.

### `dvs prep`: preparing the sequence data

Convert sequence data into a more efficient format for the diversity assessment. This must be done before running either the `nmost` or `max` commands.

<details>
    <summary>CLI options for dvs prep</summary>

<!-- [[[cog
import cog
from diverse_seq.cli import main
from click.testing import CliRunner
runner = CliRunner()
result = runner.invoke(main, ["prep", "--help"])
help = result.output.replace("Usage: main", "Usage: dvs")
cog.out(
    "```\n{}\n```".format(help)
)
]]] -->
```
Usage: dvs prep [OPTIONS]

  Writes processed sequences to a <HDF5 file>.dvseqs.

Options:
  -s, --seqdir PATH        directory containing sequence files  [required]
  -sf, --suffix TEXT       sequence file suffix  [default: fa]
  -o, --outpath PATH       write processed seqs to this filename  [required]
  -np, --numprocs INTEGER  number of processes  [default: 1]
  -F, --force_overwrite    Overwrite existing file if it exists
  -m, --moltype [dna|rna]  Molecular type of sequences, defaults to DNA
                           [default: dna]
  -L, --limit INTEGER      number of sequences to process
  -hp, --hide_progress     hide progress bars
  --help                   Show this message and exit.

```
<!-- [[[end]]] -->

</details>

### `dvs nmost`: select the n-most diverse sequences

Selects the n sequences that maximise the total JSD. We recommend using `nmost` for large datasets.

> **Note**
> A fuller explanation is coming soon!

<details>
    <summary>Options for command line dvs nmost</summary>

<!-- [[[cog
import cog
from diverse_seq.cli import main
from click.testing import CliRunner
runner = CliRunner()
result = runner.invoke(main, ["nmost", "--help"])
help = result.output.replace("Usage: main", "Usage: dvs")
cog.out(
    "```\n{}\n```".format(help)
)
]]] -->
```
Usage: dvs nmost [OPTIONS]

  Identify n seqs that maximise average delta JSD

Options:
  -s, --seqfile PATH       path to .dvtgseqs file  [required]
  -o, --outpath PATH       the input string will be cast to Path instance
  -n, --number INTEGER     number of seqs in divergent set  [required]
  -k INTEGER               k-mer size  [default: 6]
  -i, --include TEXT       seqnames to include in divergent set
  -np, --numprocs INTEGER  number of processes  [default: 1]
  -L, --limit INTEGER      number of sequences to process
  -v, --verbose            is an integer indicating number of cl occurrences
                           [default: 0]
  -hp, --hide_progress     hide progress bars
  --help                   Show this message and exit.

```
<!-- [[[end]]] -->

</details>

<details>
    <summary>Options for cogent3 app dvs_select_nmost</summary>

The `dvs nmost` is also available as the [cogent3 app](https://cogent3.org/doc/app/index.html) `dvs_select_nmost`. The result of using `cogent3.app_help("dvs_select_nmost")` is shown below.

<!-- [[[cog
import cog
import contextlib
import io


from cogent3 import app_help

buffer = io.StringIO()

with contextlib.redirect_stdout(buffer):
  app_help("dvs_select_nmost")
cog.out(
    "```\n{}\n```".format(buffer.getvalue())
)
]]] -->
```
Overview
--------
selects the n-most diverse seqs from a sequence collection

Options for making the app
--------------------------
dvs_select_nmost_app = get_app(
    'dvs_select_nmost',
    n=10,
    moltype='dna',
    include=None,
    k=6,
    seed=None,
)

Parameters
----------
n
    the number of divergent sequences
moltype
    molecular type of the sequences
k
    k-mer size
include
    sequence names to include in the final result
seed
    random number seed

Notes
-----
If called with an alignment, the ungapped sequences are used.
The order of the sequences is randomised. If include is not None, the
named sequences are added to the final result.

Input type
----------
SequenceCollection, Alignment, ArrayAlignment

Output type
-----------
SequenceCollection, Alignment, ArrayAlignment

```
<!-- [[[end]]] -->
</details>


### `dvs max`: maximise variance in the selected sequences

The result of the `max` command is typically a set that are modestly more diverse than that from `nmost`.

> **Note**
> A fuller explanation is coming soon!

<details>
    <summary>Options for command line dvs max</summary>

<!-- [[[cog
import cog
from diverse_seq.cli import main
from click.testing import CliRunner
runner = CliRunner()
result = runner.invoke(main, ["max", "--help"])
help = result.output.replace("Usage: main", "Usage: dvs")
cog.out(
    "```\n{}\n```".format(help)
)
]]] -->
```
Usage: dvs max [OPTIONS]

  Identify the seqs that maximise average delta JSD

Options:
  -s, --seqfile PATH       path to .dvtgseqs file  [required]
  -o, --outpath PATH       the input string will be cast to Path instance
  -z, --min_size INTEGER   minimum size of divergent set  [default: 7]
  -zp, --max_size INTEGER  maximum size of divergent set
  -k INTEGER               k-mer size  [default: 6]
  -st, --stat [stdev|cov]  statistic to maximise  [default: stdev]
  -i, --include TEXT       seqnames to include in divergent set
  -np, --numprocs INTEGER  number of processes  [default: 1]
  -L, --limit INTEGER      number of sequences to process
  -T, --test_run           reduce number of paths and size of query seqs
  -v, --verbose            is an integer indicating number of cl occurrences
                           [default: 0]
  -hp, --hide_progress     hide progress bars
  --help                   Show this message and exit.

```
<!-- [[[end]]] -->

</details>

<details>
<summary>Options for cogent3 app dvs_select_max</summary>

The `dvs max` is also available as the [cogent3 app](https://cogent3.org/doc/app/index.html) `dvs_select_max`. 

<!-- [[[cog
import cog
import contextlib
import io


from cogent3 import app_help

buffer = io.StringIO()

with contextlib.redirect_stdout(buffer):
  app_help("dvs_select_max")
cog.out(
    "```\n{}\n```".format(buffer.getvalue())
)
]]] -->
```
Overview
--------
selects the maximally divergent seqs from a sequence collection

Options for making the app
--------------------------
dvs_select_max_app = get_app(
    'dvs_select_max',
    min_size=5,
    max_size=30,
    stat='stdev',
    moltype='dna',
    include=None,
    k=6,
    seed=None,
)

Parameters
----------
min_size
    minimum size of the divergent set
max_size
    the maximum size of the divergent set
stat
    either stdev or cov, which represent the statistics
    std(delta_jsd) and cov(delta_jsd) respectively
moltype
    molecular type of the sequences
include
    sequence names to include in the final result
k
    k-mer size
seed
    random number seed

Notes
-----
If called with an alignment, the ungapped sequences are used.
The order of the sequences is randomised. If include is not None, the
named sequences are added to the final result.

Input type
----------
SequenceCollection, Alignment, ArrayAlignment

Output type
-----------
SequenceCollection, Alignment, ArrayAlignment

```
<!-- [[[end]]] -->
</details>