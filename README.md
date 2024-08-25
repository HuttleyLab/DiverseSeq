[![CI](https://github.com/HuttleyLab/Divergent/actions/workflows/ci.yml/badge.svg)](https://github.com/HuttleyLab/Divergent/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/HuttleyLab/Divergent/badge.svg?branch=main)](https://coveralls.io/github/HuttleyLab/Divergent?branch=main)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/ef3010ea162f47a2a5a44e0f3f6ed1f0)](https://app.codacy.com/gh/HuttleyLab/Divergent/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

# Divergent

## Identifying the sequences that maximise Jensen-Shannon

## The available commands

<!-- [[[cog
import cog
from divergent.cli import main
from click.testing import CliRunner
runner = CliRunner()
result = runner.invoke(main, ["--help"])
help = result.output.replace("Usage: main", "Usage: dvgt")
cog.out(
    "```\n{}\n```".format(help)
)
]]] -->
```
Usage: dvgt [OPTIONS] COMMAND [ARGS]...

  dvgt -- alignment free detection of most divergent sequences using JSD

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  prep   Writes processed sequences to an outpath ending with .dvgtseqs file.
  max    Identify the seqs that maximise average delta JSD
  nmost  Identify n seqs that maximise average delta JSD

```
<!-- [[[end]]] -->

### `dvgt prep`: Preparing the sequence data

Convert sequence data into a more efficient format for the diversity assessment. This must be done before running either the `nmost` or `max` commands.

#### Usage:

<!-- [[[cog
import cog
from divergent.cli import main
from click.testing import CliRunner
runner = CliRunner()
result = runner.invoke(main, ["prep", "--help"])
help = result.output.replace("Usage: main", "Usage: dvgt")
cog.out(
    "```\n{}\n```".format(help)
)
]]] -->
```
Usage: dvgt prep [OPTIONS]

  Writes processed sequences to an outpath ending with .dvgtseqs file.

Options:
  -s, --seqdir PATH        directory containing sequence files  [required]
  -sf, --suffix TEXT       sequence file suffix  [default: fa]
  -o, --outpath PATH       location to write processed seqs  [required]
  -np, --numprocs INTEGER  number of processes  [default: 1]
  -F, --force_overwrite    Overwrite existing file if it exists
  -m, --moltype [dna|rna]  Molecular type of sequences, defaults to DNA
                           [default: dna]
  -L, --limit INTEGER      number of sequences to process
  --help                   Show this message and exit.

```
<!-- [[[end]]] -->

### `dvgt nmost`: Select the n-most divergent sequences

We recommend using `nmost` for large datasets.

> **Note**
> A fuller explanation is coming soon!

#### Command line usage:

<!-- [[[cog
import cog
from divergent.cli import main
from click.testing import CliRunner
runner = CliRunner()
result = runner.invoke(main, ["nmost", "--help"])
help = result.output.replace("Usage: main", "Usage: dvgt")
cog.out(
    "```\n{}\n```".format(help)
)
]]] -->
```
Usage: dvgt nmost [OPTIONS]

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
  --help                   Show this message and exit.

```
<!-- [[[end]]] -->

#### As a cogent3 plugin:

The `dvgt_select_nmost` is also available as a [cogent3 app](https://cogent3.org/doc/app/index.html). The result of using `cogent3.app_help("dvgt_select_nmost")` is shown below.

<!-- [[[cog
import cog
import contextlib
import io


from cogent3 import app_help

buffer = io.StringIO()

with contextlib.redirect_stdout(buffer):
  app_help("dvgt_select_nmost")
cog.out(
    "```\n{}\n```".format(buffer.getvalue())
)
]]] -->
```
Overview
--------
selects the n-most divergent seqs from a sequence collection

Options for making the app
--------------------------
dvgt_select_nmost_app = get_app(
    'dvgt_select_nmost',
    n=3,
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
Alignment, SequenceCollection, ArrayAlignment

Output type
-----------
Alignment, SequenceCollection, ArrayAlignment

```
<!-- [[[end]]] -->

### `dvgt max`: Maximise average delta JSD

The result of the `max` command is typically a set that are modestly more divergent than that fron `nmost`.

> **Note**
> A fuller explanation is coming soon!

#### Command line usage:

<!-- [[[cog
import cog
from divergent.cli import main
from click.testing import CliRunner
runner = CliRunner()
result = runner.invoke(main, ["max", "--help"])
help = result.output.replace("Usage: main", "Usage: dvgt")
cog.out(
    "```\n{}\n```".format(help)
)
]]] -->
```
Usage: dvgt max [OPTIONS]

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
  --help                   Show this message and exit.

```
<!-- [[[end]]] -->


#### As a cogent3 plugin:

The `dvgt_select_nmost` is also available as a [cogent3 app](https://cogent3.org/doc/app/index.html). The result of using `cogent3.app_help("dvgt_select_nmost")` is shown below.

<!-- [[[cog
import cog
import contextlib
import io


from cogent3 import app_help

buffer = io.StringIO()

with contextlib.redirect_stdout(buffer):
  app_help("dvgt_select_max")
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
dvgt_select_max_app = get_app(
    'dvgt_select_max',
    min_size=3,
    max_size=10,
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
    the maximum size if the divergent set
stat
    statistic for maximising the set, either mean_delta_jsd, mean_jsd, total_jsd
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
Alignment, SequenceCollection, ArrayAlignment

Output type
-----------
Alignment, SequenceCollection, ArrayAlignment

```
<!-- [[[end]]] -->
