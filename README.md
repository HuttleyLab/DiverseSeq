[![CI](https://github.com/HuttleyLab/DiverseSeq/actions/workflows/ci.yml/badge.svg)](https://github.com/HuttleyLab/DiverseSeq/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/HuttleyLab/DiverseSeq/badge.svg?branch=main)](https://coveralls.io/github/HuttleyLab/DiverseSeq?branch=main)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/ef3010ea162f47a2a5a44e0f3f6ed1f0)](https://app.codacy.com/gh/HuttleyLab/DiverseSeq/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![CodeQL](https://github.com/HuttleyLab/DiverseSeq/actions/workflows/codeql.yml/badge.svg)](https://github.com/HuttleyLab/DiverseSeq/actions/workflows/codeql.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# `diverse_seq` provides alignment-free algorithms to facilitate phylogenetic workflows

`diverse-seq` implements computationally efficient alignment-free algorithms that enable efficient prototyping for phylogenetic workflows. It can accelerate parameter selection searches for sequence alignment and phylogeny estimation by identifying a subset of sequences that are representative of the diversity in a collection. We show that selecting representative sequences with an entropy measure of *k*-mer frequencies correspond well to sampling via conventional genetic distances. The computational performance is linear with respect to the number of sequences and can be run in parallel. Applied to a collection of 10.5k whole microbial genomes on a laptop took ~8 minutes to prepare the data and 4 minutes to select 100 representatives. `diverse-seq` can further boost the performance of phylogenetic estimation by providing a seed phylogeny that can be further refined by a more sophisticated algorithm. For ~1k whole microbial genomes on a laptop, it takes ~1.8 minutes to estimate a bifurcating tree from mash distances.

You can read more about the methods implemented in `diverse_seq` in the preprint [here](https://biorxiv.org/cgi/content/short/2024.11.10.622877v1).

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
  -m, --moltype [dna|rna]  Molecular type of sequences  [default: dna]
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
  -s, --seqfile PATH       path to .dvseqs file  [required]
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
    <summary>Options for cogent3 app dvs_nmost</summary>

The `dvs nmost` is also available as the [cogent3 app](https://cogent3.org/doc/app/index.html) `dvs_nmost`. The result of using `cogent3.app_help("dvs_nmost")` is shown below.

<!-- [[[cog
import cog
import contextlib
import io


from cogent3 import app_help

buffer = io.StringIO()

with contextlib.redirect_stdout(buffer):
  app_help("dvs_nmost")
cog.out(
    "```\n{}\n```".format(buffer.getvalue())
)
]]] -->
```
Overview
--------
select the n-most diverse seqs from a sequence collection

Options for making the app
--------------------------
dvs_nmost_app = get_app(
    'dvs_nmost',
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
ArrayAlignment, SequenceCollection, Alignment

Output type
-----------
ArrayAlignment, SequenceCollection, Alignment

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
  -s, --seqfile PATH       path to .dvseqs file  [required]
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
<summary>Options for cogent3 app dvs_max</summary>

The `dvs max` is also available as the [cogent3 app](https://cogent3.org/doc/app/index.html) `dvs_max`. 

<!-- [[[cog
import cog
import contextlib
import io


from cogent3 import app_help

buffer = io.StringIO()

with contextlib.redirect_stdout(buffer):
  app_help("dvs_max")
cog.out(
    "```\n{}\n```".format(buffer.getvalue())
)
]]] -->
```
Overview
--------
select the maximally divergent seqs from a sequence collection

Options for making the app
--------------------------
dvs_max_app = get_app(
    'dvs_max',
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
ArrayAlignment, SequenceCollection, Alignment

Output type
-----------
ArrayAlignment, SequenceCollection, Alignment

```
<!-- [[[end]]] -->
</details>

### `dvs ctree`: build a phylogeny using k-mers

The result of the `ctree` command is a newick formatted tree string without distances.

> **Note**
> A fuller explanation is coming soon!

<details>
    <summary>Options for command line dvs ctree</summary>

<!-- [[[cog
import cog
from diverse_seq.cli import main
from click.testing import CliRunner
runner = CliRunner()
result = runner.invoke(main, ["ctree", "--help"])
help = result.output.replace("Usage: main", "Usage: dvs")
cog.out(
    "```\n{}\n```".format(help)
)
]]] -->
```
Usage: dvs ctree [OPTIONS]

  Quickly compute a cluster tree based on kmers for a collection of sequences.

Options:
  -s, --seqfile PATH              path to .dvseqs file  [required]
  -o, --outpath PATH              the input string will be cast to Path instance
  -m, --moltype [dna|rna]         Molecular type of sequences  [default: dna]
  -k INTEGER                      k-mer size  [default: 6]
  --sketch-size INTEGER           sketch size for mash distance
  -d, --distance [mash|euclidean]
                                  distance measure for tree construction
                                  [default: mash]
  -c, --canonical-kmers           consider kmers identical to their reverse
                                  complement
  -L, --limit INTEGER             number of sequences to process
  -np, --numprocs INTEGER         number of processes  [default: 1]
  -hp, --hide_progress            hide progress bars
  --help                          Show this message and exit.

```
<!-- [[[end]]] -->

</details>

<details>
    <summary>Options for cogent3 app dvs_ctree</summary>

The `dvs ctree` is also available as the [cogent3 app](https://cogent3.org/doc/app/index.html) `dvs_ctree` or `dvs_par_ctree`. The latter is not composable, but can run the analysis for a single collection in parallel.

<!-- [[[cog
import cog
import contextlib
import io


from cogent3 import app_help

buffer = io.StringIO()

with contextlib.redirect_stdout(buffer):
  app_help("dvs_ctree")
cog.out(
    "```\n{}\n```".format(buffer.getvalue())
)
]]] -->
```
Overview
--------
Create a cluster tree from kmer distances.

Options for making the app
--------------------------
dvs_ctree_app = get_app(
    'dvs_ctree',
    k=12,
    sketch_size=3000,
    moltype='dna',
    distance_mode='mash',
    mash_canonical_kmers=None,
    show_progress=False,
)

Initialise parameters for generating a kmer cluster tree.

Parameters
----------
k
    kmer size
sketch_size
    size of sketches, only applies to mash distance
moltype
    seq collection molecular type
distance_mode
    mash distance or euclidean distance between kmer freqs
mash_canonical_kmers
    whether to use mash canonical kmers for mash distance
show_progress
    whether to show progress bars

Notes
-----
This app is composable.

If mash_canonical_kmers is enabled when using the mash distance,
kmers are considered identical to their reverse complement.

References
----------
.. [1] Ondov, B. D., Treangen, T. J., Melsted, P., Mallonee, A. B.,
   Bergman, N. H., Koren, S., & Phillippy, A. M. (2016).
   Mash: fast genome and metagenome distance estimation using MinHash.
   Genome biology, 17, 1-14.

Input type
----------
ArrayAlignment, SequenceCollection, Alignment

Output type
-----------
PhyloNode

```
<!-- [[[end]]] -->


<!-- [[[cog
import cog
import contextlib
import io


from cogent3 import app_help

buffer = io.StringIO()

with contextlib.redirect_stdout(buffer):
  app_help("dvs_par_ctree")
cog.out(
    "```\n{}\n```".format(buffer.getvalue())
)
]]] -->
```
Overview
--------
Create a cluster tree from kmer distances in parallel.

Options for making the app
--------------------------
dvs_par_ctree_app = get_app(
    'dvs_par_ctree',
    k=12,
    sketch_size=3000,
    moltype='dna',
    distance_mode='mash',
    mash_canonical_kmers=None,
    show_progress=False,
    max_workers=None,
    parallel=True,
)

Initialise parameters for generating a kmer cluster tree.

Parameters
----------
k
    kmer size
sketch_size
    size of sketches, only applies to mash distance
moltype
    seq collection molecular type
distance_mode
    mash distance or euclidean distance between kmer freqs
mash_canonical_kmers
    whether to use mash canonical kmers for mash distance
show_progress
    whether to show progress bars
numprocs
    number of workers, defaults to running serial

Notes
-----
This app is not composable but can run in parallel. It is
best suited to a single large sequence collection.

If mash_canonical_kmers is enabled when using the mash distance,
kmers are considered identical to their reverse complement.

References
----------
.. [1] Ondov, B. D., Treangen, T. J., Melsted, P., Mallonee, A. B.,
   Bergman, N. H., Koren, S., & Phillippy, A. M. (2016).
   Mash: fast genome and metagenome distance estimation using MinHash.
   Genome biology, 17, 1-14.

Input type
----------
ArrayAlignment, SequenceCollection, Alignment

Output type
-----------
PhyloNode

```
<!-- [[[end]]] -->

</details>
