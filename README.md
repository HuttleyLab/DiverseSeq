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
  prep  Writes processed sequences to an HDF5 file.
  max   Identify the seqs that maximise average delta JSD

```
<!-- [[[end]]] -->

### `dvgt prep`: Preparing the sequence data

The sequences need to be processed before running the `max` command. This is done with the `prep` command. 

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

  Writes processed sequences to an HDF5 file.

Options:
  -s, --seqdir PATH        directory containing fasta formatted sequence files
                           [required]
  -o, --outpath PATH       location to write processed seqs as HDF5  [required]
  -p, --parallel           run in parallel
  -F, --force_overwrite    Overwrite existing file if it exists
  -m, --moltype [dna|rna]  Molecular type of sequences, defaults to DNA
                           [default: dna]
  --help                   Show this message and exit.

```
<!-- [[[end]]] -->

### `dvgt max`: Maximise average delta JSD

Once the sequence data has been prepared using `dvgt prep`, the `max` command can be used to identify the sequences that maximise the Jensen-Shannon divergence. The kmer frequencies of the sequences are used to determine the Jensen-Shannon divergence

#### Usage:

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
  -s, --seqfile PATH              HDF5 file containing sequences, must have been
                                  processed by the 'prep' command  [required]
  -o, --outpath PATH              the input string will be cast to Path instance
  -z, --min_size INTEGER          minimum size of divergent set  [default: 7]
  -zp, --max_size INTEGER         maximum size of divergent set
  -x, --fixed_size                result will have size number of seqs
  -k INTEGER                      k-mer size  [default: 3]
  -st, --stat [total_jsd|mean_delta_jsd|mean_jsd]
                                  statistic to maximise  [default:
                                  mean_delta_jsd]
  -p, --parallel                  run in parallel
  -L, --limit INTEGER             number of sequences to process
  -T, --test_run                  reduce number of paths and size of query seqs
  -v, --verbose                   is an integer indicating number of cl
                                  occurrences  [default: 0]
  --help                          Show this message and exit.

```
<!-- [[[end]]] -->

## Running the tests

```
$ pytest -n auto
```

This runs in parallel, greatly speeding things up.

