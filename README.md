# Overview

## Identifying the sequences that maximise Jensen-Shannon

### `dvgt prep`: Preparing the sequence data

The sequences need to be processed before running the `max` command. This is done with the `prep` command. 

#### Typical usage:

```bash
dvgt prep -s <seqdir> -o <outpath> -p
```

#### Argument details:

- `-s`, `--seqdir`: Sequences can be a single fasta file or a directory full of fasta files. 

- `-o`, `--outpath`: Location to write processed sequences as HDF5.
- `-p`, `--parallel`: If set, run in parallel.
- `-F`, `--force_overwrite`: If set, overwrite existing output file if it exists.
- `-m`,` --moltype`: Molecular type of sequences, defaults to DNA.

### `dvgt max`: Maximise average delta JSD

Once the sequence data has been prepared using `dvgt prep`, the `max` command can be used to identify the sequences that maximise the Jensen-Shannon divergence. The kmer frequencies of the sequences are used to determine the Jensen-Shannon divergence

#### Typical usage:

```bash
dvgt max -s <seqfile> -o <outpath> -p -z 10 
```

#### Argument Details

- `-s`, `--seqfile`: HDF5 file containing sequences, must have been processed by the `prep` command.
- `-o`, `--outpath`: Path to write the output table.
- `-z`, `--min_size`: Minimum size of divergent set (default: 7).
- `-zp`, `--max_size`: Maximum size of divergent set.
- `-x`, `--fixed_size`: Result will have a fixed size number of sequences.
- `-k`: k-mer size (default: 3).
- `-st`, `--stat`: Statistic to maximise (choices: total_jsd, mean_delta_jsd, mean_jsd; default: mean_delta_jsd).
- `-p`, `--parallel`: Run in parallel.
- `-L`, `--limit`: Number of sequences to process.
- `-T`, `--test_run`: Reduce number of paths and size of query sequences.
- `-v`, `--verbose`: Increase verbosity.

## Running the tests

```
$ pytest -n auto
```

This runs in parallel, greatly speeding things up.

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
  max   Identify the seqs that maximise average delta JSD
  prep  Writes processed sequences to an HDF5 file.

```
<!-- [[[end]]] -->