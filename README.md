# Overview

## Identifying the sssequences that maximise Jensen-Shannon

```
$ dvgt max -s
```

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

  dvgt -- alignment free measurement of divergent sequences

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  find-species  find occurrences of signature k-mers in query sequences
  max           identify the seqs that maximise average delta entropy
  seqs2kmers    write kmer data for seqs
  sig-kmers     write k-mers that uniquely identify each sequence.

```
<!-- [[[end]]] -->