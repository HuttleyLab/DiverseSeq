# Overview

## Identifying the sequences that maximise Jensen-Shannon

### Maximise average delta JSD

The input can be a single file (as below) or a directory full of files. In either case, the k-mer frequencies of the sequences are used to maximise the delta jsd values. This command will start with 10 sequences and use 3-mers. It will generate a tsv file with only sequences that have a +ve delta jsd. The `-v` flag displays some additional output.
```
$ dvgt max -s ~/repos/Cogent3/tests/data/brca1.fasta -o ~/Desktop/Outbox/delme/delme.tsv -k 3 -v -z 10 
```

The following command takes a directory that has over 1k fasta files of bacterial genomes. The computations are as above except the `-p` flag indicates the reading of data will be done in parallell using 6 cores.
```
$ dvgt max -s soil_reference_genomes_fasta -o ~/Desktop/Outbox/delme/delme.tsv -p -k 3 -v 
```

The following command is the same as above except the `-x` flag means strictly 10 sequences will be output.
```
$ dvgt max -s soil_reference_genomes_fasta -o ~/Desktop/Outbox/delme/delme.tsv -p -k 3 -z 10 -x 
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
  max  identify the seqs that maximise average delta entropy

```
<!-- [[[end]]] -->