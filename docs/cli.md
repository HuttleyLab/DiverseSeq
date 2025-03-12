```bash exec="1"
mkdir demo # markdown-exec: hide
```

## Creating a demo dataset

The exported fasta formatted sequence file `demo.fa` is a collection of 55 unaligned sequences. If you provide an alignment, `diverse-seq` applications will remove any gaps before processing.

```console exec="1" source="console" result="ansi" workdir="./demo"
$ dvs demo-data
```

Displaying the first few lines of that file.

```console exec="1" source="console" result="ansi" workdir="./demo"
$ head demo.fa
```

## The `prep` command

This command converts either a the sequences in a single file, or a directory of files, into a HDF5 file format. This is more efficient for analysis.

```console exec="1" source="console" result="ansi" workdir="./demo"
$ dvs prep -s demo.fa -o demo.dvseqs -hp
```

## The `nmost` command

This command selects the *n* most diverse sequences, outputting them to a `.tsv` file. We specify the *k*-mer size (`-k 6`), the value of *n* (`-n 10`) and we hide progress bars (`-hp`).

```console exec="1" source="console" result="ansi" workdir="./demo"
$ dvs nmost -s demo.dvseqs -o demo-nmost.tsv -k 6 -n 10 -hp
```

The output file has two coilumns, the first is the name of the file the sequence came from, and the second is the delta_jsd value, the contribution of this sequence to the Jensen-Shannon Divergence of the final collection.

```console exec="1" source="console" result="ansi" workdir="./demo"
$ head demo-nmost.tsv
```

## Selecting the maximally diverse sequences using `max` command

The `max` command maximises the standard deviation of delta_jsd in a collection. The user specifies the minimum (`-z 5`) and maximum (`-zp 10`) size of the final collection. We also set the random number seed (`--seed 1741676171`). If the verbose flag is set (`-v`), the command will show the random seed used (which defaults to the system time).

```console exec="1" source="console" result="ansi" workdir="./demo"
$ dvs max -s demo.dvseqs -o demo-max.tsv -k 6 -z 5 -zp 10 -hp --seed 1741676171
```

## Estimating a tree from mash distances using `ctree`

The `ctree` command produces an approximate tree from a collection of unaligned sequences using either the Euclidean distance or the Mash distance. We specify the *k*-mer size (`-k 12`), the sketch size (`--sketch-size 3000`), and the distance metric (`-d mash`). This command ouputs a newick formatted trees string to file.

```console exec="1" source="console" result="ansi" workdir="./demo"
$ dvs ctree -s demo.dvseqs -o demo-ctree.nwk -k 12 -d mash --sketch-size 3000 -hp
```

```bash exec="1"
rm -rf demo # markdown-exec: hide
```
