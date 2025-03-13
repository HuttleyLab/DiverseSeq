# Scripts and notebooks for the `diverse-seq` manuscript

To reproduce the analyses in the manuscript we recommend you create a conda environment with python>=3.12. 

> **Note**
> We do not set seeds for the random number generation components of this work, so results will vary slightly from those in the manuscript. But overall trends will be the same.

In general, we pair a python script and a jupyter notebook. The former provides a command line interface to performing an analysis. The latter aggregates the results of the analaysis and provides interpretation plus produces the figures used in the manuscript.

The `project_path.py` script includes the defined paths for different components of the analyses.

## Getting the data

The `python get_data_sets.py` script will download *most* of the data necessary. We suggest you directly download some of the larger files (see comments in that script for details).

## Generating synthetic known data sets

This is done by the `synthetic_known.py` script, which writes out the results of the analysis.

## JSD versus genetic distance

Relates Jensen-Shannon Divergence based measures with explicit genetic distance statistics. The `jsd_vs_dist.py` produces the output that is interrogated by the corresponding notebook.

## ctree analyses

See the ctree/README.md for details on how the ctree analyses were run.

## plugin demo's

A jupyter notebook demonstrating the use of the `diverse-seq` cogent3 apps. Also used to produce some manuscript figures.
