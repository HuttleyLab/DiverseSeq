# Files used for the ctree experiments

## `experiment.py`

Calls ctree on the mammals alignment for different values of k and sketch size.

## `iq_experiment.py`

Uses iqtree to find maximum likelihood tree on mammals alignment.

> **WARNING**: This required using a local build of [piqtree](https://github.com/iqtree/piqtree) as the PyPI version does not yet support Python 3.13. That issue should soon be rectified.

## `likelihoods.py`

Find likelihood of all generated trees from `experiment.py` and `iq_experiment.py`.
