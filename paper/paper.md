---
title: 'Divergent: an application for selecting representative biological sequences'
tags:
  - Python
  - genomics
  - statistics
  - machine learning
  - bioinformatics
  - molecular evolution
  - phylogenetics
authors:
  - name: Gavin Huttley
    orcid: 0000-0001-7224-2074
    affiliation: 1
  - name: Katherine Caley
    affiliation: 1
affiliations:
 - name: Research School of Biology, Australian National University, Australia
   index: 1
date: 13 August 2024
bibliography: paper.bib

---

# Summary

for many biological analyses, it's often beneficial to develop an understanding of the properties of a data set by working on a representative subset

sequences do not have to be related to each other

it may be that the sampled data are descended from a common ancestor, or that they are not

Many analyses concerning biological sequences have poor scalability. For example, phylogenetic reconstruction, multiple sequence alignment, detecting the influence of natural selection. selection. The process of performing these expensive computational tasks can benefit from working on representative sequences. Existing solutions to this problem while themselves can be efficient may rely on a pre-processing step that is highly inefficient and has poor scalability. For instance, a phylogenetic tree provides a good basis for sampling representative sequences but the computational cost of producing that tree can be prohibitive. Here we present Divergent, an alignment free algorithm for efficiently sampling representative sequences.

Divergent is implemented using Python and provides both a command-line interface and cogent3 plugins. The output of Divergent is a set of representative sequences which can then be used for efficiently prototyping larger scale analyses.

# Statement of need

current tools [@balaban.2019.plosone] are computationally costly in terms of requiring existence of a multiple sequence alignment or a phylogenetic tree or a pairwise distance matrix

drawing insights from biological sequence data typically involves establishing relationships to existing sequence data

as the scale of data sets becomes larger, the value of identifying analysis parameters on a representative subset of the data increases

if we imagine a 1D line, the idea is to sample points that are approximately evenly dispersed along the line

# Definitions

$k$-mer is a subsequence of length $k$.

Define $f_i$ as the $k$-mer frequency vector for sequence $i$, and the set of $N$ such vectors as $\mathbb{F}$. The Jensen-Shannon Divergence for $\mathbb{F}$ is

\begin{equation*}
JSD=H \left( \frac{1}{N}\sum_i^N f_i \right) - \overline{H(f)}
\end{equation*}

where $\overline{H(f)}$ is the mean of the frequency vectors Shannon entropies.

For sequence $i$, it's contribution to the total JSD of $\mathbb{F}$ is

\begin{equation*}
JSD_{\delta}(i)=JSD(\mathbb{F})-JSD(\mathbb{F} - \{i\})
\end{equation*}

with this expression it becomes clearer that to measure entropy of the collection we only need to keep track of the totals of each $k$-mer, so the algorithm can be implemented with a single pass through the data with a small constant due to the need to update the chosen sequences

Add a figure describing the core algorithm as a flow chart.

# Algorithm

the statistic

choice of k

# `dvgt` command line application

- prep
- sample *n*
- sample maximally divergent

# `dvgt` cogent3 apps

counterparts to the above

# Performance

## recovery of representatives from synthetic knowns

## statistical correspondence to tree-based distances




## compute time and memory

- on greengenes
- on 1k microbial genomes

# Figures

blah

# Tables


**Table 1** Jensen-Shanon Divergence (JSD) for different relationships between two sequences.\label{JSD-examples}

+--------------+--------+--------+-----+
| Relationship |   seq1 |   seq2 | JSD |
+==============+========+========+=====+
|    Identical | `ATCG` | `TCGA` | 0.0 |
+--------------+--------+--------+-----+
|   No overlap | `AAAA` | `TTTT` | 1.0 |
+--------------+--------+--------+-----+
| Intermediate | `ATCG` | `ATCC` | 0.5 |
+==============+========+========+=====+


# Acknowledgements

blah blah

# References