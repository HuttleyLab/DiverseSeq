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
header-includes:
  - \input{header.tex}
---

# Summary

alignment free, tree free, low compute cost, high scalability

sequences do not have to be related to each other

for many biological analyses, it's often beneficial to develop an understanding of the properties of a data set by working on a representative subset


it may be that the sampled data are descended from a common ancestor, or that they are not

Many analyses concerning biological sequences have poor scalability. For example, phylogenetic reconstruction, multiple sequence alignment, detecting the influence of natural selection. selection. The process of performing these expensive computational tasks can benefit from working on representative sequences. Existing solutions to this problem while themselves can be efficient may rely on a pre-processing step that is highly inefficient and has poor scalability. For instance, a phylogenetic tree provides a good basis for sampling representative sequences but the computational cost of producing that tree can be prohibitive. Here we present Divergent, an alignment free algorithm for efficiently sampling representative sequences.

Divergent is implemented using Python and provides both a command-line interface and cogent3 plugins. The output of Divergent is a set of representative sequences which can then be used for efficiently prototyping larger scale analyses.

# Statement of need

Many bioinformatics analyses are costly in terms of compute time. Tools that facilitate the development of smaller scale prototypes can accelerate the execution of research projects without requiring enormous resources. 

as the scale of data sets becomes larger, the value of identifying analysis parameters on a representative subset of the data increases

Current tools [@balaban.2019.plosone] require existence of a multiple sequence alignment or a phylogenetic tree or a pairwise distance matrix. Algorithms for all these approaches require homologous sequences as input. Each of those components is in turn computationally expensive.

Divergent is more flexible than published approaches. It is alignment free and does not require sequences to be related. As we show, in the case of homologous sequences, the set selected by divergent is comparable to what would be expected under published approaches. Moreover, the algorithm is linear in time.


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

with this expression it becomes clearer that to measure entropy of the collection we only need to keep track of the totals of each $k$-mer. Thus, the algorithm can be implemented with a single pass through the data with a small constant due to the need to update the chosen sequences

Add a figure describing the core algorithm as a flow chart.

# Algorithm

the statistic

choice of k

# `dvgt` command line application

- `prep`converts sequences into numpy arrays for faster processing
- `nmost` samples the n sequences that increase JSD 
- `max` samples divergent sequences that maximise a user specified statistic, either the standard deviation or the coefficient of variation of $JSD_{\delta}$.

# `dvgt` cogent3 apps

counterparts to the above

# Performance

## recovery of representatives from synthetic knowns

## Testing a hypothesis about the relationship with tree-based sampling

For homologous DNA sequences, increasing the amount of elapsed time since they shared a common ancestor increases their genetic distance. We also expect that the JSD between two sequences will increase proportional to the amount of time since they last shared a common ancestor. These lead to the expectation that there should be a relationship between genetic distance and JSD. This in turn leads us to formulate the following hypothesis for a divergent set with $N$ sequences. If JSD is uninformative, then the set of sequences chosen by `divergent` will be no better than a randomly selected set of size $N$. Under the alternate hypothesis, we expect the minimum genetic distance between the sequences chosen by `divergent` will be larger than that between a randomly selected set of $N$ sequences.


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