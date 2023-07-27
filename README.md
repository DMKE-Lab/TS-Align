# TS-Align

Code for  paper “TS-Align: A Temporal Similarity-Aware Entity Alignment Model for Temporal Knowledge Graphs”.

## Requirements

- Anaconda>=4.5.11
- Python>=3.7.11
- pytorch>=1.10.1

## Datasets

The datasets are from [TEA-GNN](https://github.com/soledad921/TEA-GNN).  

ent_ids_1: ids for entities in source KG;  

ent_ids_2: ids for entities in target KG;  

triples_1: relation triples encoded by ids in source KG;  

triples_2: relation triples encoded by ids in target KG;  

rel_ids_1: ids for entities in source KG;  

rel_ids_2: ids for entities in target KG;  

sup_pairs + ref_pairs: entity alignments  

## Usage:

Use the following command:  

``` bash
python main.py
```

## Acknowledgement

We refer to the code of [STEA](@inproceedings{STEA2022,
author = {Li Cai, Xin Mao, Meirong Ma, Hao Yuan, Jianchao Zhu, Man Lan},
title = {A Simple Temporal Information Matching Mechanism for Entity Alignment Between Temporal Knowledge Graphs},
booktitle = {COLLING},
year = {2022}，
}). Thanks for their contributions.