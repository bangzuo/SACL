# SACL: Self-Augmented Contrastive Learning for Knowledge-aware Recommendation

This is the PyTorch Implementation for the paper: Self-Augmented Contrastive Learning for Knowledge-aware Recommendation

## Introduction

Self-Augmented Contrastive Learning for Knowledge-aware Recommendation (SACL) is a knowledge-aware recommendation framework, which can be used to extract unbiased graph structures, and uses the patterns of association between the head and tail nodes in an unbiased graph to normalize the representations of the tail nodes. It aims to facilitate unbiased and distribution-agnostic knowledge-aware contrastive learning across views.

```

## Environment Requirement

The code has been tested running under Python 3.8.0. The required packages are as follows:

- pytorch == 2.1.0
- networkx == 2.5.1
- numpy == 1.22.4
- pandas == 1.4.3
- scikit-learn == 1.1.1
- scipy == 1.7.0
- torch-cluster == 1.6.2
- torch-scatter == 2.1.2
- torch-sparse == 0.6.18

## Usage

The instruction of commands has been clearly stated in the codes (see the parser function in utils/parser.py). 
```

- Last-fm dataset

```
python main.py --dataset last-fm --lr 0.0001 --cross_cl_tau 0.1 --context_hops 2 --mae_rate 0.005 --aug_ui_rate 0.2  --aug_kg_rate 0.8
```

- Book-Crossing dataset

```
python main.py --dataset MovieLens --lr 0.0001 --cross_cl_tau 0.1 --context_hops 1 --mae_rate 0.01 --aug_ui_rate 0.8 --aug_kg_rate 0.2
```

- MovieLens dataset

```
python main.py --dataset MovieLens --lr 0.0001 --cross_cl_tau 0.9 --context_hops 2 --n_iter mae_rate 0.01 --aug_ui_rate 0.2 --aug_kg_rate 0.8
```


## Dataset

We provide three processed datasets: Last-FM and MovieLens.

- You can find the full version of recommendation datasets via [Last-FM](https://grouplens.org/datasets/hetrec-2011/) and [Book-crossing](http://www2.informatik.uni-freiburg.de/~cziegler/BX/) and [MovieLens](https://grouplens.org/datasets/movielens/1m/).
- We follow the previous study to preprocess the datasets.

|                       |               |     Last-FM |  Book-Crossing | MovieLens |
| :-------------------: | :------------ | ----------: |  ------------: | --------: |
| User-Item Interaction | #Users        |       1,872 |      17860     |     6,036 |
|                       | #Items        |       3,915 |      14967     |     2,347 |
|                       | #Interactions |      42,346 |     139746     |   753,772 |
|                       | #Density      |      5.8e-3 |     5.1e-2     |   5.2e-4  |
|    Knowledge Graph    | #Entities     |       9,366 |      77903     |     6,729 |
|                       | #Relations    |          60 |         25     |         7 |
|                       | #Triplets     |      15,518 |     151500     |    20,195 |


