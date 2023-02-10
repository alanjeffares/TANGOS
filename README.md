## TANGOS: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization
This repository contains the code associated with [our ICLR 2023 paper](https://openreview.net/forum?id=n6H86gW8u0d) where we introduce a novel regularizer for training deep neural networks. Tabular Neural Gradient Orthogonalization and Specialization (TANGOS) provides a framework for regularization in the tabular setting built on latent unit attributions. For further details, please consult the paper.


### Getting Started With TANGOS
TODO: add getting started notebook here.

### Experiments
**Contents**

These folders are associtated with the commented experiments from the paper.
```
└── src
    ├── behavior-analysis            # TANGOS Behavior Analysis.
    ├── compute                      # Approximation and Algorithm.
    ├── in-tandem-regularization     # Generalisaton: In Tandem Regularization.
    ├── larger-data                  # Performance With Increasing Data Size.
    └── stand-alone-regularization   # Generalization: Stand-Alone Regularization.
```

The main experiments can be run by navigating to the root folder and running the following command.

```python src/<experiment name>/main.py```

The behavior analysis and compute expirements are included in ```.ipynb``` notebooks with instructions included.

### Citation
If you use this code, please cite the associated paper.
```
@inproceedings{
jeffares2023tangos,
title={{TANGOS}: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization},
author={Alan Jeffares and Tennison Liu and Jonathan Crabb{\'e} and Fergus Imrie and Mihaela van der Schaar},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=n6H86gW8u0d}
}
```
