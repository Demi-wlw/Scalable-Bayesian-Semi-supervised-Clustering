# BASIL: Bayesian Semi-supervised Clustering with Feature Selection and Adaptive Constraint Weighting

Reference implementation for the ICML 2026 paper
*Scalable Bayesian Semi-supervised Clustering with Feature Selection and Adaptive Constraint Weighting*.

BASIL is a unified Bayesian generative model that jointly captures
HMRF-structured cluster assignments, per-cluster feature relevance, and latent
constraint reliability within a single hierarchical formulation. Stochastic
variational inference (SVI) provides scalable joint posterior inference,
yielding interpretable per-cluster feature attribution while remaining robust
to noisy pairwise supervision.

## Repository layout

```
BASIL/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── model/
│   ├── __init__.py        # exposes the `BASIL` class
│   ├── basil.py           # BASIL model class with SVI inference
│   └── utils.py           # constraint-graph builders, ACC matching, batch helpers
└── experiments/
    ├── __init__.py
    ├── data.py            # DIGITS and MNIST loaders + constraint sampling
    └── runBASIL.py        # single CLI entry point for both DIGITS and MNIST
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or newer is recommended. Runtime dependencies:
`numpy`, `scipy`, `scikit-learn`, `kmedoids`, `kmodes`, `tqdm`, `matplotlib`.

## Quick start (programmatic)

```python
import numpy as np
from model import BASIL

X = np.load("X.npy")             # (N, D) binary features
c_idx = np.load("c_idx.npy")     # (M,)  flattened upper-triangular indices
c_val = np.load("c_val.npy")     # (M,)  +1 must-link, -1 cannot-link

model = BASIL(K=10, likelihood="bernoulli", batch_size=500)
model.Inference(X,
                constraint_idx=c_idx, constraint_val=c_val,
                max_iter=2000, feat_select=True)
preds = model.predict(X)
```

## Reproducing the paper (CLI)

A single command-line entry point handles both DIGITS and MNIST:

```bash
# DIGITS at 20% supervision (default (1, D+FS) of Table 1)
python -m experiments.runBASIL --dataset digits --supervision 0.20 --seed 1234

# DIGITS at 50% supervision
python -m experiments.runBASIL --dataset digits --supervision 0.50 --seed 1234

# MNIST at 50% supervision (~1.6h on a single EPYC 7742 core)
python -m experiments.runBASIL --dataset mnist --supervision 0.50 --seed 1234

# Adaptive constraint weighting under noisy supervision
python -m experiments.runBASIL --dataset digits --supervision 0.20 \
    --w-learn --no-KL --seed 1234

# Aggregate 10 seeds in parallel and serialize results
python -m experiments.runBASIL --dataset digits --supervision 0.20 \
    --runs 10 --seed 1234 --max-workers 4 --output digits20.pkl
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--dataset` | required | `digits` or `mnist` |
| `--supervision` | `0.20` | Fraction of training samples labelled |
| `--K` | `10` | Number of clusters |
| `--seed` | `1234` | Base random seed |
| `--runs` | `1` | Number of seeds (seeds = base + 0..runs-1) |
| `--max-workers` | all CPUs | Parallel workers for multi-seed runs |
| `--batch-size` | `500` | Mini-batch size for SVI |
| `--max-iter` | 2000 / 3000 | Max SVI iterations (DIGITS / MNIST defaults) |
| `--init` | `kmeans` | `kmeans` auto-dispatches to K-medoids (small Bernoulli N) or K-modes (large Bernoulli N); `random` for ablation |
| `--max-pairs` | `5e6` | Cap on the number of constraint pairs sampled |
| `--w-learn` | `False` | Enable adaptive constraint weighting |
| `--no-KL` | `False` | Disable cluster-divergence KL potential (use Potts) |
| `--no-FS` | `False` | Disable feature selection |
| `--wprior` | `1.0` | Gamma-prior rate for constraint weights |
| `--fsprior` | `0.1` | Beta-prior rate for feature relevance |
| `--warmup-prop` | `0.0` | Fraction of iterations under fixed weights before adaptive switch |
| `--output` | none | Optional `.pkl` path to save results |

Default configuration mirrors Table 1's `(1, D+FS)`: fixed constraint weights
($w_{nm}=1$), cluster-divergence KL potential, integrated feature selection.

Full help: `python -m experiments.runBASIL --help`.

## Expected results

From Table 1 of the paper, on a single AMD EPYC 7742 core:

| Dataset | Supervision | Test ACC (mean ± SE) | Time |
|---|---|---|---|
| DIGITS | 0% | 0.68 ± 0.01 | ~6 s |
| DIGITS | 20% | 0.80 ± 0.02 | ~7 min |
| DIGITS | 50% | 0.87 ± 0.00 | ~44 min |
| MNIST | 0% | 0.59 ± 0.02 | ~19 min |
| MNIST | 20% | 0.68 ± 0.02 | ~1.5 hr |
| MNIST | 50% | 0.79 ± 0.00 | ~1.6 hr |

## Citing

If you use this code, please cite the ICML 2026 paper:

```bibtex
@inproceedings{wang2026basil,
    title     = {Scalable Bayesian Semi-supervised Clustering with Feature
                 Selection and Adaptive Constraint Weighting},
    author    = {Wang, Luwei and Panas, Dagmara and Wang, Ke and
                 Guthrie, Bruce and Seth, Sohan},
    booktitle = {Proceedings of the 43rd International Conference on
                 Machine Learning (ICML)},
    year      = {2026}
}
```

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Demi-wlw.
