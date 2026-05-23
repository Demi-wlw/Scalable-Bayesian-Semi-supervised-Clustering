"""Data loaders and constraint sampling for DIGITS and MNIST experiments.

This module provides minimal, paper-faithful data preparation utilities:
  - load_digits_binarized: sklearn DIGITS (1797 samples, 8x8 -> 64-d binary)
  - load_mnist_binarized:  OpenML MNIST  (70k samples, 28x28 -> 784-d binary)
  - sample_constraints:    draw pairwise (must-link / cannot-link) constraints
                           from ground-truth labels at a given supervision level.

Binarisation: fixed threshold at 0.5 of the normalised pixel intensity, matching
Appendix~``Data Preprocessing'' of the paper.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split


def _binarize_fixed(X_flat: np.ndarray) -> np.ndarray:
    """Threshold normalised pixel intensities at 0.5 to obtain binary features."""
    return (X_flat > 0.5).astype(np.int64)


def load_digits_binarized(test_size: float = 0.2, split_seed: int = 42) -> dict:
    """Load sklearn DIGITS, binarise, and split 80/20 with a fixed split seed.

    A fixed default ``split_seed=42`` matches the train/test partition used in
    the paper, so every multi-seed run shares the same canonical test set and
    only the model initialisation and constraint sampling vary.

    Returns
    -------
    dict with keys ``X_train, y_train, X_test, y_test``.
    """
    digits = load_digits()
    X = digits.data / 16.0  # normalise to [0, 1]
    y = digits.target.astype(np.int64)
    X = _binarize_fixed(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=split_seed
    )
    return {"X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te}


def load_mnist_binarized() -> dict:
    """Load MNIST from OpenML, binarise, and apply the standard 60k/10k split.

    Returns
    -------
    dict with keys ``X_train, y_train, X_test, y_test``.
    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)
    X = _binarize_fixed(X)
    # canonical 60k / 10k split (first 60k are train)
    return {
        "X_train": X[:60000],
        "y_train": y[:60000],
        "X_test": X[60000:],
        "y_test": y[60000:],
    }


def _flat_upper_triangular_index(i: np.ndarray, j: np.ndarray, n_samples: int) -> np.ndarray:
    """Map (i, j) with i < j and i, j in [0, n_samples) to flat upper-triangular index.

    Index ordering: for (i, j) with i < j,
    ``flat = i * (n_samples - 1) - i * (i - 1) // 2 + (j - i - 1)``.
    """
    i = i.astype(np.int64, copy=False)
    j = j.astype(np.int64, copy=False)
    return i * (n_samples - 1) - i * (i - 1) // 2 + (j - i - 1)


def _pair_index_to_rowcol(k: np.ndarray, n_labelled: int):
    """Bijection ``k -> (row, col)`` over the unordered pairs of ``n_labelled`` items.

    For k in [0, n_labelled * (n_labelled - 1) // 2), returns ``(row, col)``
    with row < col in [0, n_labelled).
    """
    two_n_minus_one = 2 * n_labelled - 1
    rows = np.floor(
        (two_n_minus_one - np.sqrt(two_n_minus_one**2 - 8 * k)) / 2
    ).astype(np.int64)
    cum = rows * (2 * n_labelled - rows - 1) // 2
    cols = (k - cum) + rows + 1
    return rows, cols


def sample_constraints(
    y: np.ndarray,
    supervision: float,
    rng: np.random.Generator,
    max_pairs: int = 5_000_000,
):
    """Sample pairwise must-link / cannot-link constraints from ground truth.

    A fraction ``supervision`` of samples is designated as labelled, then up to
    ``max_pairs`` pairs among the labelled samples become constraints:
    same-class -> +1 (must-link), different-class -> -1 (cannot-link).

    The output format matches BASIL's expectations: ``constraint_idx`` is a
    1D array of flattened upper-triangular indices into the full ``N x N``
    pair matrix, paired element-wise with ``constraint_val`` in ``{-1, +1}``.

    Parameters
    ----------
    y : np.ndarray of shape (N,)
        Ground-truth labels.
    supervision : float in [0, 1]
        Fraction of samples to treat as labelled.
    rng : np.random.Generator
        Source of randomness.
    max_pairs : int
        Upper bound on the number of constraint pairs to keep.

    Returns
    -------
    constraint_idx : np.ndarray of shape (M,)
        Flat upper-triangular indices of the constrained pairs.
    constraint_val : np.ndarray of shape (M,)
        +1 for must-link, -1 for cannot-link.
    """
    N = int(y.shape[0])
    n_labelled = int(round(supervision * N))
    if n_labelled < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int8)

    labelled = rng.choice(N, size=n_labelled, replace=False)
    labelled.sort()  # ensure i < j after lookup

    total_pairs = n_labelled * (n_labelled - 1) // 2
    n_keep = int(min(total_pairs, max_pairs))

    # Sample pair indices without materialising the full pair list.
    pair_local_indices = rng.choice(total_pairs, size=n_keep, replace=False)
    rows, cols = _pair_index_to_rowcol(pair_local_indices, n_labelled)

    sample_i = labelled[rows]
    sample_j = labelled[cols]

    flat_idx = _flat_upper_triangular_index(sample_i, sample_j, N)
    same = y[sample_i] == y[sample_j]
    vals = np.where(same, 1, -1).astype(np.int8)
    return flat_idx, vals
