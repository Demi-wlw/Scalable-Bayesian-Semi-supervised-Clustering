"""Utility functions used by the BASIL model.

Only includes helpers actually referenced from ``basil.py`` (and ``cluster_acc``
used by the experiment scripts). Plot/IO/initialisation helpers belonging to
other research artefacts have been removed.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import digamma


# -------------------------------------------------------------------------
# Sequence and graph helpers
# -------------------------------------------------------------------------

def cumsum_ex(arr):
    """Exclusive cumulative sum: ``cum_sum_arr[i] = sum(arr[:i])`` with the first entry zero."""
    cum_sum_arr = np.zeros_like(arr)
    for i in range(len(arr)):
        if i == 0:
            cum_sum_arr[i] = 0
        else:
            cum_sum_arr[i] = np.cumsum(arr[:i])[-1]
    return cum_sum_arr


def get_tuples(dict_N):
    """Return unique unordered pairs ``(n, m)`` from a neighbour-list dictionary."""
    tuples = []
    seen = set()
    for n, neighbors in dict_N.items():
        for m in neighbors:
            pair = (n, m) if n <= m else (m, n)
            if pair not in seen:
                seen.add(pair)
                tuples.append(pair)
    return tuples


def _inverse_upper_triangular_indices(flat_idx, n_samples):
    """Convert flattened upper-triangular indices back to ``(row, col)`` pairs."""
    n_samples = int(n_samples)
    total_pairs = n_samples * (n_samples - 1) // 2
    flat_idx_arr = np.atleast_1d(np.asarray(flat_idx, dtype=np.int64))
    if np.any((flat_idx_arr < 0) | (flat_idx_arr >= total_pairs)):
        raise ValueError("flat_idx entries must be within the upper-triangular range")

    two_n_minus_one = 2 * n_samples - 1
    rows = np.floor(
        (two_n_minus_one - np.sqrt(two_n_minus_one**2 - 8 * flat_idx_arr)) / 2
    ).astype(np.int64)
    prior_counts = rows * (2 * n_samples - rows - 1) // 2
    cols = (flat_idx_arr - prior_counts) + rows + 1
    return rows, cols


def _build_constraint_graph_sparse(constraint_idx, constraint_val, n_samples, target_value):
    """Build an adjacency-list graph from sparse constraint arrays.

    Parameters
    ----------
    constraint_idx : array-like
        Flat upper-triangular indices of non-zero constraints.
    constraint_val : array-like
        Constraint values (+1 for must-link, -1 for cannot-link).
    n_samples : int
        Number of samples in the dataset.
    target_value : int
        Constraint type to extract (+1 must-link, -1 cannot-link).

    Returns
    -------
    graph : dict[int, list[int]]
        Adjacency list mapping sample index to its constrained neighbours.
    mask : np.ndarray of shape (n_samples,)
        Binary mask marking samples that participate in at least one constraint.
    """
    graph = {i: [] for i in range(n_samples)}
    mask = np.zeros((n_samples,), dtype=int)
    constraint_idx = np.asarray(constraint_idx, dtype=np.int64).ravel()
    constraint_val = np.asarray(constraint_val, dtype=np.int8).ravel()
    if constraint_idx.size == 0:
        return graph, mask
    valid_idx = constraint_idx[constraint_val == target_value]
    if valid_idx.size == 0:
        return graph, mask
    rows, cols = _inverse_upper_triangular_indices(valid_idx, n_samples)
    for r, c in zip(rows, cols):
        graph[int(r)].append(int(c))
        graph[int(c)].append(int(r))
        mask[int(r)] = 1
        mask[int(c)] = 1
    return graph, mask


def construct_mustlink_sparse(constraint_idx, constraint_val, n_samples):
    """Build adjacency lists for must-link constraints (+1) from sparse arrays."""
    return _build_constraint_graph_sparse(constraint_idx, constraint_val, n_samples, target_value=1)


def construct_cannotlink_sparse(constraint_idx, constraint_val, n_samples):
    """Build adjacency lists for cannot-link constraints (-1) from sparse arrays."""
    return _build_constraint_graph_sparse(constraint_idx, constraint_val, n_samples, target_value=-1)


# -------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------

def cluster_acc(Y_pred, Y):
    """Clustering accuracy under the best label permutation (Hungarian matching).

    Parameters
    ----------
    Y_pred : np.ndarray of shape (N,)
        Predicted cluster indices.
    Y : np.ndarray of shape (N,)
        Ground-truth class indices.

    Returns
    -------
    accuracy : float
        Fraction of correctly assigned samples under the optimal permutation.
    confusion : np.ndarray of shape (D, D)
        Confusion matrix in the original label space.
    """
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / Y_pred.size, w


# -------------------------------------------------------------------------
# SVI helpers
# -------------------------------------------------------------------------

def normalize(params, axis=0):
    """Normalise ``params`` so its entries sum to 1 along ``axis``."""
    return params / np.sum(params, axis=axis, keepdims=True)


def get_batch(X, phi, M):
    """Draw an index-aligned mini-batch of size ``M`` from ``X`` and ``phi``."""
    N = X.shape[0]
    if N == 0:
        raise ValueError("Cannot draw batch from empty dataset")
    valid_indices = np.arange(N)
    batch_indices = np.random.choice(valid_indices, size=M, replace=False)
    batch_phi = phi[batch_indices, :]
    return batch_indices, batch_phi


def stochastic_update(params_h, old_params, pho):
    """Robbins-Monro convex combination: ``new = (1 - pho) * old + pho * params_h``."""
    new_params = {}
    for key in params_h.keys():
        new_params[key] = (1 - pho) * old_params[key] + pho * params_h[key]
    return new_params


# -------------------------------------------------------------------------
# Special functions
# -------------------------------------------------------------------------

def multivar_digamma(nu, d):
    """Multivariate digamma function for degree-of-freedom parameter ``nu``."""
    if np.array(nu).shape == ():
        nu = np.reshape(nu, [1])
    K = np.array(nu).shape[0]
    return np.sum(
        digamma(0.5 * (nu.reshape(1, K) + 1 - np.arange(1, d + 1).reshape(d, 1))),
        0,
    ).squeeze()
