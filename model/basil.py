"""BASIL: Bayesian Semi-supervised clustering with feature selection and adaptive constraint weighting.

Bernoulli mixture model with HMRF-structured constraints, learnable Gamma-prior
constraint weights, and Beta-prior feature relevance, fit via stochastic
variational inference (SVI).
"""
import time
import logging
import numpy as np
from numpy import linalg as LA
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kmedoids import KMedoids
from kmodes.kmodes import KModes
from tqdm import tqdm
from scipy.special import logsumexp, betaln, digamma
from scipy.stats import dirichlet, beta

from .utils import (
    cumsum_ex,
    get_tuples,
    construct_mustlink_sparse,
    construct_cannotlink_sparse,
    normalize,
    get_batch,
    stochastic_update,
    multivar_digamma,
)

logger = logging.getLogger(__name__)

# Guardrails preventing floating-point warnings from propagating across the model updates.
_SAFE_EXP_MIN = -745.0  # exp(-745) is ~5e-324 which is close to float underflow.
_SAFE_EXP_MAX = 709.0   # exp(709) is ~8.2e307 which is near float overflow.

def _safe_exp(values):
    """Exponentiate after clipping so extreme logits do not trigger under/overflow."""
    clipped = np.clip(values, _SAFE_EXP_MIN, _SAFE_EXP_MAX)
    with np.errstate(over="ignore", under="ignore"):
        return np.exp(clipped)

def _safe_logsumexp(values, axis=None, keepdims=False):
    """Stable logsumexp variant that silences benign floating-point warnings."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return logsumexp(values, axis=axis, keepdims=keepdims)

def initialise_phi_with_kmeans(X, K, likelihood="gaussian"):
    """Seed the responsibility matrix using clustering heuristics.

    Parameters
    ----------
    X : ndarray of shape (N, d)
        Training data used only for initialisation.
    K : int
        Maximum number of clusters.
    likelihood : {"gaussian", "bernoulli"}, default="gaussian"
        Likelihood family that dictates which clustering primitive is used.

    Returns
    -------
    phi : ndarray of shape (N, K)
        Normalised responsibilities derived from distances to the K prototypes.
    mu : ndarray
        Prototype matrix (cluster centroids or medoids) matching ``likelihood``.
    """
    if likelihood == "bernoulli":
        if X.shape[0] < 1000:
            logger.info("KMedoids initialization of phi")
            km = KMedoids(K, metric='hamming', method='fasterpam').fit(X)
            mu = X[km.medoid_indices_]
        else:
            logger.info("KModes initialization of phi")
            km = KModes(K, n_init=5).fit(X)
            mu = km.cluster_centroids_
        diffs = np.not_equal(X[:, np.newaxis, :], mu[np.newaxis, :, :]).mean(axis=2)
        phi = _safe_exp(-diffs)
    elif likelihood == "gaussian":
        logger.info("KMeans initialization of phi")
        mu = KMeans(K).fit(X).cluster_centers_
        phi = _safe_exp(-0.5 * LA.norm(X.reshape(X.shape[0], 1, X.shape[1]) - mu.reshape(1, K, X.shape[1]), 2, 2))
    return normalize(phi, 1), mu


class BASIL:
    """Hierarchical Markov random field Dirichlet process mixture.

    The model implements stochastic variational inference (SVI) for a
    semi-supervised Dirichlet process mixture with pairwise must-link and
    cannot-link constraints enforced through an HMRF penalty.

    Parameters
    ----------
    K : int
        Number of clusters / Stick-breaking truncation level (maximum latent clusters) for Dirichlet process.
    alpha : float, default=1.0
        Concentration prior for the mixture weights.
    epsilon : float, default=1e-9
        Numerical stability constant used throughout the updates.
    init : {"kmeans", "random"}, default="kmeans"
        Strategy for initialising the responsibilities.
    weight_prior : {"Dirichelet distribution", "Dirichelet Process"}
        Choice between a finite Dirichlet or stick-breaking prior.
    aa : float, default=1.0
        Stick-breaking Beta(1, aa) prior hyperparameter.
    likelihood : {"bernoulli", "gaussian"}, default="bernoulli"
        Observation model family.
    batch_size : int, default=500
        Mini-batch size for SVI.
    tau0 : float, default=1.0
        Delay parameter that shapes the Robbins-Monro schedule.
    kappa_svi : float, default=0.7
        Forgetting rate in (0.5, 1] for the SVI step size.
    elbo_ema_decay : float, default=0.9
        Exponential decay used to smooth ELBO traces.
    w_learn : bool, default=False
        Whether to learn adaptive constraint weights. When True, the pairwise
        constraint weights are updated based on KL divergence; when False,
        all weights are fixed to 1.
    KL : bool, default=True
        Whether to use KL divergence to compute potentials in Bernoulli likelihood.
        If True, use KL divergence (option 2). If False, use uniform potentials (option 1).
    beta_02 : float, default=1.0
        Prior rate parameter for must-link constraint weights (larger values forces all mustlink weights to be smaller).
    betabar_02 : float, default=1.0
        Prior rate parameter for cannot-link constraint weights (larger values forces all cannotlink weights to be smaller).
    dd_0 : float, default=0.1
        Feature-importance Beta prior hyperparameter (only when feature selection is enabled, higher values mean stronger prior of all unimportant features).
    warmup_prop : float, default=0.0
        Proportion of initial iterations to run with adaptive weights disabled
        (w_learn forced to False). This "staged training" allows cluster
        parameters and feature selection to stabilize before enabling
        adaptive constraint weighting. Change to 0.0 for w+FS only and 1.0 for 1+D+FS only.
    Attributes
    ----------
    N : int
        Number of samples (set after Inference is called).
    d : int
        Feature dimensionality (set after Inference is called).
    phi : ndarray of shape (N, K)
        Current soft assignments (set after Inference is called).
    params_0 : dict
        Prior hyperparameters for the likelihood family.
    global_stats : dict
        Variational parameters tracked by SVI (epsilon/gamma, m, L, etc.).
    feat_imp : ndarray, optional
        Feature importance scores (only when feature selection is enabled).
    elbo_ema : list[float]
        Smoothed ELBO history useful for diagnostics. Empty if ``speedup`` is True.
    
    Notes
    -----
    Constraint parameters (constraint_idx, constraint_val, constraints) are passed
    to the Inference() method rather than __init__ for memory efficiency.
    """
    def __init__(
        self,
        K,
        alpha=1.0,
        epsilon=1e-9,
        init="kmeans",
        weight_prior="Dirichelet distribution",
        aa=1.0,
        likelihood="bernoulli",
        batch_size=500,
        tau0=1.0,
        kappa_svi=0.7,
        elbo_ema_decay=0.9,
        w_learn=False,
        KL=False,
        beta_02=1.0,
        betabar_02=1.0,
        dd_0=0.1,
        warmup_prop=0.0,
    ):
        """Instantiate the HMRF-DPM model.

        See the class docstring for a complete description of the arguments and
        internal state that is created here.
        """
        self.K = K
        self.eps = epsilon
        self.likelihood = likelihood.lower()
        if self.likelihood not in {"gaussian", "bernoulli"}:
            raise ValueError("likelihood must be either 'gaussian' or 'bernoulli'")
        
        # Store hyperparameters
        self.alpha = alpha
        self.aa = aa
        self.beta_02 = beta_02
        self.betabar_02 = betabar_02
        self.dd_0 = dd_0
        self.init = init
        
        self.weight_prior = weight_prior
        self.batch_size = batch_size  # Store requested batch size
        self.tau0 = tau0
        self.kappa_svi = kappa_svi
        if not 0.0 < elbo_ema_decay < 1.0:
            raise ValueError("elbo_ema_decay must be in (0, 1)")
        self.elbo_ema_decay = elbo_ema_decay
        self.elbo_ema = []
        self.param_delta_trace = [None] # trace of parameter changes EMA
        self.param_delta_ema_decay = 0.8
        self.w_learn = w_learn
        self.KL = KL
        self.warmup_prop = warmup_prop
        self.feat_imp = None
        self.feat_imp_trace = []
        
        # These will be initialized in Inference() when X is provided
        self.N = None
        self.d = None
        self.phi = None
        self.params_0 = None
        self.global_stats = None
        self.mustlink_graph = None
        self.cannotlink_graph = None
        self.mask = None
        self.tuples_ml = None
        self.tuples_cl = None
        self.W = None
        self.Wbar = None


    def _initialize_priors(self, alpha=1.0, aa=1.0):
        """Create prior hyperparameters for the model.

        Parameters
        ----------
        alpha : float, default=1.0
            Dirichlet concentration prior; internally rescaled by K.
        aa : float, default=1.0
            Stick-breaking Beta(1, aa) prior hyperparameter.

        Returns
        -------
        params : dict[str, np.ndarray]
            Dictionary containing all prior hyperparameters.
        """
        alpha = alpha / self.K
        aa = aa / self.K
        epsilon_0 = alpha * np.ones((self.K, ))
        params: dict[str, np.ndarray] = {"epsilon_0": epsilon_0,"aa_0": aa}
        # weights for pairwise constraint potentials
        params.update({
            "beta_01": 1.0,
            "beta_02": self.beta_02,
            "betabar_01": 1.0,
            "betabar_02": self.betabar_02,
        })
        if self.likelihood == "gaussian":
            kappa_0 = np.ones((self.K, ))
            nu_0 = self.d * np.ones((self.K, ))
            m_0 = np.zeros((self.K, self.d))
            L_0 = np.eye(self.d).reshape(1, self.d, self.d) * np.ones((self.K, 1, 1))
            params.update({
                "kappa_0": kappa_0,
                "nu_0": nu_0,
                "m_0": m_0,
                "L_0": L_0,
            })
        elif self.likelihood == "bernoulli":
            alpha_0 = np.ones((self.K, self.d))
            beta_0 = np.ones((self.K, self.d))
            params.update({"alpha_0": alpha_0, "beta_0": beta_0})
        else:
            raise ValueError(f"Unsupported likelihood '{self.likelihood}'")
        return params

    def _initialize_phi(self, N):
        """Create a random, row-stochastic responsibility matrix."""
        logger.info("Random initialization of phi")
        phi = np.random.rand(N, self.K)
        phi = normalize(phi, axis=1)
        return phi

    def _compute_N(self, phi):
        """Compute effective counts $N_k = sum_n phi_{nk}$ for each cluster."""
        return np.sum(phi, 0)

    def _compute_gamma_1(self, N):
        """First stick-breaking variational parameter: $gamma_{1k} = 1 + N_k$."""
        gamma_1 = 1 + N
        return gamma_1

    def _compute_gamma_2(self, N):
        """Second stick-breaking parameter using the cumulative surplus counts."""
        gamma_2 = self.params_0["aa_0"] + cumsum_ex(N[::-1])[::-1]
        return gamma_2

    def _compute_epsilon(self, N):
        """Dirichlet-posterior parameter update: $epsilon_k = epsilon_{0k} + N_k$."""
        epsilon = self.params_0["epsilon_0"] + N
        return epsilon

    def _compute_nu(self, N): # for GMM
        """Wishart degrees-of-freedom posterior: $nu_k = nu_{0k} + N_k + 1$."""
        nu = self.params_0["nu_0"] + N + 1
        return nu

    def _compute_kappa(self, N): # for GMM
        """Gaussian mean precision update $kappa_k = kappa_{0k} + N_k$."""
        kappa = self.params_0["kappa_0"] + N
        return kappa

    def _compute_m(self, phi, N, X, target_idx=None): # for GMM
        """Posterior Gaussian means given responsibilities.

        Parameters
        ----------
        phi : ndarray of shape (batch, K)
            Responsibilities for the selected batch.
        N : ndarray of shape (K,)
            Effective counts for each component.
        X : ndarray of shape (N, d)
            Observation matrix.
        target_idx : array-like, optional
            Indices corresponding to ``phi``; defaults to the full dataset.

        Returns
        -------
        m : ndarray of shape (K, d)
            Updated mean parameters.
        """
        indices = np.arange(self.N) if target_idx is None else np.asarray(target_idx, dtype=int)
        scale = self.N / indices.size
        m = (self.params_0["kappa_0"].reshape(self.K, 1) * self.params_0["m_0"]
             + scale * np.sum(np.reshape(phi, [len(indices), self.K, 1]) * np.reshape(X[indices], [len(indices), 1, self.d]), axis=0) )\
            / (self.params_0["kappa_0"] + N ).reshape(self.K, 1)
        return m

    def _compute_L(self, phi, m, X, target_idx=None): # for GMM
        """Posterior precision matrices for the Gaussian components."""
        indices = np.arange(self.N) if target_idx is None else np.asarray(target_idx, dtype=int)
        scale = self.N / indices.size
        m_0 = np.reshape(self.params_0["m_0"], [self.K, self.d, 1])
        m =  np.reshape(m, [self.K, self.d, 1])
        L_inv = self.params_0["L_0"] + self.params_0["kappa_0"].reshape(self.K, 1, 1) * np.matmul(m - m_0, np.reshape(m - m_0,(self.K, 1, self.d))) \
                + scale * np.sum(np.reshape(phi, [len(indices), self.K, 1, 1]) * np.matmul(X[indices].reshape(len(indices), 1, self.d, 1) - m.reshape(1, self.K, self.d, 1),
                                                                             X[indices].reshape(len(indices), 1, 1, self.d) - m.reshape(1, self.K, 1, self.d) ) , axis=0)
        return LA.inv(L_inv)

    def _compute_bern_param_ab(self, phi, X, target_idx=None, feat_select=False):
        """Posterior Beta parameters for Bernoulli means."""
        feat_E = self.feat_imp if feat_select else 1.0
        indices = np.arange(self.N) if target_idx is None else np.asarray(target_idx, dtype=int)
        scale = self.N / indices.size
        ones = scale * feat_E * np.dot(phi.T, X[indices])
        zeros = scale * feat_E * np.dot(phi.T, 1 - X[indices])
        alpha_post = self.params_0["alpha_0"] + ones
        beta_post = self.params_0["beta_0"] + zeros
        alpha_post = np.maximum(alpha_post, self.eps)
        beta_post = np.maximum(beta_post, self.eps)
        return alpha_post, beta_post

    def _compute_popu_mean(self, X):
        """Empirical feature-wise mean used by optional feature selection."""
        popu_mean = np.mean(X, axis=0)
        return popu_mean

    def _compute_feat_imp_param(self, phi, alpha, beta_param, X, target_idx=None):
        """Update feature-importance Beta hyperparameters when enabled."""
        indices = np.arange(self.N) if target_idx is None else np.asarray(target_idx, dtype=int)
        scale = self.N / indices.size
        E_log = digamma(alpha) - digamma(alpha + beta_param)
        E_log_1 = digamma(beta_param) - digamma(alpha + beta_param)
        ll = (E_log - np.log(self.popu_mean + self.eps)) * np.dot(phi.T, X[indices]) + (E_log_1 - np.log(1 - self.popu_mean + self.eps)) * np.dot(phi.T, 1 - X[indices])
        s_c = np.where(ll > 0, ll, 0) # ReLu
        s_d = np.where(ll < 0, -ll, 0)
        #s_c = 1 / (1 + np.exp(-ll)) # logistic
        #s_d = np.exp(-ll) / (1 + np.exp(-ll))
        c_post = self.params_0["cc_0"] + scale * s_c
        d_post = self.params_0["dd_0"] + scale * s_d
        return c_post, d_post

    def _compute_constraint_weights(self, phi, Vm, Vc, target_idx=None, warmup=False):
        """Update Gamma rate parameters for batched pairwise constraint weights.
        
        Parameters
        ----------
        warmup : bool, default=False
            If True, return uniform weights regardless of w_learn setting.
            Used during warmup phase of staged training.
        """
        _ = {1:"W=1", 2:"adaptive W"}
        choose = 2 if (self.w_learn and not warmup) else 1
        indices = np.arange(self.N) if target_idx is None else np.asarray(target_idx, dtype=int)
        W = {}
        Wbar = {}
        # Update must-link weights (only beta_nm2)
        for n in indices:
            for m in self.mustlink_graph[n]:
                key = (min(n, m), max(n, m))
                if choose == 1:
                    ### All 1s, disable adaptive weights
                    W[key] = self.params_0["beta_01"] / (self.params_0["beta_02"] + self.eps)
                    continue
                kl_term = np.sum(phi[n, :].reshape(self.K, 1) * phi[m, :].reshape(1, self.K) * Vm)
                w_beta2 = self.params_0["beta_02"] + kl_term
                W[key] = self.params_0["beta_01"] / (w_beta2 + self.eps)
        # Update cannot-link weights (only betabar_nm2)
            for m in self.cannotlink_graph[n]:
                key = (min(n, m), max(n, m))
                if choose == 1:
                    ### All 1s, disable adaptive weights
                    Wbar[key] = self.params_0["betabar_01"] / (self.params_0["betabar_02"] + self.eps)
                    continue
                kl_term = np.sum(phi[n, :].reshape(self.K, 1) * phi[m, :].reshape(1, self.K) * Vc)
                wbar_beta2 = self.params_0["betabar_02"] + kl_term
                Wbar[key] = self.params_0["betabar_01"] / (wbar_beta2 + self.eps)
        return W, Wbar
    
    def _initialize_stats(self):
        """Deep-copy prior hyperparameters to seed the variational state."""
        stats = {}
        if self.weight_prior == "Dirichelet distribution":
            stats["epsilon"] = self.params_0["epsilon_0"].copy()
        else:
            stats["gamma_1"] = np.ones(self.K)
            stats["gamma_2"] = np.ones(self.K) * self.params_0["aa_0"]
        if self.likelihood == "gaussian":
            stats["nu"] = self.params_0["nu_0"].copy()
            stats["kappa"] = self.params_0["kappa_0"].copy()
            stats["m"] = self.params_0["m_0"].copy()
            stats["L"] = self.params_0["L_0"].copy()
        else:
            stats["alpha"] = self.params_0["alpha_0"].copy()
            stats["beta"] = self.params_0["beta_0"].copy()
        return stats

    def _step_size(self, iteration):
        """Robbins-Monro decrement $rho_t = (tau_0 + t)^{-kappa}$."""
        return (self.tau0 + iteration) ** (-self.kappa_svi)

    def _parameter_change(self, prev_stats, new_stats):
        """EMA-smoothed relative RMS change for each variational block."""
        total = 0.0
        blocks = 0
        for key in prev_stats:
            prev = np.asarray(prev_stats[key], dtype=float)
            current = np.asarray(new_stats[key], dtype=float)
            diff = current - prev
            block_size = diff.size
            diff_rms = np.sqrt(np.mean(diff * diff))
            prev_rms = np.sqrt(np.mean(prev * prev))
            rel_change = diff_rms / (prev_rms + self.eps)
            total += rel_change / block_size
            blocks += 1
        raw_change = 0.0 if blocks == 0 else self.N**0.25 * total / blocks**2 / 2
        prev_ema = self.param_delta_trace[-1]
        if prev_ema is None or not np.isfinite(prev_ema):
            ema_change = raw_change
        else:
            ema_change = self.param_delta_ema_decay * prev_ema + (1.0 - self.param_delta_ema_decay) * raw_change
        self.param_delta_trace.append(ema_change)
        return ema_change

    def _compute_phi_bernoulli(self, alpha, beta_param, epsilon, Vm, Vc, phi_t, gamma_1, gamma_2, X, target_idx=None, feat_select=False):
        """E-step for Bernoulli likelihoods incorporating constraint potentials."""
        if self.weight_prior == "Dirichelet distribution":
            val = digamma(epsilon) - digamma(np.sum(epsilon))
        else :
            val = digamma(gamma_1) - digamma(gamma_1 + gamma_2) + cumsum_ex(digamma(gamma_2) - digamma(gamma_1 + gamma_2))

        if feat_select:
            Elog = self.feat_imp * (digamma(alpha) - digamma(alpha + beta_param)) + (1 - self.feat_imp) * np.log(self.popu_mean + self.eps)
            Elog_1 = self.feat_imp * (digamma(beta_param) - digamma(alpha + beta_param)) + (1 - self.feat_imp) * np.log(1 - self.popu_mean + self.eps)
        else:
            Elog = digamma(alpha) - digamma(alpha + beta_param)
            Elog_1 = digamma(beta_param) - digamma(alpha + beta_param)
        
        indices = np.arange(self.N) if target_idx is None else np.asarray(target_idx, dtype=int)
        log_phi = np.zeros((indices.size, self.K))
        for row_idx, n in enumerate(indices):
            row_vals = np.zeros(self.K)
            if self.mask[n] == 0:
                row_vals = row_vals + val
            else:
                for j in self.mustlink_graph[n]:
                    w_nj = self.W.get((min(n, j), max(n, j)))
                    row_vals = row_vals - w_nj * np.sum(phi_t[j, :].reshape(1, self.K) * Vm, axis=1)
                for j in self.cannotlink_graph[n]:
                    wbar_nj = self.Wbar.get((min(n, j), max(n, j)))
                    row_vals = row_vals - wbar_nj * np.sum(phi_t[j, :].reshape(1, self.K) * Vc, axis=1)
            # likelihood term
            row_vals = row_vals + np.sum(Elog.reshape(self.K, self.d) * X[n, :].reshape(1, self.d), axis=1) \
                        + np.sum(Elog_1.reshape(self.K, self.d) * (1 - X[n, :].reshape(1, self.d)), axis=1)
            log_phi[row_idx, :] = row_vals

        log_phi = log_phi - _safe_logsumexp(log_phi, axis=1, keepdims=True)
        phi = _safe_exp(log_phi)
        return normalize(phi, axis=1)


    def _compute_phi_gaussian(self, m, L, epsilon, Vm, Vc, nu, phi_t, kappa, gamma_1, gamma_2, X, target_idx=None):
        """Gaussian counterpart of the variational E-step."""
        if self.weight_prior == "Dirichelet distribution":
            val = digamma(epsilon) - digamma(np.sum(epsilon))
        else :
            val = digamma(gamma_1) - digamma(gamma_1 + gamma_2) + cumsum_ex(digamma(gamma_2) - digamma(gamma_1 + gamma_2))

        indices = np.arange(self.N) if target_idx is None else np.asarray(target_idx, dtype=int)
        log_phi = np.zeros((indices.size, self.K))
        for row_idx, n in enumerate(indices):
            row_vals = np.zeros(self.K)
            if self.mask[n] == 0:
                row_vals = row_vals + val
            else:
                for j in self.mustlink_graph[n]:
                    w_nj = self.W.get((min(n, j), max(n, j)))
                    row_vals = row_vals - w_nj * np.sum(phi_t[j, :].reshape(1, self.K) * Vm, axis=1)
                for j in self.cannotlink_graph[n]:
                    wbar_nj = self.Wbar.get((min(n, j), max(n, j)))
                    row_vals = row_vals - wbar_nj * np.sum(phi_t[j, :].reshape(1, self.K) * Vc, axis=1)
            # likelihood term
            row_vals = row_vals - 0.5 * self.d/kappa - 0.5 * nu * np.trace(
                np.matmul(
                    L,
                    np.matmul(
                        (X[n, :].reshape(1, self.d) - m).reshape(self.K, self.d, 1),
                        (X[n, :].reshape(1, self.d) - m).reshape(self.K, 1, self.d),
                    ),
                ),
                axis1=1,
                axis2=2,
            ) + 0.5 * (LA.slogdet(L)[1] + multivar_digamma(nu, self.d)) - 0.5 * self.d * np.log(np.pi)
            log_phi[row_idx, :] = row_vals

        log_phi = log_phi - _safe_logsumexp(log_phi, axis=1, keepdims=True)
        phi = _safe_exp(log_phi)
        return normalize(phi, axis=1)


    def _compute_potentials_bernoulli(self, alpha, beta_param, feat_select=False) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Vm/Vc via symmetrised KL divergence for Bernoulli means."""
        _ = {1:"No KL/D", 2:"D", 3:"KL"}
        choose = 2 if self.KL else 1
        ### No KL divergence, all 1s
        if choose == 1:
            Vm = np.ones((self.K, self.K))
            Vc = np.eye(self.K)
            Vm = Vm - Vc
            return Vm, Vc
        betaln_vals = betaln(alpha, beta_param)
        dig_alpha = digamma(alpha)
        dig_beta = digamma(beta_param)
        dig_sum = digamma(alpha + beta_param)
        Elog = dig_alpha - dig_sum
        Elog_1 = dig_beta - dig_sum
        if choose == 2:
        ### conpute the KL between Bernoulli distributions using Expectations 
            E = alpha / (alpha + beta_param)
            if feat_select:
                E = self.feat_imp * E + (1 - self.feat_imp) * self.popu_mean
                Elog = self.feat_imp * (dig_alpha - dig_sum) + (1 - self.feat_imp) * np.log(self.popu_mean + self.eps)
                Elog_1 = self.feat_imp * (dig_beta - dig_sum) + (1 - self.feat_imp) * np.log(1 - self.popu_mean + self.eps)
            Elog_diff = Elog[:, np.newaxis, :] - Elog[np.newaxis, :, :]
            Elog_1_diff = Elog_1[:, np.newaxis, :] - Elog_1[np.newaxis, :, :]
            kl_forward = (Elog_diff * E[:, np.newaxis, :] + Elog_1_diff * (1 - E)[:, np.newaxis, :])
        else:
            alpha_diff = alpha[:, np.newaxis, :] - alpha[np.newaxis, :, :]
            beta_diff = beta_param[:, np.newaxis, :] - beta_param[np.newaxis, :, :]
            betaln_diff = betaln_vals[np.newaxis, :, :] - betaln_vals[:, np.newaxis, :]
            kl_forward = (betaln_diff + alpha_diff * Elog[:, np.newaxis, :] + beta_diff * Elog_1[:, np.newaxis, :])

        kl_sym = 0.5 * np.sum(kl_forward + np.swapaxes(kl_forward, 0, 1), axis=2)
        kl_max = np.max(kl_sym)
        Vm = kl_sym - np.diag(np.diag(kl_sym))
        Vc = np.diag([kl_max] * self.K)
        return Vm, Vc


    def _compute_potentials_gaussian(self, m, L, nu, kappa):
        """Pairwise KL potentials for Gaussian components (returns Vm, Vc)."""
        m = np.asarray(m)
        L = np.asarray(L)
        nu = np.asarray(nu).reshape(self.K)
        kappa = np.asarray(kappa).reshape(self.K)

        delta = m[:, None, :] - m[None, :, :]
        L_inv = LA.inv(L)
        # Quadratic terms from mean differences for every component pair.
        quad_from_k = np.einsum('kij,klj,kli->kl', L, delta, delta)
        quad_from_l = np.einsum('lij,klj,kli->kl', L, delta, delta)
        quad_term = nu[:, None] * quad_from_k + nu[None, :] * quad_from_l

        # Kappa contributions (symmetrized automatically via transpose).
        ratio = (
            self.d * (kappa[:, None] / (self.eps + kappa[None, :]) - 1.0)
            + np.log((kappa[:, None] + self.eps) / (self.eps + kappa[None, :]))
        )
        ratio_term = ratio + ratio.T

        log_det_L = np.log(self.eps + LA.det(L))
        multi_digamma_vals = np.array([multivar_digamma(val, self.d) for val in nu])
        diff_term = 0.5 * (nu[:, None] - nu[None, :]) * (
            (log_det_L[:, None] - log_det_L[None, :])
            + (multi_digamma_vals[:, None] - multi_digamma_vals[None, :])
        )

        trace_LinvL = np.einsum('lij,kji->kl', L_inv, L)
        trace_term = (
            0.5 * nu[:, None] * (trace_LinvL - self.d)
            + 0.5 * nu[None, :] * (trace_LinvL.T - self.d)
        )

        KL = ratio_term + quad_term + diff_term + trace_term
        KL = 0.5 * (KL + KL.T)  # enforce exact symmetry
        kl_max = np.max(KL)
        Vc = np.diag([kl_max] * self.K)
        Vm = KL - np.diag(np.diag(KL))
        return Vm, Vc

    def Inference(self, X, constraint_idx=None, constraint_val=None, constraints=None, max_iter=1000, rel_tol=1e-5, debug=False, speedup=True, feat_select=False):
        """Run stochastic variational inference until convergence.

        Parameters
        ----------
        X : ndarray of shape (N, d)
            Observation matrix.
        constraint_idx : ndarray, optional
            Flat upper-triangular indices of non-zero pairwise constraints.
            Memory-efficient alternative to dense ``constraints`` array.
        constraint_val : ndarray, optional
            Constraint values at ``constraint_idx``: +1 for must-link, -1 for cannot-link.
            Must be provided together with ``constraint_idx``.
        constraints : ndarray, optional (deprecated)
            Row-flattened upper-triangular indicator matrix encoding must-links (+1),
            cannot-links (-1) and unconstraints (0). Use ``constraint_idx``/``constraint_val``
            for better memory efficiency.
        max_iter : int, default=1000
            Hard iteration cap.
        rel_tol : float, default=1e-5
            Stopping tolerance based on relative parameter change (and EMA ELBO).
        debug : bool, default=False
            Print timing/diagnostic information per iteration when True.
        speedup : bool, default=True
            Skip ELBO computation and rely solely on parameter stability.
        feat_select : bool, default=False
            Activate feature-importance updates.

        Returns
        -------
        loss : list[float]
            Mini-batch ELBO trace (empty when ``speedup`` is True).
        """
        X = np.asarray(X, dtype=float)
        
        # Initialize data-dependent attributes
        self.N = X.shape[0]
        self.d = X.shape[1]
        self.batch_size = min(self.batch_size, self.N)
        
        # Build constraint graphs from sparse or dense input
        if constraints is None and constraint_idx is None and constraint_val is None:
            # No constraints provided - empty graphs
            self.mustlink_graph = {i: [] for i in range(self.N)}
            self.cannotlink_graph = {i: [] for i in range(self.N)}
            mask_must = np.zeros(self.N, dtype=int)
            mask_cannot = np.zeros(self.N, dtype=int)
        else:
            if constraints is not None or (constraint_val is not None and constraint_idx is not None):
                if constraint_idx is None and constraint_val is None and constraints is not None:
                    # Fallback to dense representation and extract non-zero entries
                    nonzero_idx = np.where(constraints != 0)[0]
                    constraint_idx, constraint_val = nonzero_idx.astype(np.int64), constraints[nonzero_idx].copy().astype(np.int8)
                # Use memory-efficient sparse representation
                self.mustlink_graph, mask_must = construct_mustlink_sparse(constraint_idx, constraint_val, self.N)
                self.cannotlink_graph, mask_cannot = construct_cannotlink_sparse(constraint_idx, constraint_val, self.N)
            else:
                raise ValueError("Both 'constraint_idx' and 'constraint_val' must be provided.")
        self.mask = mask_must | mask_cannot
        
        # Initialize priors (depends on d)
        self.params_0 = self._initialize_priors(alpha=self.alpha, aa=self.aa)
        
        # Initialize phi
        if self.init == "kmeans":
            self.phi, mu_0 = initialise_phi_with_kmeans(X, self.K, self.likelihood)
            if self.likelihood == "gaussian":
                self.params_0["m_0"] = mu_0
        else:
            self.phi = self._initialize_phi(self.N)
        
        # Initialize constraint tuples and weights
        self.tuples_ml = get_tuples(self.mustlink_graph)
        self.tuples_cl = get_tuples(self.cannotlink_graph)
        self.global_stats = self._initialize_stats()
        self.W = {(min(n, m), max(n, m)): self.params_0["beta_01"] / (self.params_0["beta_02"] + self.eps) for n, m in self.tuples_ml}
        self.Wbar = {(min(n, m), max(n, m)): self.params_0["betabar_01"] / (self.params_0["betabar_02"] + self.eps) for n, m in self.tuples_cl}
        
        if feat_select:
            logger.info("Feature selection mode activated.")
            c_0 = np.ones((self.K, self.d)) * 0.5
            if self.mask.sum() > 0:
                d_0 = np.ones((self.K, self.d)) * self.dd_0
            else:
                d_0 = np.ones((self.K, self.d)) * 0.1 # weak feature selection when no constraints
            self.params_0["cc_0"] = c_0
            self.params_0["dd_0"] = d_0
            self.global_stats["cc"] = c_0
            self.global_stats["dd"] = d_0
            self.popu_mean = self._compute_popu_mean(X)
            self.feat_imp = c_0 / (c_0 + d_0)  # Expectation: c/(c+d) dim(K,d)
            self.feat_imp_trace.append(self.feat_imp.copy())
        loss = []
        progress = tqdm(total=max_iter, desc="HMRF-DPM", unit="iter", leave=False)

        if self.w_learn and self.KL and feat_select:
            warm_iters = int(self.warmup_prop * max_iter)
        else:
            warm_iters = 0
        try:
            for i in range(max_iter):
                # Staged training: disable adaptive weights during warmup
                in_warmup = i < warm_iters
                if in_warmup and i == 0 and self.w_learn:
                    progress.write(f"[INFO] Warmup phase: w_learn disabled for first {warm_iters} iterations")
                    logger.info(f"Warmup phase: w_learn disabled for first {warm_iters} iterations")
                if not in_warmup and i == warm_iters and self.w_learn:
                    progress.write(f"[INFO] Warmup complete: w_learn enabled from iteration {i}")
                    logger.info(f"Warmup complete: w_learn enabled from iteration {i}")
                    self.KL = False if feat_select else self.KL  # disable KL divergence when feat_select is on
                
                if debug:
                    progress.write(f"[DEBUG] Iteration {i + 1}")

                rho = self._step_size(i)
                batch_idx, batch_phi = get_batch(X, self.phi, self.batch_size)
                M = self._compute_N(batch_phi)
                scale = self.N / batch_idx.size
                N_hat = scale * M

                updates = {}
                update_t0 = time.perf_counter()
                if self.likelihood == "gaussian":
                    updates["nu"] = self._compute_nu(N_hat)
                    updates["kappa"] = self._compute_kappa(N_hat)
                    updates["m"] = self._compute_m(batch_phi, N_hat, X, target_idx=batch_idx)
                    updates["L"] = self._compute_L(batch_phi, updates["m"], X, target_idx=batch_idx)
                else:
                    alpha_param, beta_param = self._compute_bern_param_ab(batch_phi, X, target_idx=batch_idx, feat_select=feat_select)
                    updates["alpha"] = alpha_param
                    updates["beta"] = beta_param

                if self.weight_prior == "Dirichelet distribution":
                    updates["epsilon"] = self._compute_epsilon(N_hat)
                else:
                    updates["gamma_1"] = self._compute_gamma_1(N_hat)
                    updates["gamma_2"] = self._compute_gamma_2(N_hat)

                prev_stats = {key: np.copy(self.global_stats[key]) for key in updates}
                blended = stochastic_update(updates, prev_stats, rho)
                self.global_stats.update(blended)
                epsilon = self.global_stats.get("epsilon")
                gamma_1 = self.global_stats.get("gamma_1")
                gamma_2 = self.global_stats.get("gamma_2")
                phi_t = np.copy(self.phi)

                update_time = time.perf_counter() - update_t0
                progress.set_postfix_str(f"upd {update_time:.2f}s", refresh=False)
                potential_assign_t0 = time.perf_counter()

                if self.likelihood == "gaussian":
                    Vm, Vc = self._compute_potentials_gaussian(self.global_stats["m"], self.global_stats["L"],self.global_stats["nu"],self.global_stats["kappa"])
                    W, Wbar = self._compute_constraint_weights(self.phi, Vm, Vc, target_idx=batch_idx, warmup=in_warmup)
                    self.W.update(W)
                    self.Wbar.update(Wbar)
                    batch_phi_new = self._compute_phi_gaussian(self.global_stats["m"], self.global_stats["L"], epsilon, Vm, Vc, self.global_stats["nu"], phi_t,self.global_stats["kappa"],gamma_1,gamma_2,X,target_idx=batch_idx)
                else:
                    Vm, Vc = self._compute_potentials_bernoulli(self.global_stats["alpha"],self.global_stats["beta"], feat_select=feat_select)
                    W, Wbar = self._compute_constraint_weights(self.phi, Vm, Vc, target_idx=batch_idx, warmup=in_warmup)
                    self.W.update(W)
                    self.Wbar.update(Wbar)
                    batch_phi_new = self._compute_phi_bernoulli(self.global_stats["alpha"],self.global_stats["beta"], epsilon,Vm,Vc,phi_t,gamma_1,gamma_2,X,target_idx=batch_idx, feat_select=feat_select)
                self.phi[batch_idx, :] = batch_phi_new

                if feat_select:
                    c_post, d_post = self._compute_feat_imp_param(batch_phi_new, self.global_stats["alpha"], self.global_stats["beta"], X, target_idx=batch_idx)
                    updates_fs = {"cc": c_post, "dd": d_post}
                    prev_fs_stats = {key: np.copy(self.global_stats[key]) for key in updates_fs}
                    prev_stats.update(prev_fs_stats)
                    blended_fs = stochastic_update(updates_fs, prev_fs_stats, rho)
                    blended.update(blended_fs)
                    self.global_stats.update(blended_fs)
                    self.feat_imp = blended_fs["cc"] / (blended_fs["cc"] + blended_fs["dd"])
                    self.feat_imp_trace.append(self.feat_imp.copy())

                potential_assign_time = time.perf_counter() - potential_assign_t0
                progress.set_postfix_str(
                    f"phi {potential_assign_time:.2f}s",
                    refresh=False,
                )
                elbo_t0 = time.perf_counter()

                if not speedup:
                    if self.likelihood == "gaussian":
                        l, log_likelihood_term, hmrf_term = self._compute_stochastic_elbo_gaussian(batch_idx,self.phi,epsilon,Vm,Vc,gamma_1,gamma_2,X)
                    else:
                        l, log_likelihood_term, hmrf_term = self._compute_stochastic_elbo_bernoulli(batch_idx,self.phi,epsilon,Vm,Vc,gamma_1,gamma_2,X, feat_select=feat_select)
                    elbo_time = time.perf_counter() - elbo_t0
                else:
                    l = None
                    log_likelihood_term = None
                    hmrf_term = None
                    elbo_time = 0.0
                progress.set_postfix_str(
                    f"elbo {elbo_time:.2f}s",
                    refresh=False,
                )

                param_delta = self._parameter_change(prev_stats, blended)
                if not speedup:
                    prev_ema = self.elbo_ema[-1] if self.elbo_ema else None
                    if prev_ema is None:
                        current_ema = l
                        self.elbo_ema.append(current_ema)
                        ema_change = None
                    else:
                        current_ema = self.elbo_ema_decay * prev_ema + (1.0 - self.elbo_ema_decay) * l
                        self.elbo_ema.append(current_ema)
                        ema_change = np.abs(current_ema - prev_ema) / (np.abs(prev_ema) + self.eps)
                else:
                    prev_ema = None
                    ema_change = None

                progress.set_postfix_str(
                    f"phi {potential_assign_time:.2f}s | Δ {param_delta:.2e}",
                    refresh=False,
                )
                progress.update(1)

                if debug:
                    if not speedup:
                        progress.write(f"- stochastic elbo is  {l}")
                        progress.write(f"- log_likelihood_term is  {log_likelihood_term}")
                        progress.write(f"- hmrf_term is  {hmrf_term}")
                        if prev_ema is not None:
                            progress.write(f"- smoothed elbo is  {current_ema}")
                            progress.write(f"- smoothed elbo change is  {ema_change}")
                    progress.write(f"- parameter delta is  {param_delta}")
                    progress.write(
                        "[TIMING] updates {:.3f}s | potentials/phi {:.3f}s |  elbo {:.3f}s".format(
                            update_time,
                            potential_assign_time,
                            elbo_time,
                        )
                    )
                    if speedup:
                        progress.write("- speedup mode active: ELBO skipped; convergence via parameter delta only")

                if not speedup:
                    loss.append(l)

                smoothed_stable = ema_change is None or ema_change < rel_tol
                if param_delta < rel_tol and smoothed_stable and i > 0:
                    progress.write(f"[INFO] Converged at iteration  {i}")
                    logger.info(f'Converged at iteration {i}')
                    break
                if i == max_iter - 1:
                    progress.write(f"[INFO] Reached max ({max_iter}) iterations without convergence")
                    logger.info(f'Reached max ({max_iter}) iterations without convergence')
        finally:
            progress.close()
        return loss


    def _compute_stochastic_elbo_bernoulli(self, batch_idx, phi, epsilon, Vm, Vc, gamma_1, gamma_2, X, feat_select=False):
        """Mini-batch estimator of the Bernoulli ELBO."""
        scale = self.N / len(batch_idx)
        hmrf_term = 0.0
        log_likelihood_term = 0.0
        alpha = self.global_stats["alpha"]
        beta_param = self.global_stats["beta"]

        if self.weight_prior == "Dirichelet distribution":
            val = digamma(epsilon) - digamma(np.sum(epsilon))
        else:
            val = digamma(gamma_1) - digamma(gamma_1 + gamma_2) + cumsum_ex(
                digamma(gamma_2) - digamma(gamma_1 + gamma_2)
            )

        if feat_select:
            Elog = self.feat_imp * (digamma(alpha) - digamma(alpha + beta_param)) + (1 - self.feat_imp) * np.log(self.popu_mean + self.eps)
            Elog_1 = self.feat_imp * (digamma(beta_param) - digamma(alpha + beta_param)) + (1 - self.feat_imp) * np.log(1 - self.popu_mean + self.eps)
        else:
            Elog = digamma(alpha) - digamma(alpha + beta_param)
            Elog_1 = digamma(beta_param) - digamma(alpha + beta_param)
        entropy_z = 0.0
        for n in batch_idx:
            phi_n = phi[n, :]
            logp_n = (
                np.sum(
                    (Elog).reshape(self.K, self.d) * X[n, :].reshape(1, self.d),
                    axis=1,
                )
                + np.sum(
                    (Elog_1).reshape(self.K, self.d) * (1 - X[n, :].reshape(1, self.d)),
                    axis=1,
                )
            )
            log_likelihood_term += scale * np.dot(phi_n, logp_n)
            if self.mask[n] == 0:
                log_likelihood_term += scale * np.dot(phi_n, val)
            entropy_z -= np.dot(phi_n, np.log(phi_n + self.eps))

        log_likelihood_term += scale * entropy_z

        batch_set = set(batch_idx.tolist())
        batch_mustlink = {idx: neighbors for idx, neighbors in self.mustlink_graph.items() if idx in batch_set}
        batch_cannotlink = {idx: neighbors for idx, neighbors in self.cannotlink_graph.items() if idx in batch_set}
        batch_ml = get_tuples(batch_mustlink)
        batch_cl = get_tuples(batch_cannotlink)

        if batch_ml and self.tuples_ml:
            ml_energy = 0.0
            for i, j in batch_ml:
                w_ij = self.W.get((min(i, j), max(i, j)))
                ml_energy += w_ij * np.dot(phi[i], Vm @ phi[j])
            hmrf_term += -ml_energy * (len(self.tuples_ml) / len(batch_ml))
        if batch_cl and self.tuples_cl:
            cl_energy = 0.0
            for i, j in batch_cl:
                wbar_ij = self.Wbar.get((min(i, j), max(i, j)))
                cl_energy += wbar_ij * np.dot(phi[i], Vc @ phi[j])
            hmrf_term += -cl_energy * (len(self.tuples_cl) / len(batch_cl))

        alpha_0 = self.params_0["alpha_0"]
        beta_0 = self.params_0["beta_0"]
        beta_kl = (
            betaln(alpha_0, beta_0) - betaln(alpha, beta_param)
            + (alpha - alpha_0) * (digamma(alpha) - digamma(alpha + beta_param))
            + (beta_param - beta_0) * (digamma(beta_param) - digamma(alpha + beta_param))
        )
        log_likelihood_term -= np.sum(beta_kl)

        if feat_select:
            c_0 = self.params_0["cc_0"]
            d_0 = self.params_0["dd_0"]
            cc = self.global_stats["cc"]
            dd = self.global_stats["dd"]
            beta_fs_kl = (
                betaln(c_0, d_0) - betaln(cc, dd)
                + (cc - c_0) * (digamma(cc) - digamma(cc + dd))
                + (dd - d_0) * (digamma(dd) - digamma(cc + dd))
            )
            log_likelihood_term -= np.sum(beta_fs_kl)

        if self.weight_prior != "Dirichelet distribution":
            for k in range(self.K):
                log_likelihood_term += beta(gamma_1[k], gamma_2[k]).entropy()
        else:
            log_likelihood_term += dirichlet(epsilon).entropy()

        elbo = log_likelihood_term + hmrf_term
        return elbo / self.N, log_likelihood_term, hmrf_term


    def _compute_stochastic_elbo_gaussian(self, batch_idx, phi, epsilon, Vm, Vc, gamma_1, gamma_2, X):
        """Mini-batch estimator of the Gaussian ELBO using Normal-Wishart expectations."""
        batch_idx = np.asarray(batch_idx, dtype=int)
        batch_phi = phi[batch_idx, :]
        scale = self.N / batch_idx.size

        nu = self.global_stats["nu"]
        kappa = self.global_stats["kappa"]
        m = self.global_stats["m"]
        L = self.global_stats["L"]

        # Expected log-determinants under q(Lambda_k) with guardrails for numerical sign flips.
        logdet_L = np.empty(self.K)
        for k in range(self.K):
            sign, logabs = LA.slogdet(L[k])
            logdet_L[k] = logabs if sign > 0 else np.log(self.eps)
        e_log_det = multivar_digamma(nu, self.d) + logdet_L

        # Data likelihood term E_q[log p(X|Z, mu, Lambda)].
        x_batch = X[batch_idx]
        diff = x_batch[:, None, :] - m[None, :, :]
        quad = np.einsum('bkd,kde,bke->bk', diff, L, diff)
        log_norm = 0.5 * (e_log_det - self.d / (self.eps + kappa))
        log_components = log_norm.reshape(1, self.K) - 0.5 * quad * nu.reshape(1, self.K)
        log_likelihood_term = scale * np.sum(batch_phi * log_components)

        # Mixture weight expectations E_q[log p(z|pi)].
        if self.weight_prior == "Dirichelet distribution":
            val = digamma(epsilon) - digamma(np.sum(epsilon))
        else:
            val = digamma(gamma_1) - digamma(gamma_1 + gamma_2) + cumsum_ex(
                digamma(gamma_2) - digamma(gamma_1 + gamma_2)
            )
        unconstrained = self.mask[batch_idx] == 0
        if np.any(unconstrained):
            weight_term = np.sum(batch_phi[unconstrained] * val.reshape(1, self.K), axis=1)
            log_likelihood_term += scale * np.sum(weight_term)

        # Entropy of q(Z).
        entropy_z = -np.sum(batch_phi * np.log(batch_phi + self.eps))
        log_likelihood_term += scale * entropy_z

        # Constraint contributions scaled to full graph.
        batch_set = set(batch_idx.tolist())
        batch_mustlink = {idx: nbrs for idx, nbrs in self.mustlink_graph.items() if idx in batch_set}
        batch_cannotlink = {idx: nbrs for idx, nbrs in self.cannotlink_graph.items() if idx in batch_set}
        batch_ml = get_tuples(batch_mustlink)
        batch_cl = get_tuples(batch_cannotlink)

        hmrf_term = 0.0
        if batch_ml and self.tuples_ml:
            ml_energy = 0.0
            for i, j in batch_ml:
                w_ij = self.W.get((min(i, j), max(i, j)))
                ml_energy += w_ij * np.dot(phi[i], Vm @ phi[j])
            hmrf_term += -ml_energy * (len(self.tuples_ml) / len(batch_ml))
        if batch_cl and self.tuples_cl:
            cl_energy = 0.0
            for i, j in batch_cl:
                wbar_ij = self.Wbar.get((min(i, j), max(i, j)))
                cl_energy += wbar_ij * np.dot(phi[i], Vc @ phi[j])
            hmrf_term += -cl_energy * (len(self.tuples_cl) / len(batch_cl))

        # Prior/entropy terms for mixture weights with built-in entropies.
        if self.weight_prior == "Dirichelet distribution":
            epsilon_0 = self.params_0["epsilon_0"]
            e_log_pi = val
            prior_weights = np.sum((epsilon_0 - 1.0) * e_log_pi)
            entropy_weights = dirichlet(epsilon).entropy()
        else:
            e_log_one_minus_v = digamma(gamma_2) - digamma(gamma_1 + gamma_2)
            prior_weights = (self.params_0["aa_0"] - 1.0) * np.sum(e_log_one_minus_v)
            entropy_weights = 0.0
            for k in range(self.K):
                entropy_weights += beta(gamma_1[k], gamma_2[k]).entropy()

        # Normal-Wishart prior vs posterior terms.
        diff_means = m - self.params_0["m_0"]
        quad_prior = np.einsum('kd,kde,ke->k', diff_means, L, diff_means)
        mean_term = self.d / (self.eps + kappa) + nu * quad_prior
        expected_precision = nu[:, None, None] * L
        trace_prior = np.einsum('kij,kji->k', self.params_0["L_0"], expected_precision)
        log_p_mu_lambda = np.sum(
            0.5 * self.d * np.log(self.params_0["kappa_0"] + self.eps)
            + 0.5 * e_log_det
            - 0.5 * self.params_0["kappa_0"] * mean_term
            + 0.5 * (self.params_0["nu_0"] - self.d - 1.0) * e_log_det
            - 0.5 * trace_prior
        )
        log_q_mu_lambda = np.sum(
            0.5 * self.d * np.log(self.eps + kappa)
            + 0.5 * e_log_det
            - 0.5 * self.d
            + 0.5 * (nu - self.d - 1.0) * e_log_det
            - 0.5 * nu * self.d
        )

        log_prior_terms = (prior_weights + entropy_weights) + (log_p_mu_lambda - log_q_mu_lambda)
        log_terms_total = log_likelihood_term + log_prior_terms
        total_elbo = log_terms_total + hmrf_term
        return total_elbo / self.N, log_terms_total, hmrf_term

    def infer_clusters(self):
        """Hard-assign each sample to its MAP cluster."""
        return np.argmax(self.phi, axis=1)

    def _predict_prob_bernoulli(self, X_new: np.ndarray, mustLink: dict | None = None, cannotLink: dict | None = None) -> np.ndarray:
        """Compute cluster assignment probabilities for new data points (Bernoulli likelihood)."""
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim != 2 or X_new.shape[1] != self.d:
            raise ValueError(f"X_new must be a 2-D array with {self.d} features.")

        if self.weight_prior == "Dirichelet distribution":
            epsilon = self.global_stats["epsilon"]
            E_logpi = digamma(epsilon) - digamma(np.sum(epsilon))
        else :
            gamma_1 = self.global_stats["gamma_1"]
            gamma_2 = self.global_stats["gamma_2"]
            E_logpi = digamma(gamma_1) - digamma(gamma_1 + gamma_2) + cumsum_ex(digamma(gamma_2) - digamma(gamma_1 + gamma_2))

        alpha = self.global_stats["alpha"]
        beta_param = self.global_stats["beta"]
        if self.feat_imp is not None:
            Elog = self.feat_imp * (digamma(alpha) - digamma(alpha + beta_param)) + (1 - self.feat_imp) * np.log(self.popu_mean + self.eps)
            Elog_1 = self.feat_imp * (digamma(beta_param) - digamma(alpha + beta_param)) + (1 - self.feat_imp) * np.log(1 - self.popu_mean + self.eps)
        else:
            Elog = digamma(alpha) - digamma(alpha + beta_param)
            Elog_1 = digamma(beta_param) - digamma(alpha + beta_param)

        log_phi_new = np.zeros((X_new.shape[0], self.K), dtype=float)
        for n in range(X_new.shape[0]):
            logp_n = (
                np.sum(
                    (Elog).reshape(self.K, self.d) * X_new[n, :].reshape(1, self.d),
                    axis=1,
                )
                + np.sum(
                    (Elog_1).reshape(self.K, self.d) * (1 - X_new[n, :].reshape(1, self.d)),
                    axis=1,
                )
            )
            log_phi_new[n, :] = logp_n + E_logpi

            if mustLink is not None and n in mustLink:
                for idx in mustLink[n]:
                    if idx < self.N:
                        log_phi_new[n, :] += np.log(self.phi[idx, :] + self.eps)
            if cannotLink is not None and n in cannotLink:
                for idx in cannotLink[n]:
                    if idx < self.N:
                        log_phi_new[n, :] += np.log(1.0 - self.phi[idx, :] + self.eps)

        log_phi_new -= _safe_logsumexp(log_phi_new, axis=1, keepdims=True)
        phi_new = _safe_exp(log_phi_new)
        return normalize(phi_new, axis=1)

    def _predict_prob_gaussian(self, X_new: np.ndarray, mustLink: dict | None = None, cannotLink: dict | None = None) -> np.ndarray:
        """Compute cluster assignment probabilities for new data points (Gaussian likelihood)."""
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim != 2 or X_new.shape[1] != self.d:
            raise ValueError(f"X_new must be a 2-D array with {self.d} features.")

        nu = self.global_stats["nu"]
        kappa = self.global_stats["kappa"]
        m = self.global_stats["m"]
        L = self.global_stats["L"]

        logdet_L = np.empty(self.K)
        for k in range(self.K):
            sign, logabs = LA.slogdet(L[k])
            logdet_L[k] = logabs if sign > 0 else np.log(self.eps)
        e_log_det = multivar_digamma(nu, self.d) + logdet_L

        if self.weight_prior == "Dirichelet distribution":
            epsilon = self.global_stats["epsilon"]
            E_logpi = digamma(epsilon) - digamma(np.sum(epsilon))
        else:
            gamma_1 = self.global_stats["gamma_1"]
            gamma_2 = self.global_stats["gamma_2"]
            E_logpi = digamma(gamma_1) - digamma(gamma_1 + gamma_2) + cumsum_ex(digamma(gamma_2) - digamma(gamma_1 + gamma_2))

        log_phi_new = np.zeros((X_new.shape[0], self.K), dtype=float)
        for n in range(X_new.shape[0]):
            diff = X_new[n, :] - m
            quad = np.einsum('kd,kde,ke->k', diff, L, diff)
            log_norm = 0.5 * (e_log_det - self.d / (self.eps + kappa))
            log_components = log_norm - 0.5 * quad * nu
            log_phi_new[n, :] = log_components + E_logpi

            if mustLink is not None and n in mustLink:
                for idx in mustLink[n]:
                    if idx < self.N:
                        log_phi_new[n, :] += np.log(self.phi[idx, :] + self.eps)
            if cannotLink is not None and n in cannotLink:
                for idx in cannotLink[n]:
                    if idx < self.N:
                        log_phi_new[n, :] += np.log(1.0 - self.phi[idx, :] + self.eps)

        log_phi_new -= _safe_logsumexp(log_phi_new, axis=1, keepdims=True)
        phi_new = _safe_exp(log_phi_new)
        return normalize(phi_new, axis=1)

    def predict_probability(self, X_new: np.ndarray, mustLink: dict | None = None, cannotLink: dict | None = None) -> np.ndarray:
        """Compute cluster assignment probabilities for new data points.

        Parameters
        ----------
        X_new : np.ndarray, shape (n_samples, n_features)
            New data points to predict.
        mustLink : dict, optional
            Must-link constraints for new data points {N_new_idx: list of indices of training data points}.
        cannotLink : dict, optional
            Cannot-link constraints for new data points {N_new_idx: list of indices of training data points}.

        Returns
        -------
        phi_new : np.ndarray, shape (n_samples, n_clusters)
            Cluster assignment probabilities for each new data point.
        """
        if self.likelihood == "gaussian":
            return self._predict_prob_gaussian(X_new, mustLink=mustLink, cannotLink=cannotLink)
        else:
            return self._predict_prob_bernoulli(X_new, mustLink=mustLink, cannotLink=cannotLink)

    def predict(self, X_new: np.ndarray, mustLink: dict | None = None, cannotLink: dict | None = None) -> np.ndarray:
        """Hard-assign each new sample to its MAP cluster.

        Parameters
        ----------
        X_new : np.ndarray, shape (n_samples, n_features)
            New data points to predict.
        mustLink : dict, optional
            Must-link constraints for new data points {N_new_idx: list of indices of training data points}.
        cannotLink : dict, optional
            Cannot-link constraints for new data points {N_new_idx: list of indices of training data points}.

        Returns
        -------
        cluster_assignments : np.ndarray, shape (n_samples,)
            Hard cluster assignments for each new data point.
        """
        phi_new = self.predict_probability(X_new, mustLink=mustLink, cannotLink=cannotLink)
        return np.argmax(phi_new, axis=1)


    def get_constraint_weights(self, top: int | None = None) -> Tuple[dict[Tuple[int, int], float], dict[Tuple[int, int], float]]:
        """Retrieve the ordered top learned constraint weights for must-link and cannot-link pairs.

        Returns
        -------
        W : dict
            Learned sorted must-link weights {(i, j): weight}.
        Wbar : dict
            Learned sorted cannot-link weights {(i, j): weight}.
        """
        sorted_W = sorted(self.W.items(), key=lambda item: item[1], reverse=True)
        sorted_Wbar = sorted(self.Wbar.items(), key=lambda item: item[1], reverse=True)
        if top is not None:
            return dict(sorted_W[:top]), dict(sorted_Wbar[:top])
        else:
            return dict(sorted_W), dict(sorted_Wbar)

    def plot_param_change(self, ax: plt.Axes | None = None, save_path: str | Path | None = None, show: bool | None = None, title: str | None = None, tick_interval: int = 20) -> plt.Axes:
        """Plot parameter change values per iteration with x-ticks every ``tick_interval`` steps."""

        changes = np.asarray(self.param_change_trace, dtype=float)
        if changes.ndim != 1:
            raise ValueError("param_changes must be a 1-D sequence of scalars")
        if changes.size != 0:
            iterations = np.arange(1, changes.size + 1)
            created_fig = False
            if ax is None:
                fig, ax = plt.subplots(figsize=(7, 4))
                created_fig = True
            else:
                fig = ax.figure

            ax.plot(iterations, changes, linewidth=1.6, color="tab:orange")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Delta (log scale)")
            ax.set_yscale("log")
            ax.set_title(title if title is not None else "Parameter Change (Relative RMS EMA)")
            ax.grid(True, linestyle="--", alpha=0.3)

            if tick_interval <= 0:
                raise ValueError("tick_interval must be positive")
            max_tick = ((iterations[-1] - 1) // tick_interval + 1) * tick_interval
            ticks = np.arange(tick_interval, max_tick + 1, tick_interval)
            ticks = np.concatenate(([0], ticks)) if ticks.size else np.array([0])
            ax.set_xticks(ticks)

            if save_path is not None:
                fig.savefig(Path(save_path), dpi=250, bbox_inches="tight")

            should_show = show if show is not None else created_fig
            if should_show:
                plt.show()
        return ax

    def plot_elbo(
        self,
        elbos: list[float] | np.ndarray | None = None,
        ax: plt.Axes | None = None,
        save_path: str | Path | None = None,
        show: bool | None = None,
        title: str | None = None,
        tick_interval: int = 20,
    ) -> plt.Axes:
        """Plot Stochastic (EMA) ELBO values per iteration with x-ticks every ``tick_interval`` steps."""

        if elbos is None:
            elbo_values = np.asarray(self.elbo_ema, dtype=float)
        else:
            elbo_values = np.asarray(elbos, dtype=float)
        if elbo_values.ndim != 1:
            raise ValueError("elbos must be a 1-D sequence of scalars")
        if elbo_values.size != 0:
            iterations = np.arange(1, elbo_values.size + 1)
            created_fig = False
            if ax is None:
                fig, ax = plt.subplots(figsize=(7, 4))
                created_fig = True
            else:
                fig = ax.figure

            ax.plot(iterations, elbo_values, linewidth=1.6, color="tab:blue")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("ELBO")
            ax.set_title(title if title is not None else "Stochastic ELBO")
            ax.grid(True, linestyle="--", alpha=0.3)

            if tick_interval <= 0:
                raise ValueError("tick_interval must be positive")
            max_tick = ((iterations[-1] - 1) // tick_interval + 1) * tick_interval
            ticks = np.arange(tick_interval, max_tick + 1, tick_interval)
            ticks = np.concatenate(([0], ticks)) if ticks.size else np.array([0])
            ax.set_xticks(ticks)

            if save_path is not None:
                fig.savefig(Path(save_path), dpi=250, bbox_inches="tight")

            should_show = show if show is not None else created_fig
            if should_show:
                plt.show()
        return ax

    def plot_feature_imp(
        self,
        cov_name: list[str] | None = None,
        cluster_name: list[str] | None = None,
        feature_order: np.ndarray | list[int] | None = None,
        ax: plt.Axes | None = None,
        title: str | None = None,
        cmap: str = "Reds",
        fontsize: float = 12,
        bar_offset: float = 0.45,
        bar_fraction: float = 0.04,
        value_format: str = ".2f",
        show: bool | None = None,
        save_path: str | Path | None = None,
    ) -> plt.Axes:
        """Plot a heatmap of feature importances with optional custom axis labels.

        Parameters
        ----------
        cov_name : list[str], optional
            Custom names for features (columns). Must match number of features.
        cluster_name : list[str], optional
            Custom names for clusters (rows). Must match number of clusters.
        feature_order : array-like, optional
            Index array to reorder features (columns) in the heatmap.
            For example, [2, 0, 1] will place feature 2 first, then 0, then 1.
        ax : plt.Axes, optional
            Matplotlib axes to plot on. If None, creates a new figure.
        title : str, optional
            Title for the heatmap.
        cmap : str, default="Reds"
            Colormap for the heatmap.
        fontsize : float, default=12
            Base font size for labels.
        bar_offset : float, default=0.45
            Offset for colorbar label.
        bar_fraction : float, default=0.04
            Fraction of axes for colorbar width.
        value_format : str, default=".2f"
            Format string for cell values.
        show : bool, optional
            Whether to display the plot.
        save_path : str or Path, optional
            Path to save the figure.

        Returns
        -------
        ax : plt.Axes
            The matplotlib axes containing the plot.
        """
        if self.feat_imp is None:
            raise RuntimeError("Feature importances are unavailable. Enable feature selection in inference first.")

        feat_imp = np.asarray(self.feat_imp, dtype=float)
        if feat_imp.ndim != 2:
            raise ValueError("self.feat_imp must be a 2-D array.")

        num_clusters, num_features = feat_imp.shape

        # Apply feature reordering if provided
        if feature_order is not None:
            feature_order = np.asarray(feature_order, dtype=int)
            if feature_order.size != num_features:
                raise ValueError(f"feature_order length ({feature_order.size}) must match number of features ({num_features}).")
            feat_imp = feat_imp[:, feature_order]

        cov_name = (
            cov_name if cov_name is not None else [f"X{i + 1}" for i in range(num_features)]
        )
        cluster_name = (
            cluster_name if cluster_name is not None else [f"Cluster {i + 1}" for i in range(num_clusters)]
        )
        if len(cov_name) != num_features:
            raise ValueError("cov_name length must match the number of columns in self.feat_imp.")
        if len(cluster_name) != num_clusters:
            raise ValueError("cluster_name length must match the number of rows in self.feat_imp.")

        # Reorder cov_name if feature_order is provided
        if feature_order is not None:
            cov_name = [cov_name[i] for i in feature_order]

        created_fig = False
        if ax is None:
            fig_width = max(6.0, num_features * 0.6)
            fig_height = max(4.0, num_clusters * 0.5)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            created_fig = True
        else:
            fig = ax.figure

        im = ax.imshow(feat_imp, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(np.arange(num_features))
        ax.set_yticks(np.arange(num_clusters))
        xtick_kwargs = {"rotation": 45, "ha": "right"}
        xtick_kwargs["fontsize"] = fontsize
        ax.set_xticklabels(cov_name, **xtick_kwargs)
        ax.set_yticklabels(cluster_name, fontsize=fontsize)
        if title:
            ax.set_title(title, fontsize=fontsize + 2)

        norm = im.norm
        cbar = fig.colorbar(im, ax=ax, fraction = bar_fraction, location="top", orientation="horizontal")
        cbar.set_ticks(np.linspace(0.0, 1.0, num=6))
        cbar.ax.tick_params(labelsize=fontsize - 2)
        cbar.set_label("Importance", size=fontsize, loc="left")
        cbar.ax.xaxis.set_label_coords(-bar_offset, 0.)

        for row in range(num_clusters):
            for col in range(num_features):
                value = feat_imp[row, col]
                color = "white" if norm(value) > 0.6 else "black"
                ax.text(col, row, format(value, value_format), ha="center", va="center", fontsize=fontsize - 2, color=color)

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(Path(save_path), dpi=250, bbox_inches="tight")

        should_show = show if show is not None else created_fig
        if should_show:
            plt.show()
        elif created_fig:
            plt.close(fig)
        return ax


