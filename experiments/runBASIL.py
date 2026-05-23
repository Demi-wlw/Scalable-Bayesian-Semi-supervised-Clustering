"""Single entry point for reproducing BASIL DIGITS / MNIST experiments.

Examples
--------
DIGITS at 20% supervision with the (1, D+FS) configuration of Table 1::

    python -m experiments.runBASIL --dataset digits --supervision 0.20 --seed 1234

MNIST at 50% supervision::

    python -m experiments.runBASIL --dataset mnist  --supervision 0.50 --seed 1234

Adaptive constraint weighting (noisy regime, drop KL, keep Potts + FS)::

    python -m experiments.runBASIL --dataset digits --supervision 0.20 \\
        --w-learn --no-KL --seed 1234

Multiple seeds in parallel::

    python -m experiments.runBASIL --dataset digits --runs 10 --seed 1234 \\
        --max-workers 4 --output digits20_results.pkl
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import BASIL  # noqa: E402
from model.utils import cluster_acc  # noqa: E402
from experiments.data import (  # noqa: E402
    load_digits_binarized,
    load_mnist_binarized,
    sample_constraints,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("runBASIL")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """Hyperparameters shared across all parallel runs."""
    dataset: str
    supervision: float
    K: int
    batch_size: int
    max_iter: int
    rel_tol: float
    w_learn: bool
    KL: bool
    feat_select: bool
    wprior: float
    fsprior: float
    warmup_prop: float
    init: str
    max_pairs: int = 5_000_000

    X_train: np.ndarray = field(default=None, repr=False)
    y_train: np.ndarray = field(default=None, repr=False)
    X_test: np.ndarray = field(default=None, repr=False)
    y_test: np.ndarray = field(default=None, repr=False)


_CONTEXT: RunConfig | None = None  # populated inside each worker process


def _init_worker(cfg: RunConfig) -> None:
    """Cache the shared run configuration inside each worker."""
    global _CONTEXT
    _CONTEXT = cfg


# ---------------------------------------------------------------------------
# Single-seed runner
# ---------------------------------------------------------------------------

def _run_single(seed: int, cfg: RunConfig | None = None) -> dict[str, Any]:
    """Run a single BASIL fit at the given random seed and return metrics."""
    cfg = cfg or _CONTEXT
    if cfg is None:
        raise RuntimeError("Worker invoked without an initialised configuration")

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    c_idx, c_val = sample_constraints(
        cfg.y_train, cfg.supervision, rng, max_pairs=cfg.max_pairs
    )

    model = BASIL(
        K=cfg.K,
        likelihood="bernoulli",
        batch_size=cfg.batch_size,
        init=cfg.init,
        w_learn=cfg.w_learn,
        KL=cfg.KL,
        beta_02=cfg.wprior,
        betabar_02=cfg.wprior,
        dd_0=cfg.fsprior,
        warmup_prop=cfg.warmup_prop,
    )

    t0 = time.perf_counter()
    model.Inference(
        cfg.X_train,
        constraint_idx=c_idx,
        constraint_val=c_val,
        max_iter=cfg.max_iter,
        rel_tol=cfg.rel_tol,
        feat_select=cfg.feat_select,
    )
    elapsed = time.perf_counter() - t0

    y_tr_pred = model.predict(cfg.X_train)
    y_te_pred = model.predict(cfg.X_test)
    train_acc, _ = cluster_acc(y_tr_pred, cfg.y_train)
    test_acc, _ = cluster_acc(y_te_pred, cfg.y_test)

    return {
        "seed": seed,
        "n_ML": int((c_val == 1).sum()),
        "n_CL": int((c_val == -1).sum()),
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
        "elapsed_sec": float(elapsed),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_dataset(name: str) -> dict[str, np.ndarray]:
    """Load DIGITS or MNIST with a canonical (fixed) train/test split.

    The split does not depend on the user's run seed: identical test sets are
    used across multi-seed sweeps so that only model initialisation and
    constraint sampling vary.
    """
    name = name.lower()
    if name == "digits":
        return load_digits_binarized()  # default split_seed=42
    if name == "mnist":
        return load_mnist_binarized()
    raise ValueError(f"Unknown dataset '{name}'. Choose 'digits' or 'mnist'.")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(cfg: RunConfig, seeds: list[int], max_workers: int | None) -> list[dict[str, Any]]:
    """Execute BASIL fits for each seed, optionally in parallel."""
    requested = max_workers if max_workers is not None else (os.cpu_count() or 1)

    if requested > 1 and len(seeds) > 1:
        workers = min(requested, len(seeds))
        results_map: dict[int, dict[str, Any]] = {}
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(cfg,),
        ) as pool:
            futures = {pool.submit(_run_single, s): s for s in seeds}
            for fut in as_completed(futures):
                s = futures[fut]
                results_map[s] = fut.result()
        results = [results_map[s] for s in sorted(results_map)]
    else:
        results = [_run_single(s, cfg) for s in seeds]

    return sorted(results, key=lambda r: r["seed"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reproduce BASIL DIGITS / MNIST results from the ICML 2026 paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", choices=("digits", "mnist"), required=True)
    p.add_argument("--supervision", type=float, default=0.20,
                   help="Fraction of training samples labelled.")
    p.add_argument("--K", type=int, default=10, help="Number of clusters.")
    p.add_argument("--seed", type=int, default=1234, help="Base random seed.")
    p.add_argument("--runs", type=int, default=1,
                   help="Number of seeds to evaluate (seeds = base + 0..runs-1).")
    p.add_argument("--max-workers", type=int, default=None,
                   help="Parallel workers (default: all CPU cores).")

    p.add_argument("--max-iter", type=int, default=None,
                   help="Max SVI iterations (default: 2000 for DIGITS, 3000 for MNIST).")
    p.add_argument("--rel-tol", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--init", type=str, default="kmeans",
                   choices=("kmeans", "random"),
                   help="Phi initialisation strategy. 'kmeans' dispatches to K-medoids "
                        "(small Bernoulli N) or K-modes (large Bernoulli N) automatically.")
    p.add_argument("--max-pairs", type=int, default=5_000_000,
                   help="Cap on the number of pairwise constraints sampled.")

    # Model configuration knobs
    p.add_argument("--w-learn", action="store_true",
                   help="Enable adaptive constraint weighting (default: fixed w=1).")
    p.add_argument("--no-KL", action="store_true",
                   help="Disable the cluster-divergence KL potential (use Potts).")
    p.add_argument("--no-FS", action="store_true",
                   help="Disable feature selection.")
    p.add_argument("--wprior", type=float, default=1.0,
                   help="Gamma-prior rate for constraint weights (beta_02 / betabar_02).")
    p.add_argument("--fsprior", type=float, default=0.1,
                   help="Beta-prior rate for feature relevance (dd_0).")
    p.add_argument("--warmup-prop", type=float, default=0.0,
                   help="Fraction of iterations under fixed weights before switching to adaptive.")

    p.add_argument("--output", type=Path, default=None,
                   help="Optional path to a .pkl file collecting per-seed results.")
    return p


def _default_max_iter(dataset: str, user_max_iter: int | None) -> int:
    if user_max_iter is not None:
        return user_max_iter
    return 3000 if dataset == "mnist" else 2000


def _default_init(dataset: str, user_init: str | None) -> str:
    """Return ``user_init`` if provided, else ``'kmeans'`` (smart Bernoulli dispatcher)."""
    return user_init or "kmeans"


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_argparser().parse_args(argv)

    dataset = args.dataset
    logger.info(f"[{dataset.upper()}] loading data")
    data = _load_dataset(dataset)
    logger.info(f"  train: {data['X_train'].shape}, test: {data['X_test'].shape}")

    cfg = RunConfig(
        dataset=dataset,
        supervision=args.supervision,
        K=args.K,
        batch_size=args.batch_size,
        max_iter=_default_max_iter(dataset, args.max_iter),
        rel_tol=args.rel_tol,
        w_learn=args.w_learn,
        KL=not args.no_KL,
        feat_select=not args.no_FS,
        wprior=args.wprior,
        fsprior=args.fsprior,
        warmup_prop=args.warmup_prop,
        init=_default_init(dataset, args.init),
        max_pairs=args.max_pairs,
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_test=data["X_test"],
        y_test=data["y_test"],
    )

    seeds = [args.seed + i for i in range(args.runs)]
    logger.info(f"[{dataset.upper()}] running {len(seeds)} seed(s): {seeds}")
    logger.info(f"  supervision={cfg.supervision:.0%}, K={cfg.K}, batch={cfg.batch_size}, "
                f"init={cfg.init}, max_iter={cfg.max_iter}")
    logger.info(f"  KL={cfg.KL}, FS={cfg.feat_select}, w_learn={cfg.w_learn}, "
                f"wprior={cfg.wprior}, fsprior={cfg.fsprior}, warmup_prop={cfg.warmup_prop}")

    results = run(cfg, seeds, args.max_workers)

    # Aggregate across seeds
    test_accs = np.array([r["test_acc"] for r in results])
    train_accs = np.array([r["train_acc"] for r in results])
    times = np.array([r["elapsed_sec"] for r in results])

    logger.info(f"[{dataset.upper()}] aggregated over {len(results)} seed(s)")
    logger.info(f"  Train ACC: {train_accs.mean():.3f} +/- {train_accs.std():.3f}")
    logger.info(f"  Test  ACC: {test_accs.mean():.3f} +/- {test_accs.std():.3f}")
    if dataset == "mnist":
        logger.info(f"  Time:      {times.mean()/3600:.2f} +/- {times.std()/3600:.2f} hr")
    else:
        logger.info(f"  Time:      {times.mean()/60:.2f} +/- {times.std()/60:.2f} min")

    output = {
        "config": vars(args),
        "results": results,
        "summary": {
            "train_acc_mean": float(train_accs.mean()),
            "train_acc_std": float(train_accs.std()),
            "test_acc_mean": float(test_accs.mean()),
            "test_acc_std": float(test_accs.std()),
            "time_sec_mean": float(times.mean()),
            "time_sec_std": float(times.std()),
        },
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("wb") as fh:
            pickle.dump(output, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"[SAVED] {args.output}")

    return output


if __name__ == "__main__":
    main()
