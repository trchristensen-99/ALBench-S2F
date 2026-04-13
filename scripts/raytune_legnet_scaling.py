#!/usr/bin/env python
"""RayTune HP sweep for LegNet scaling curves.

For each (strategy, training_size), runs Bayesian HP optimization over:
  - learning_rate: loguniform(1e-4, 1e-2)
  - batch_size: choice(256, 512, 1024, 2048)
  - weight_decay: loguniform(1e-6, 1e-3)

Uses ASHA scheduler for early stopping of bad trials.
Uses Optuna search algorithm for Bayesian optimization.

Usage:
    uv run --no-sync python scripts/raytune_legnet_scaling.py \
        --strategy random --size 10000 --n-trials 20 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pool_data(strategy, n_train, seed, repo_path):
    """Load sequences and labels from pre-generated pool."""
    pool_2m = Path(repo_path) / f"outputs/labeled_pools_2m/k562/ag_s2/{strategy}/pool.npz"
    pool_618k = Path(repo_path) / f"outputs/labeled_pools/k562/ag_s2/{strategy}/pool.npz"
    pool_path = pool_2m if pool_2m.exists() else pool_618k

    data = np.load(pool_path, allow_pickle=True)
    all_seqs = data["sequences"]
    all_labels = data["labels"]
    pool_size = len(all_seqs)

    if n_train > pool_size:
        raise ValueError(f"n_train={n_train} > pool_size={pool_size}")

    # Deterministic subset
    rng = np.random.default_rng(seed)
    perm = rng.permutation(pool_size)
    idx = perm[:n_train]
    return [str(all_seqs[i]) for i in idx], all_labels[idx].astype(np.float32)


def train_legnet(config, strategy, n_train, seed, repo_path):
    """Train LegNet with given HP config and report val pearson to Ray."""
    import sys

    sys.path.insert(0, repo_path)

    import ray.train

    from models.legnet_student import LegNetStudent

    # Load data
    seqs, labels = load_pool_data(strategy, n_train, seed, repo_path)

    # Train/val split (90/10)
    rng = np.random.default_rng(seed + 1000)
    n_val = max(500, int(0.1 * len(seqs)))
    perm = rng.permutation(len(seqs))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_seqs = [seqs[i] for i in train_idx]
    train_labels = labels[train_idx]
    val_seqs = [seqs[i] for i in val_idx]
    val_labels = labels[val_idx]

    # Train LegNet
    model = LegNetStudent(ensemble_size=1)
    result = model.train(
        sequences=train_seqs,
        labels=train_labels,
        val_sequences=val_seqs,
        val_labels=val_labels,
        learning_rate=config["lr"],
        batch_size=config["batch_size"],
        weight_decay=config.get("weight_decay", 1e-5),
        epochs=80,
        early_stop_patience=10,
        verbose=False,
    )

    ray.train.report({"val_pearson": result["best_val_pearson"]})


def run_raytune(args):
    """Run RayTune HP search for one (strategy, size) combo."""
    import tempfile

    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch

    tmp_dir = tempfile.mkdtemp(prefix="raytune_")
    ray.init(
        num_cpus=args.cpus,
        num_gpus=1,
        log_to_driver=False,
        runtime_env={"working_dir": tmp_dir},
    )

    search_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([256, 512, 1024, 2048]),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
    }

    scheduler = ASHAScheduler(
        max_t=80,
        grace_period=5,
        reduction_factor=3,
    )

    # Optuna for Bayesian optimization
    search_alg = OptunaSearch(metric="val_pearson", mode="max")

    trainable = tune.with_parameters(
        train_legnet,
        strategy=args.strategy,
        n_train=args.size,
        seed=args.seed,
        repo_path=str(REPO),
    )
    trainable = tune.with_resources(trainable, {"gpu": 0.5})  # allow 2 trials per GPU

    storage_path = str(REPO / "outputs" / "raytune_results")

    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="val_pearson",
            mode="max",
            num_samples=args.n_trials,
            scheduler=scheduler,
            search_alg=search_alg,
        ),
        run_config=ray.train.RunConfig(
            name=f"legnet_{args.strategy}_n{args.size}_s{args.seed}",
            storage_path=storage_path,
            verbose=1,
        ),
    )

    results = tuner.fit()
    best = results.get_best_result("val_pearson", "max")

    print(f"\n=== Best config for {args.strategy} n={args.size} ===")
    print(f"  Config: {best.config}")
    print(f"  Val Pearson: {best.metrics['val_pearson']:.4f}")

    # Save
    out_dir = REPO / "outputs" / "raytune_best" / args.strategy / f"n{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"best_config_seed{args.seed}.json", "w") as f:
        json.dump(
            {
                "config": best.config,
                "val_pearson": best.metrics["val_pearson"],
                "strategy": args.strategy,
                "n_train": args.size,
                "seed": args.seed,
                "n_trials": args.n_trials,
            },
            f,
            indent=2,
        )

    ray.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--cpus", type=int, default=8)
    args = parser.parse_args()

    run_raytune(args)


if __name__ == "__main__":
    main()
