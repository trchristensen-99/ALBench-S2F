#!/usr/bin/env python
"""RayTune HP sweep for LegNet scaling curves.

For each (strategy, training_size), runs a Bayesian HP search over:
  - learning_rate: loguniform(1e-4, 1e-2)
  - batch_size: choice(256, 512, 1024, 2048)
  - weight_decay: loguniform(1e-6, 1e-3)

Uses ASHA scheduler for early stopping of bad trials.
Reports best config and test metrics.

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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_and_eval(config, strategy, n_train, seed, pool_base_dir, output_base):
    """Train LegNet with given config and return val pearson."""
    from scripts.generate_labeled_pools import load_pool_subset

    # Load data from pool
    pool_path = Path(pool_base_dir) / strategy / "pool.npz"
    seqs, labels = load_pool_subset(pool_path, n_train, seed=seed)

    # Split into train/val (90/10)
    rng = np.random.default_rng(seed)
    n_val = max(1000, int(0.1 * n_train))
    perm = rng.permutation(len(seqs))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_seqs = [seqs[i] for i in train_idx]
    train_labels = labels[train_idx]
    val_seqs = [seqs[i] for i in val_idx]
    val_labels = labels[val_idx]

    # Train
    from models.legnet_student import LegNetStudent

    model = LegNetStudent(ensemble_size=1)
    result = model.train(
        train_seqs,
        train_labels,
        val_seqs=val_seqs,
        val_labels=val_labels,
        learning_rate=config["lr"],
        batch_size=config["batch_size"],
        weight_decay=config.get("weight_decay", 1e-5),
        epochs=80,
        early_stop_patience=10,
        verbose=False,
    )

    # Report to Ray
    import ray.train

    ray.train.report({"val_pearson": result["best_val_pearson"]})


def run_raytune(args):
    """Run RayTune HP search for one (strategy, size) combo."""
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    ray.init(num_cpus=args.cpus, num_gpus=1, log_to_driver=False)

    search_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([256, 512, 1024, 2048]),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
    }

    scheduler = ASHAScheduler(
        max_t=80,  # max epochs
        grace_period=5,
        reduction_factor=3,
    )

    # Determine pool dir
    pool_2m = REPO / f"outputs/labeled_pools_2m/k562/ag_s2"
    pool_618k = REPO / f"outputs/labeled_pools/k562/ag_s2"
    pool_dir = str(pool_2m) if (pool_2m / args.strategy / "pool.npz").exists() else str(pool_618k)

    trainable = tune.with_parameters(
        train_and_eval,
        strategy=args.strategy,
        n_train=args.size,
        seed=args.seed,
        pool_base_dir=pool_dir,
        output_base=str(REPO / "outputs"),
    )

    # Use GPU
    trainable = tune.with_resources(trainable, {"gpu": 1})

    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="val_pearson",
            mode="max",
            num_samples=args.n_trials,
            scheduler=scheduler,
        ),
        run_config=ray.train.RunConfig(
            name=f"legnet_{args.strategy}_n{args.size}_s{args.seed}",
            storage_path=str(REPO / "outputs" / "raytune_results"),
        ),
    )

    results = tuner.fit()
    best = results.get_best_result("val_pearson", "max")

    print(f"\n=== Best config for {args.strategy} n={args.size} ===")
    print(f"  Config: {best.config}")
    print(f"  Val Pearson: {best.metrics['val_pearson']:.4f}")

    # Save best config
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
