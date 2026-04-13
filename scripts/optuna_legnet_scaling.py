#!/usr/bin/env python
"""Lightweight Optuna HP search for LegNet scaling curves.

Unlike the RayTune version, this runs trials sequentially in a single
process — no Ray overhead, no worker spawning, no directory packaging.
Much faster for small-N jobs where training takes seconds.

For each (strategy, training_size), runs Optuna TPE search over:
  - learning_rate: loguniform(5e-5, 5e-2)
  - batch_size: choice(128, 256, 512, 1024, 2048, 4096)
  - weight_decay: loguniform(1e-7, 1e-2)

Then runs 3 replicates with the best HP.

Usage:
    uv run --no-sync python scripts/optuna_legnet_scaling.py \
        --strategy random --size 10000 --n-trials 10 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

os.environ["TORCHDYNAMO_DISABLE"] = "1"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pool_data(strategy, n_train, seed):
    """Load sequences and labels from pre-generated pool."""
    pool_2m = REPO / f"outputs/labeled_pools_2m/k562/ag_s2/{strategy}/pool.npz"
    pool_618k = REPO / f"outputs/labeled_pools/k562/ag_s2/{strategy}/pool.npz"
    pool_path = pool_2m if pool_2m.exists() else pool_618k

    data = np.load(pool_path, allow_pickle=True)
    all_seqs = data["sequences"]
    all_labels = data["labels"]
    pool_size = len(all_seqs)

    if n_train > pool_size:
        raise ValueError(f"n_train={n_train} > pool_size={pool_size}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(pool_size)
    idx = perm[:n_train]
    return [str(all_seqs[i]) for i in idx], all_labels[idx].astype(np.float32)


def train_and_evaluate(seqs, labels, lr, batch_size, weight_decay, seed):
    """Train LegNet and return val pearson."""
    from models.legnet_student import LegNetStudent

    # Train/val split
    rng = np.random.default_rng(seed + 1000)
    n_val = max(500, int(0.1 * len(seqs)))
    perm = rng.permutation(len(seqs))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_seqs = [seqs[i] for i in train_idx]
    train_labels = labels[train_idx]
    val_seqs = [seqs[i] for i in val_idx]
    val_labels = labels[val_idx]

    from models.legnet_student import TrainConfig

    config = TrainConfig(
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        epochs=80,
        early_stopping_patience=10,
    )
    model = LegNetStudent(ensemble_size=1, train_config=config)
    model.fit(
        sequences=train_seqs,
        labels=train_labels,
        val_sequences=val_seqs,
        val_labels=val_labels,
    )
    # Get val pearson from the model's prediction
    preds = model.predict(val_seqs)
    from scipy.stats import pearsonr

    val_r, _ = pearsonr(val_labels, preds)
    return float(val_r)


def run_optuna_search(args):
    """Run Optuna HP search then 3 replicates."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Load data once
    logger.info(f"Loading pool data: {args.strategy} n={args.size} seed={args.seed}")
    seqs, labels = load_pool_data(args.strategy, args.size, args.seed)
    logger.info(f"Loaded {len(seqs)} sequences")

    def objective(trial):
        lr = trial.suggest_float("lr", 5e-5, 5e-2, log=True)
        bs = trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048, 4096])
        wd = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
        val_r = train_and_evaluate(seqs, labels, lr, bs, wd, args.seed)
        return val_r

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed)
    )
    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial
    logger.info(f"Best trial: val={best.value:.4f} params={best.params}")

    # Save best config
    best_dir = REPO / "outputs" / "optuna_best" / args.strategy / f"n{args.size}"
    best_dir.mkdir(parents=True, exist_ok=True)
    with open(best_dir / f"best_config_seed{args.seed}.json", "w") as f:
        json.dump(
            {
                "config": best.params,
                "val_pearson": best.value,
                "strategy": args.strategy,
                "n_train": args.size,
                "seed": args.seed,
                "n_trials": args.n_trials,
            },
            f,
            indent=2,
        )

    # Run 3 replicates with best HP using exp1_1_scaling
    logger.info("Running 3 replicates with best HP...")
    pool_2m = REPO / "outputs/labeled_pools_2m/k562/ag_s2"
    pool_618k = REPO / "outputs/labeled_pools/k562/ag_s2"
    pool_dir = str(pool_2m) if (pool_2m / args.strategy / "pool.npz").exists() else str(pool_618k)

    out_dir = str(REPO / "outputs" / "exp1_1_optuna_final" / "k562" / "legnet_ag_s2")

    for rep_seed in [42, 1042, 2042]:
        result_path = (
            Path(out_dir)
            / args.strategy
            / f"n{args.size}"
            / "hp0"
            / f"seed{rep_seed}"
            / "result.json"
        )
        if result_path.exists():
            logger.info(f"  Skip seed={rep_seed} (exists)")
            continue
        logger.info(f"  Replicate seed={rep_seed}")
        os.system(
            f"uv run --no-sync python experiments/exp1_1_scaling.py "
            f"--task k562 --student legnet --oracle ag_s2 "
            f"--reservoir {args.strategy} "
            f"--pool-base-dir {pool_dir} "
            f"--n-replicates 1 --seed {rep_seed} "
            f"--output-dir {out_dir} "
            f"--training-sizes {args.size} "
            f"--chr-split --lr {best.params['lr']} --batch-size {best.params['batch_size']} "
            f"--epochs 80 --ensemble-size 1 --early-stop-patience 10"
        )

    logger.info("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=10)
    args = parser.parse_args()
    run_optuna_search(args)


if __name__ == "__main__":
    main()
