#!/usr/bin/env python
"""Exhaustive HP search method comparison.

Tests many Optuna sampler configurations at large N (50K, 100K, 200K)
across multiple strategies and iterations to find the most robust and
consistent HP optimization approach.

For each method, runs 3 independent iterations (different data subsamples)
and measures:
  1. Best val found per iteration
  2. Consistency of HP configs across iterations
  3. Cross-evaluation: best HP from each iteration tested on all 3 subsamples

The goal is to find a method that CONSISTENTLY identifies the same near-optimal
HP, guaranteeing monotonic scaling curves even when the data subsample changes.

Methods tested:
  TPE family (7 variants):
    tpe_10_cold        — 10 trials, no warm-start (baseline)
    tpe_20_cold        — 20 trials, no warm-start
    tpe_20_warm        — 20 trials, 4 warm-start configs
    tpe_30_warm        — 30 trials, warm-start
    tpe_50_warm        — 50 trials, warm-start (exhaustive)
    tpe_20_multi_warm  — 20 trials, multivariate=True, warm-start
    tpe_20_warm_narrow — 20 trials, warm-start, narrow LR: [5e-4, 1e-2]

  CMA-ES family (2 variants):
    cma_20_warm        — 20 trials, warm-start
    cma_30_warm        — 30 trials, warm-start

  Random/QMC baselines (2 variants):
    random_20_warm     — 20 trials, warm-start
    random_50_warm     — 50 trials, warm-start

  Ensemble approach (1 variant):
    ensemble_3x20      — 3×TPE-20-warm, cross-eval top-3 from each

  GP-based (1 variant, graceful fallback):
    gp_20_warm         — 20 trials, GPSampler, warm-start

Total: 14 methods × 3 strategies × 3 sizes = 126 jobs

Usage:
    python scripts/exhaustive_hp_search.py \
        --method tpe_20_warm --strategy random --size 50000
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# 3 iteration seeds for consistency measurement
ITER_SEEDS = [42, 1042, 2042]

# Standard warm-start configs (known-good from prior runs)
WARM_CONFIGS = [
    {"lr": 0.001, "batch_size": 512, "weight_decay": 1e-5},
    {"lr": 0.005, "batch_size": 1024, "weight_decay": 1e-5},
    {"lr": 0.002, "batch_size": 256, "weight_decay": 0.004},
    {"lr": 0.003, "batch_size": 512, "weight_decay": 1e-6},
]


def get_neighbor_configs(strategy, n_train):
    """Load best configs from neighboring sizes (if available)."""
    configs = []
    for neighbor_n in [n_train // 2, n_train * 2]:
        nf = (
            REPO
            / "outputs"
            / "optuna_best"
            / strategy
            / f"n{neighbor_n}"
            / "best_config_seed42.json"
        )
        if nf.exists():
            try:
                nc = json.loads(nf.read_text())["config"]
                configs.append(nc)
            except Exception:
                pass
    return configs


def make_objective(seqs, labels, seed, lr_range=(5e-5, 5e-2)):
    """Create an Optuna objective function."""
    from scripts.optuna_legnet_scaling import get_chr_val, train_and_evaluate

    _ = get_chr_val()

    def objective(trial):
        lr = trial.suggest_float("lr", lr_range[0], lr_range[1], log=True)
        bs = trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048, 4096])
        wd = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
        return train_and_evaluate(seqs, labels, lr, bs, wd, seed)

    return objective


def run_single_search(method, seqs, labels, seed, strategy, n_train):
    """Run one search iteration and return (best_val, best_params, all_trials)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Parse method into sampler + config
    use_warm_start = "warm" in method
    lr_range = (5e-5, 5e-2)  # default

    if method == "tpe_20_warm_narrow":
        lr_range = (5e-4, 1e-2)

    objective = make_objective(seqs, labels, seed, lr_range=lr_range)

    # Determine sampler and n_trials
    if method == "tpe_10_cold":
        sampler = optuna.samplers.TPESampler(seed=seed)
        n_trials = 10
        use_warm_start = False
    elif method == "tpe_20_cold":
        sampler = optuna.samplers.TPESampler(seed=seed)
        n_trials = 20
        use_warm_start = False
    elif method == "tpe_20_warm":
        sampler = optuna.samplers.TPESampler(seed=seed)
        n_trials = 20
    elif method == "tpe_30_warm":
        sampler = optuna.samplers.TPESampler(seed=seed)
        n_trials = 30
    elif method == "tpe_50_warm":
        sampler = optuna.samplers.TPESampler(seed=seed)
        n_trials = 50
    elif method == "tpe_20_multi_warm":
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
        n_trials = 20
    elif method == "tpe_20_warm_narrow":
        sampler = optuna.samplers.TPESampler(seed=seed)
        n_trials = 20
    elif method == "cma_20_warm":
        sampler = optuna.samplers.CmaEsSampler(seed=seed)
        n_trials = 20
    elif method == "cma_30_warm":
        sampler = optuna.samplers.CmaEsSampler(seed=seed)
        n_trials = 30
    elif method == "random_20_warm":
        sampler = optuna.samplers.RandomSampler(seed=seed)
        n_trials = 20
    elif method == "random_50_warm":
        sampler = optuna.samplers.RandomSampler(seed=seed)
        n_trials = 50
    elif method == "gp_20_warm":
        try:
            sampler = optuna.samplers.GPSampler(seed=seed)
        except AttributeError:
            logger.warning("GPSampler not available, falling back to TPE")
            sampler = optuna.samplers.TPESampler(seed=seed)
        n_trials = 20
    else:
        raise ValueError(f"Unknown method: {method}")

    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Enqueue warm-start configs
    if use_warm_start:
        all_warm = WARM_CONFIGS + get_neighbor_configs(strategy, n_train)
        for wc in all_warm:
            # Adjust LR to be within the search range
            wc_adj = dict(wc)
            wc_adj["lr"] = max(lr_range[0], min(lr_range[1], wc_adj["lr"]))
            study.enqueue_trial(wc_adj)

    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    all_trials = [
        {"val": t.value, "params": t.params, "number": t.number}
        for t in study.trials
        if t.value is not None
    ]
    return best.value, best.params, all_trials


def run_ensemble(seqs_per_seed, labels_per_seed, strategy, n_train):
    """Ensemble approach: 3×TPE-20-warm, cross-eval top-3 from each."""
    from scripts.optuna_legnet_scaling import train_and_evaluate

    all_candidates = []

    # Phase 1: 3 independent TPE-20 searches
    for seed in ITER_SEEDS:
        _, best_params, all_trials = run_single_search(
            "tpe_20_warm",
            seqs_per_seed[seed],
            labels_per_seed[seed],
            seed,
            strategy,
            n_train,
        )
        # Collect top-3 from this run
        sorted_trials = sorted(
            [t for t in all_trials if t["val"] is not None],
            key=lambda t: -t["val"],
        )
        for t in sorted_trials[:3]:
            all_candidates.append(t["params"])
        logger.info(f"  Ensemble seed {seed}: best={sorted_trials[0]['val']:.4f}")

    # Phase 2: Cross-evaluate all candidates on all 3 subsamples
    logger.info(f"  Cross-evaluating {len(all_candidates)} candidates on 3 subsamples...")
    hp_scores = []
    for hp in all_candidates:
        scores = []
        for eval_seed in ITER_SEEDS:
            val = train_and_evaluate(
                seqs_per_seed[eval_seed],
                labels_per_seed[eval_seed],
                hp["lr"],
                hp["batch_size"],
                hp["weight_decay"],
                eval_seed,
            )
            scores.append(val)
        hp_scores.append((np.mean(scores), np.std(scores), hp, scores))

    # Phase 3: Pick consensus HP
    hp_scores.sort(key=lambda x: -x[0])
    best_mean, best_std, best_hp, best_scores = hp_scores[0]

    return {
        "consensus_hp": best_hp,
        "consensus_mean_val": float(best_mean),
        "consensus_std_val": float(best_std),
        "per_subsample_val": [float(s) for s in best_scores],
        "n_candidates_tested": len(all_candidates),
        "top_5": [{"mean": float(m), "std": float(s), "hp": h} for m, s, h, _ in hp_scores[:5]],
    }


def compute_consistency_metrics(iter_results):
    """Compute how consistent the HP configs are across iterations."""
    best_vals = [r["best_val"] for r in iter_results]
    best_lrs = [r["best_params"]["lr"] for r in iter_results]
    best_bss = [r["best_params"]["batch_size"] for r in iter_results]
    best_wds = [r["best_params"]["weight_decay"] for r in iter_results]

    # LR consistency: ratio of max/min LR across iterations
    lr_ratio = max(best_lrs) / max(min(best_lrs), 1e-10)

    # BS consistency: fraction of iterations that agree on BS
    from collections import Counter

    bs_counts = Counter(best_bss)
    bs_agreement = bs_counts.most_common(1)[0][1] / len(best_bss)

    # Val consistency
    val_mean = float(np.mean(best_vals))
    val_std = float(np.std(best_vals))
    val_cv = val_std / max(abs(val_mean), 1e-10)

    return {
        "val_mean": val_mean,
        "val_std": val_std,
        "val_cv": val_cv,
        "val_per_iter": [float(v) for v in best_vals],
        "lr_ratio": float(lr_ratio),
        "lr_per_iter": [float(lr) for lr in best_lrs],
        "bs_agreement": float(bs_agreement),
        "bs_per_iter": [int(bs) for bs in best_bss],
        "wd_per_iter": [float(wd) for wd in best_wds],
    }


def cross_evaluate_hp_configs(iter_results, seqs_per_seed, labels_per_seed):
    """Test each iteration's best HP on ALL subsamples.

    This measures whether the HP generalizes across data subsamples,
    not just whether it works on the subsample it was optimized for.
    """
    from scripts.optuna_legnet_scaling import train_and_evaluate

    cross_results = []
    for i, res in enumerate(iter_results):
        hp = res["best_params"]
        scores = []
        for eval_seed in ITER_SEEDS:
            val = train_and_evaluate(
                seqs_per_seed[eval_seed],
                labels_per_seed[eval_seed],
                hp["lr"],
                hp["batch_size"],
                hp["weight_decay"],
                eval_seed,
            )
            scores.append(float(val))
        cross_results.append(
            {
                "hp": hp,
                "original_val": float(res["best_val"]),
                "cross_eval_mean": float(np.mean(scores)),
                "cross_eval_std": float(np.std(scores)),
                "cross_eval_per_seed": scores,
            }
        )
    return cross_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "tpe_10_cold",
            "tpe_20_cold",
            "tpe_20_warm",
            "tpe_30_warm",
            "tpe_50_warm",
            "tpe_20_multi_warm",
            "tpe_20_warm_narrow",
            "cma_20_warm",
            "cma_30_warm",
            "random_20_warm",
            "random_50_warm",
            "gp_20_warm",
            "ensemble_3x20",
        ],
    )
    parser.add_argument("--strategy", required=True, choices=["random", "genomic", "motif_grammar"])
    parser.add_argument("--size", type=int, required=True, choices=[50000, 100000, 200000])
    args = parser.parse_args()

    out_dir = REPO / "outputs" / "exhaustive_hp_search" / args.strategy / f"n{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.method}.json"

    if out_file.exists():
        logger.info(f"SKIP: {out_file} already exists")
        return

    from scripts.optuna_legnet_scaling import load_pool_data

    logger.info(f"=== {args.method} | {args.strategy} n={args.size} ===")

    # Pre-load data for all 3 iteration seeds
    seqs_per_seed = {}
    labels_per_seed = {}
    for seed in ITER_SEEDS:
        seqs, labels = load_pool_data(args.strategy, args.size, seed)
        seqs_per_seed[seed] = seqs
        labels_per_seed[seed] = labels
        logger.info(f"  Loaded subsample seed={seed}: {len(seqs)} sequences")

    # Handle ensemble separately
    if args.method == "ensemble_3x20":
        logger.info("Running ensemble approach (3×TPE-20-warm + cross-eval)...")
        ensemble_result = run_ensemble(seqs_per_seed, labels_per_seed, args.strategy, args.size)
        result = {
            "method": args.method,
            "strategy": args.strategy,
            "n_train": args.size,
            "ensemble": ensemble_result,
        }
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(
            f"  Consensus val: {ensemble_result['consensus_mean_val']:.4f}"
            f" ± {ensemble_result['consensus_std_val']:.4f}"
        )
        logger.info(f"Saved: {out_file}")
        return

    # Run 3 iterations with different data subsamples
    iter_results = []
    for seed in ITER_SEEDS:
        logger.info(f"  Iteration seed={seed}...")
        best_val, best_params, all_trials = run_single_search(
            args.method,
            seqs_per_seed[seed],
            labels_per_seed[seed],
            seed,
            args.strategy,
            args.size,
        )
        logger.info(
            f"    best_val={best_val:.4f} lr={best_params['lr']:.5f} bs={best_params['batch_size']}"
        )
        iter_results.append(
            {
                "seed": seed,
                "best_val": float(best_val),
                "best_params": best_params,
                "n_trials": len(all_trials),
                "all_trials": all_trials,
            }
        )

    # Compute consistency metrics
    consistency = compute_consistency_metrics(iter_results)
    logger.info(
        f"  Consistency: val={consistency['val_mean']:.4f}"
        f"±{consistency['val_std']:.4f}"
        f" LR_ratio={consistency['lr_ratio']:.1f}x"
        f" BS_agree={consistency['bs_agreement']:.0%}"
    )

    # Cross-evaluate: test each iteration's best HP on all subsamples
    logger.info("  Cross-evaluating HPs across subsamples...")
    cross_eval = cross_evaluate_hp_configs(iter_results, seqs_per_seed, labels_per_seed)
    for ce in cross_eval:
        logger.info(
            f"    HP lr={ce['hp']['lr']:.5f} bs={ce['hp']['batch_size']}:"
            f" original={ce['original_val']:.4f}"
            f" cross={ce['cross_eval_mean']:.4f}±{ce['cross_eval_std']:.4f}"
        )

    # Pick the best HP by cross-eval mean (most robust)
    best_cross = max(cross_eval, key=lambda x: x["cross_eval_mean"])

    result = {
        "method": args.method,
        "strategy": args.strategy,
        "n_train": args.size,
        "iterations": iter_results,
        "consistency": consistency,
        "cross_evaluation": cross_eval,
        "recommended_hp": best_cross["hp"],
        "recommended_hp_cross_val": best_cross["cross_eval_mean"],
    }

    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
