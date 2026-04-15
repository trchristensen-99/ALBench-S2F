#!/usr/bin/env python
"""HP search plateau analysis: find where more search stops helping.

Tests the marginal value of additional Optuna trials across multiple
dimensions to determine the minimum-cost HP search configuration that
reliably finds near-optimal HPs.

Experiments:
  1. TRIAL COUNT PLATEAU: TPE warm-start with 5/10/15/20/25/30/40/50 trials
     → At what point does the best-found val stabilize?

  2. SAMPLER COMPARISON: At the plateau trial count, compare TPE, CMA-ES,
     Random, GP, TPE-multivariate

  3. ENSEMBLE DEPTH: 1/2/3/5 independent runs with cross-evaluation
     → Does ensembling add value beyond more trials?

  4. COST MITIGATION: Strategies for large N where trials are expensive
     a. Reduced-epoch proxy: 15 epochs instead of 80 for search, retrain best
     b. Subsample proxy: Optimize HP at N/4 subsample, evaluate at full N
     c. Optuna pruning: Kill bad trials after 5 epochs
     d. Warm-transfer: Use best HP from N/2 as warm-start for N

Each experiment runs 5 iterations (seeds) for consistency measurement.

Usage:
    python scripts/hp_search_plateau.py --experiment trial_plateau \
        --strategy random --size 100000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ["TORCHDYNAMO_DISABLE"] = "1"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

ITER_SEEDS = [42, 1042, 2042, 3042, 4042]

WARM_CONFIGS = [
    {"lr": 0.001, "batch_size": 512, "weight_decay": 1e-5},
    {"lr": 0.005, "batch_size": 1024, "weight_decay": 1e-5},
    {"lr": 0.002, "batch_size": 256, "weight_decay": 0.004},
    {"lr": 0.003, "batch_size": 512, "weight_decay": 1e-6},
]


def train_and_evaluate_reduced(
    seqs, labels, lr, batch_size, weight_decay, seed, max_epochs=80, patience=10
):
    """Train LegNet with configurable epochs/patience for cost experiments."""
    from models.legnet_student import LegNetStudent, TrainConfig
    from scripts.optuna_legnet_scaling import get_chr_val

    val_seqs, val_labels = get_chr_val()
    effective_bs = min(batch_size, max(32, len(seqs) // 2))
    config = TrainConfig(
        lr=lr,
        batch_size=effective_bs,
        weight_decay=weight_decay,
        epochs=max_epochs,
        early_stopping_patience=patience,
    )
    model = LegNetStudent(ensemble_size=1, train_config=config)
    model.fit(sequences=seqs, labels=labels, val_sequences=val_seqs, val_labels=val_labels)
    preds = model.predict(val_seqs)
    from scipy.stats import pearsonr

    val_r, _ = pearsonr(val_labels, preds)
    return float(val_r)


def make_objective(seqs, labels, seed, max_epochs=80, patience=10):
    """Create objective with configurable training budget."""
    from scripts.optuna_legnet_scaling import get_chr_val

    _ = get_chr_val()

    def objective(trial):
        lr = trial.suggest_float("lr", 5e-5, 5e-2, log=True)
        bs = trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048, 4096])
        wd = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
        return train_and_evaluate_reduced(
            seqs,
            labels,
            lr,
            bs,
            wd,
            seed,
            max_epochs=max_epochs,
            patience=patience,
        )

    return objective


def make_pruning_objective(seqs, labels, seed):
    """Objective that reports intermediate values for Optuna pruning."""
    from scipy.stats import pearsonr

    from models.legnet_student import LegNetStudent, TrainConfig
    from scripts.optuna_legnet_scaling import get_chr_val

    val_seqs, val_labels = get_chr_val()

    def objective(trial):
        lr = trial.suggest_float("lr", 5e-5, 5e-2, log=True)
        bs = trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048, 4096])
        wd = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)

        effective_bs = min(bs, max(32, len(seqs) // 2))
        config = TrainConfig(
            lr=lr,
            batch_size=effective_bs,
            weight_decay=wd,
            epochs=80,
            early_stopping_patience=10,
        )
        model = LegNetStudent(ensemble_size=1, train_config=config)

        # Train with intermediate reporting for pruning
        # We'll train in chunks and report to Optuna
        model.fit(sequences=seqs, labels=labels, val_sequences=val_seqs, val_labels=val_labels)

        preds = model.predict(val_seqs)
        val_r, _ = pearsonr(val_labels, preds)
        return float(val_r)

    return objective


def run_search(
    sampler,
    n_trials,
    seqs,
    labels,
    seed,
    strategy,
    n_train,
    warm_start=True,
    max_epochs=80,
    patience=10,
):
    """Run a single HP search and return results."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize", sampler=sampler)

    if warm_start:
        for wc in WARM_CONFIGS:
            study.enqueue_trial(wc)
        # Add neighbor configs
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
                    study.enqueue_trial(nc)
                except Exception:
                    pass

    objective = make_objective(seqs, labels, seed, max_epochs=max_epochs, patience=patience)

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    wall_time = time.time() - t0

    best = study.best_trial
    all_trials = [
        {"val": t.value, "params": t.params, "number": t.number}
        for t in study.trials
        if t.value is not None
    ]

    # Track convergence: best val at each trial number
    running_best = []
    best_so_far = -999
    for t in sorted(all_trials, key=lambda x: x["number"]):
        if t["val"] is not None and t["val"] > best_so_far:
            best_so_far = t["val"]
        running_best.append(best_so_far)

    return {
        "best_val": float(best.value),
        "best_params": best.params,
        "wall_time_sec": wall_time,
        "n_trials": len(all_trials),
        "convergence_curve": running_best,
        "all_trials": all_trials,
    }


def experiment_trial_plateau(strategy, n_train):
    """Experiment 1: How many trials until best-val plateaus?"""
    import optuna

    from scripts.optuna_legnet_scaling import load_pool_data

    trial_counts = [5, 10, 15, 20, 25, 30, 40, 50]
    results = {}

    for n_trials in trial_counts:
        logger.info(f"--- Trial count: {n_trials} ---")
        iter_results = []

        for seed in ITER_SEEDS:
            seqs, labels = load_pool_data(strategy, n_train, seed)
            sampler = optuna.samplers.TPESampler(seed=seed)
            res = run_search(
                sampler, n_trials, seqs, labels, seed, strategy, n_train, warm_start=True
            )
            iter_results.append(res)
            logger.info(
                f"  seed={seed}: val={res['best_val']:.4f} "
                f"lr={res['best_params']['lr']:.5f} "
                f"bs={res['best_params']['batch_size']} "
                f"time={res['wall_time_sec']:.0f}s"
            )

        vals = [r["best_val"] for r in iter_results]
        lrs = [r["best_params"]["lr"] for r in iter_results]
        bss = [r["best_params"]["batch_size"] for r in iter_results]

        results[n_trials] = {
            "n_trials": n_trials,
            "val_mean": float(np.mean(vals)),
            "val_std": float(np.std(vals)),
            "val_per_seed": [float(v) for v in vals],
            "lr_ratio": float(max(lrs) / max(min(lrs), 1e-10)),
            "lr_per_seed": [float(lr) for lr in lrs],
            "bs_per_seed": [int(bs) for bs in bss],
            "mean_wall_time": float(np.mean([r["wall_time_sec"] for r in iter_results])),
            "convergence_curves": [r["convergence_curve"] for r in iter_results],
            "iterations": iter_results,
        }

        logger.info(
            f"  n_trials={n_trials}: val={np.mean(vals):.4f}+/-{np.std(vals):.4f} "
            f"LR_ratio={max(lrs) / max(min(lrs), 1e-10):.1f}x "
            f"time={np.mean([r['wall_time_sec'] for r in iter_results]):.0f}s"
        )

    return results


def experiment_sampler_comparison(strategy, n_train, n_trials=20):
    """Experiment 2: Compare samplers at a fixed trial count."""
    import optuna

    from scripts.optuna_legnet_scaling import load_pool_data

    samplers = {
        "tpe": lambda seed: optuna.samplers.TPESampler(seed=seed),
        "tpe_multi": lambda seed: optuna.samplers.TPESampler(seed=seed, multivariate=True),
        "cma_es": lambda seed: optuna.samplers.CmaEsSampler(seed=seed),
        "random": lambda seed: optuna.samplers.RandomSampler(seed=seed),
    }

    # Try GP if available
    try:
        _ = optuna.samplers.GPSampler(seed=0)
        samplers["gp"] = lambda seed: optuna.samplers.GPSampler(seed=seed)
    except (AttributeError, Exception):
        logger.info("GPSampler not available, skipping")

    results = {}
    for sampler_name, sampler_fn in samplers.items():
        logger.info(f"--- Sampler: {sampler_name} ---")
        iter_results = []

        for seed in ITER_SEEDS:
            seqs, labels = load_pool_data(strategy, n_train, seed)
            sampler = sampler_fn(seed)
            res = run_search(
                sampler, n_trials, seqs, labels, seed, strategy, n_train, warm_start=True
            )
            iter_results.append(res)
            logger.info(f"  seed={seed}: val={res['best_val']:.4f}")

        vals = [r["best_val"] for r in iter_results]
        results[sampler_name] = {
            "sampler": sampler_name,
            "val_mean": float(np.mean(vals)),
            "val_std": float(np.std(vals)),
            "val_per_seed": [float(v) for v in vals],
            "mean_wall_time": float(np.mean([r["wall_time_sec"] for r in iter_results])),
            "iterations": iter_results,
        }
        logger.info(f"  {sampler_name}: val={np.mean(vals):.4f}+/-{np.std(vals):.4f}")

    return results


def experiment_ensemble_depth(strategy, n_train, n_trials=20):
    """Experiment 3: How many independent runs before cross-eval stops helping?"""
    import optuna

    from scripts.optuna_legnet_scaling import load_pool_data, train_and_evaluate

    # Run 5 independent searches (one per seed)
    all_search_results = []
    seqs_per_seed = {}
    labels_per_seed = {}
    for seed in ITER_SEEDS:
        seqs, labels = load_pool_data(strategy, n_train, seed)
        seqs_per_seed[seed] = seqs
        labels_per_seed[seed] = labels
        sampler = optuna.samplers.TPESampler(seed=seed)
        res = run_search(sampler, n_trials, seqs, labels, seed, strategy, n_train, warm_start=True)
        all_search_results.append(res)
        logger.info(f"  Run seed={seed}: val={res['best_val']:.4f}")

    # Now test ensemble depths: use first K runs, cross-evaluate
    results = {}
    for depth in [1, 2, 3, 5]:
        logger.info(f"--- Ensemble depth: {depth} ---")
        runs = all_search_results[:depth]

        # Collect top-3 HPs from each run
        candidates = []
        for r in runs:
            sorted_trials = sorted(
                [t for t in r["all_trials"] if t["val"] is not None], key=lambda t: -t["val"]
            )
            for t in sorted_trials[:3]:
                candidates.append(t["params"])

        if depth == 1:
            # No cross-eval for single run, just use best
            best_hp = runs[0]["best_params"]
            # Still evaluate on all subsamples for fair comparison
            scores = []
            for eval_seed in ITER_SEEDS:
                val = train_and_evaluate_reduced(
                    seqs_per_seed[eval_seed],
                    labels_per_seed[eval_seed],
                    best_hp["lr"],
                    best_hp["batch_size"],
                    best_hp["weight_decay"],
                    eval_seed,
                )
                scores.append(val)
            results[depth] = {
                "depth": depth,
                "consensus_val_mean": float(np.mean(scores)),
                "consensus_val_std": float(np.std(scores)),
                "consensus_hp": best_hp,
                "n_candidates": 1,
            }
        else:
            # Cross-evaluate all candidates on all subsamples
            hp_scores = []
            for hp in candidates:
                scores = []
                for eval_seed in ITER_SEEDS[:3]:  # Cross-eval on 3 subsamples
                    val = train_and_evaluate_reduced(
                        seqs_per_seed[eval_seed],
                        labels_per_seed[eval_seed],
                        hp["lr"],
                        hp["batch_size"],
                        hp["weight_decay"],
                        eval_seed,
                    )
                    scores.append(val)
                hp_scores.append((np.mean(scores), np.std(scores), hp, scores))

            hp_scores.sort(key=lambda x: -x[0])
            best_mean, best_std, best_hp, best_scores = hp_scores[0]
            results[depth] = {
                "depth": depth,
                "consensus_val_mean": float(best_mean),
                "consensus_val_std": float(best_std),
                "consensus_hp": best_hp,
                "n_candidates": len(candidates),
            }

        logger.info(
            f"  depth={depth}: val={results[depth]['consensus_val_mean']:.4f}"
            f"+/-{results[depth]['consensus_val_std']:.4f}"
            f" ({results[depth]['n_candidates']} candidates)"
        )

    return results


def experiment_cost_mitigation(strategy, n_train):
    """Experiment 4: Cost-saving strategies for large N.

    Compares:
    a) Full search (80 epochs, patience=10) — ground truth
    b) Reduced-epoch proxy (15 epochs, patience=5) — 4x faster per trial
    c) Subsample proxy (search at N/4, eval at N) — 4x fewer samples
    d) Combined (reduced epochs on subsample) — 16x faster
    """
    import optuna

    from scripts.optuna_legnet_scaling import load_pool_data

    n_trials = 20
    results = {}

    for seed in ITER_SEEDS[:3]:  # 3 iterations for cost experiments
        seqs_full, labels_full = load_pool_data(strategy, n_train, seed)

        # a) Full search (ground truth)
        logger.info(f"  seed={seed} — Full search (80ep)...")
        sampler = optuna.samplers.TPESampler(seed=seed)
        res_full = run_search(
            sampler,
            n_trials,
            seqs_full,
            labels_full,
            seed,
            strategy,
            n_train,
            warm_start=True,
            max_epochs=80,
            patience=10,
        )

        # b) Reduced-epoch proxy (15 epochs)
        logger.info(f"  seed={seed} — Reduced-epoch (15ep)...")
        sampler = optuna.samplers.TPESampler(seed=seed)
        res_reduced = run_search(
            sampler,
            n_trials,
            seqs_full,
            labels_full,
            seed,
            strategy,
            n_train,
            warm_start=True,
            max_epochs=15,
            patience=5,
        )

        # c) Subsample proxy (N/4)
        sub_n = max(n_train // 4, 5000)
        seqs_sub, labels_sub = load_pool_data(strategy, sub_n, seed)
        logger.info(f"  seed={seed} — Subsample proxy (n={sub_n})...")
        sampler = optuna.samplers.TPESampler(seed=seed)
        res_sub = run_search(
            sampler,
            n_trials,
            seqs_sub,
            labels_sub,
            seed,
            strategy,
            sub_n,
            warm_start=True,
            max_epochs=80,
            patience=10,
        )

        # d) Combined (reduced epochs + subsample)
        logger.info(f"  seed={seed} — Combined (15ep + subsample)...")
        sampler = optuna.samplers.TPESampler(seed=seed)
        res_combined = run_search(
            sampler,
            n_trials,
            seqs_sub,
            labels_sub,
            seed,
            strategy,
            sub_n,
            warm_start=True,
            max_epochs=15,
            patience=5,
        )

        # Now evaluate ALL best HPs at the FULL size with FULL epochs
        # This is the real test: does the cheap proxy find good HPs?
        configs = {
            "full_80ep": res_full["best_params"],
            "reduced_15ep": res_reduced["best_params"],
            "subsample_n%d" % sub_n: res_sub["best_params"],
            "combined_15ep_n%d" % sub_n: res_combined["best_params"],
        }

        eval_results = {}
        for config_name, hp in configs.items():
            val = train_and_evaluate_reduced(
                seqs_full,
                labels_full,
                hp["lr"],
                hp["batch_size"],
                hp["weight_decay"],
                seed,
                max_epochs=80,
                patience=10,
            )
            eval_results[config_name] = {
                "hp": hp,
                "full_eval_val": float(val),
                "search_time_sec": {
                    "full_80ep": res_full["wall_time_sec"],
                    "reduced_15ep": res_reduced["wall_time_sec"],
                    "subsample_n%d" % sub_n: res_sub["wall_time_sec"],
                    "combined_15ep_n%d" % sub_n: res_combined["wall_time_sec"],
                }.get(config_name, 0),
            }
            logger.info(
                f"    {config_name}: search_val={configs[config_name]} -> full_eval={val:.4f}"
            )

        results[seed] = eval_results

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["trial_plateau", "sampler_comparison", "ensemble_depth", "cost_mitigation", "all"],
    )
    parser.add_argument("--strategy", required=True, choices=["random", "genomic", "motif_grammar"])
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument(
        "--n-trials", type=int, default=20, help="Trial count for sampler/ensemble experiments"
    )
    args = parser.parse_args()

    out_dir = REPO / "outputs" / "hp_plateau" / args.strategy / f"n{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment in ("trial_plateau", "all"):
        out_file = out_dir / "trial_plateau.json"
        if out_file.exists():
            logger.info(f"SKIP trial_plateau: {out_file} exists")
        else:
            logger.info("=== EXPERIMENT 1: Trial count plateau ===")
            results = experiment_trial_plateau(args.strategy, args.size)
            with open(out_file, "w") as f:
                json.dump(
                    {
                        "experiment": "trial_plateau",
                        "strategy": args.strategy,
                        "n_train": args.size,
                        "results": {str(k): v for k, v in results.items()},
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Saved: {out_file}")

    if args.experiment in ("sampler_comparison", "all"):
        out_file = out_dir / "sampler_comparison.json"
        if out_file.exists():
            logger.info(f"SKIP sampler_comparison: {out_file} exists")
        else:
            logger.info("=== EXPERIMENT 2: Sampler comparison ===")
            results = experiment_sampler_comparison(
                args.strategy, args.size, n_trials=args.n_trials
            )
            with open(out_file, "w") as f:
                json.dump(
                    {
                        "experiment": "sampler_comparison",
                        "strategy": args.strategy,
                        "n_train": args.size,
                        "n_trials": args.n_trials,
                        "results": results,
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Saved: {out_file}")

    if args.experiment in ("ensemble_depth", "all"):
        out_file = out_dir / "ensemble_depth.json"
        if out_file.exists():
            logger.info(f"SKIP ensemble_depth: {out_file} exists")
        else:
            logger.info("=== EXPERIMENT 3: Ensemble depth ===")
            results = experiment_ensemble_depth(args.strategy, args.size, n_trials=args.n_trials)
            with open(out_file, "w") as f:
                json.dump(
                    {
                        "experiment": "ensemble_depth",
                        "strategy": args.strategy,
                        "n_train": args.size,
                        "n_trials": args.n_trials,
                        "results": {str(k): v for k, v in results.items()},
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Saved: {out_file}")

    if args.experiment in ("cost_mitigation", "all"):
        out_file = out_dir / "cost_mitigation.json"
        if out_file.exists():
            logger.info(f"SKIP cost_mitigation: {out_file} exists")
        else:
            logger.info("=== EXPERIMENT 4: Cost mitigation ===")
            results = experiment_cost_mitigation(args.strategy, args.size)
            with open(out_file, "w") as f:
                json.dump(
                    {
                        "experiment": "cost_mitigation",
                        "strategy": args.strategy,
                        "n_train": args.size,
                        "results": {str(k): v for k, v in results.items()},
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
