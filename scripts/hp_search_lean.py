#!/usr/bin/env python
"""Lean HP search comparison: fill gaps in what we already know.

We already know:
  - 20-trial warm-start >> 10-trial cold (~0.06 Pearson at 50K)
  - Ensemble 3x20 ≈ single 20-trial (no meaningful benefit at ≥10K)
  - HP landscape is flat near optimum at larger N (top-5 spread ~0.01)

This script tests what we DON'T know:
  1. TRIAL_EXTENSION: Do 30 or 40 trials beat 20? (incremental test)
  2. SAMPLER_ALT: Do GP, QMC, or CMA-ES beat TPE? (at 20 trials)
  3. COST_PROXY: Can we use 15-epoch proxy to find HPs for large N?
  4. ENSEMBLE_LEAN: Does depth=3 or 5 help over depth=1? (quick check)

Each experiment runs 3 iterations at BOTH 50K and 100K.

Usage:
    python scripts/hp_search_lean.py --experiment trial_extension \
        --strategy random --size 50000
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

ITER_SEEDS = [42, 1042, 2042]

WARM_CONFIGS = [
    {"lr": 0.001, "batch_size": 512, "weight_decay": 1e-5},
    {"lr": 0.005, "batch_size": 1024, "weight_decay": 1e-5},
    {"lr": 0.002, "batch_size": 256, "weight_decay": 0.004},
    {"lr": 0.003, "batch_size": 512, "weight_decay": 1e-6},
]


def train_eval(seqs, labels, lr, batch_size, weight_decay, seed, max_epochs=80, patience=10):
    """Train LegNet, return val Pearson R."""
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

    return float(pearsonr(val_labels, preds)[0])


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
    """Run Optuna search, return results dict."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from scripts.optuna_legnet_scaling import get_chr_val

    _ = get_chr_val()

    study = optuna.create_study(direction="maximize", sampler=sampler)

    if warm_start:
        for wc in WARM_CONFIGS:
            study.enqueue_trial(wc)
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
                    study.enqueue_trial(json.loads(nf.read_text())["config"])
                except Exception:
                    pass

    def objective(trial):
        lr = trial.suggest_float("lr", 5e-5, 5e-2, log=True)
        bs = trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048, 4096])
        wd = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
        return train_eval(seqs, labels, lr, bs, wd, seed, max_epochs=max_epochs, patience=patience)

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    wall_time = time.time() - t0

    best = study.best_trial
    all_trials = [
        {"val": t.value, "params": t.params, "number": t.number}
        for t in study.trials
        if t.value is not None
    ]

    # Convergence: running best at each trial
    running_best = []
    best_so_far = -999
    for t in sorted(all_trials, key=lambda x: x["number"]):
        if t["val"] > best_so_far:
            best_so_far = t["val"]
        running_best.append(best_so_far)

    return {
        "best_val": float(best.value),
        "best_params": best.params,
        "wall_time_sec": wall_time,
        "convergence": running_best,
        "all_trials": all_trials,
    }


def experiment_trial_extension(strategy, n_train):
    """Test 30 and 40 trials (we already have 10 and 20 data)."""
    import optuna

    from scripts.optuna_legnet_scaling import load_pool_data

    results = {}
    for n_trials in [30, 40]:
        logger.info(f"--- {n_trials} trials ---")
        iters = []
        for seed in ITER_SEEDS:
            seqs, labels = load_pool_data(strategy, n_train, seed)
            sampler = optuna.samplers.TPESampler(seed=seed)
            res = run_search(sampler, n_trials, seqs, labels, seed, strategy, n_train)
            iters.append(res)
            logger.info(
                f"  seed={seed}: val={res['best_val']:.4f} "
                f"lr={res['best_params']['lr']:.5f} bs={res['best_params']['batch_size']} "
                f"time={res['wall_time_sec']:.0f}s"
            )

        vals = [r["best_val"] for r in iters]
        lrs = [r["best_params"]["lr"] for r in iters]
        results[n_trials] = {
            "n_trials": n_trials,
            "val_mean": float(np.mean(vals)),
            "val_std": float(np.std(vals)),
            "lr_ratio": float(max(lrs) / max(min(lrs), 1e-10)),
            "per_seed": [
                {"val": r["best_val"], "params": r["best_params"], "convergence": r["convergence"]}
                for r in iters
            ],
            "mean_wall_time": float(np.mean([r["wall_time_sec"] for r in iters])),
        }
        logger.info(f"  {n_trials}t: val={np.mean(vals):.4f}+/-{np.std(vals):.4f}")

    return results


def experiment_sampler_alt(strategy, n_train):
    """Compare alternative samplers: GP, QMC, CMA-ES vs TPE baseline."""
    import optuna

    from scripts.optuna_legnet_scaling import load_pool_data

    sampler_configs = {
        "tpe_20_warm": lambda s: optuna.samplers.TPESampler(seed=s),
        "gp_20_warm": lambda s: optuna.samplers.GPSampler(seed=s),
        "qmc_20_warm": lambda s: optuna.samplers.QMCSampler(seed=s),
        "cma_20_warm": lambda s: optuna.samplers.CmaEsSampler(seed=s),
    }

    results = {}
    for name, sampler_fn in sampler_configs.items():
        logger.info(f"--- {name} ---")
        iters = []
        for seed in ITER_SEEDS:
            seqs, labels = load_pool_data(strategy, n_train, seed)
            try:
                sampler = sampler_fn(seed)
            except Exception as e:
                logger.warning(f"  Sampler {name} failed: {e}")
                break
            res = run_search(sampler, 20, seqs, labels, seed, strategy, n_train)
            iters.append(res)
            logger.info(
                f"  seed={seed}: val={res['best_val']:.4f} time={res['wall_time_sec']:.0f}s"
            )

        if iters:
            vals = [r["best_val"] for r in iters]
            results[name] = {
                "sampler": name,
                "val_mean": float(np.mean(vals)),
                "val_std": float(np.std(vals)),
                "mean_wall_time": float(np.mean([r["wall_time_sec"] for r in iters])),
                "per_seed": [{"val": r["best_val"], "params": r["best_params"]} for r in iters],
            }
            logger.info(f"  {name}: val={np.mean(vals):.4f}+/-{np.std(vals):.4f}")

    return results


def experiment_cost_proxy(strategy, n_train):
    """Test reduced-epoch proxy: search with 15ep, evaluate best at 80ep.

    Also test subsample proxy: search at N/4, evaluate at full N.
    """
    import optuna

    from scripts.optuna_legnet_scaling import load_pool_data

    results = {}
    for seed in ITER_SEEDS:
        seqs_full, labels_full = load_pool_data(strategy, n_train, seed)
        seed_results = {}

        # a) Full 80-epoch search (ground truth)
        logger.info(f"  seed={seed} full_80ep...")
        sampler = optuna.samplers.TPESampler(seed=seed)
        res_full = run_search(
            sampler, 20, seqs_full, labels_full, seed, strategy, n_train, max_epochs=80, patience=10
        )

        # b) 15-epoch proxy
        logger.info(f"  seed={seed} proxy_15ep...")
        sampler = optuna.samplers.TPESampler(seed=seed)
        res_15ep = run_search(
            sampler, 20, seqs_full, labels_full, seed, strategy, n_train, max_epochs=15, patience=5
        )

        # c) Subsample N/4
        sub_n = max(n_train // 4, 5000)
        seqs_sub, labels_sub = load_pool_data(strategy, sub_n, seed)
        logger.info(f"  seed={seed} subsample_n{sub_n}...")
        sampler = optuna.samplers.TPESampler(seed=seed)
        res_sub = run_search(
            sampler, 20, seqs_sub, labels_sub, seed, strategy, sub_n, max_epochs=80, patience=10
        )

        # d) Combined: 15ep + subsample
        logger.info(f"  seed={seed} combined...")
        sampler = optuna.samplers.TPESampler(seed=seed)
        res_comb = run_search(
            sampler, 20, seqs_sub, labels_sub, seed, strategy, sub_n, max_epochs=15, patience=5
        )

        # Evaluate ALL best HPs at full N, full epochs
        configs = {
            "full_80ep": res_full["best_params"],
            "proxy_15ep": res_15ep["best_params"],
            "subsample": res_sub["best_params"],
            "combined": res_comb["best_params"],
        }
        for cname, hp in configs.items():
            full_val = train_eval(
                seqs_full, labels_full, hp["lr"], hp["batch_size"], hp["weight_decay"], seed
            )
            seed_results[cname] = {
                "hp": hp,
                "search_val": {
                    "full_80ep": res_full["best_val"],
                    "proxy_15ep": res_15ep["best_val"],
                    "subsample": res_sub["best_val"],
                    "combined": res_comb["best_val"],
                }.get(cname, 0),
                "full_eval_val": float(full_val),
                "search_time": {
                    "full_80ep": res_full["wall_time_sec"],
                    "proxy_15ep": res_15ep["wall_time_sec"],
                    "subsample": res_sub["wall_time_sec"],
                    "combined": res_comb["wall_time_sec"],
                }.get(cname, 0),
            }
            logger.info(
                f"    {cname}: search_time={seed_results[cname]['search_time']:.0f}s "
                f"full_eval={full_val:.4f}"
            )

        results[seed] = seed_results

    # Aggregate
    summary = {}
    for cname in ["full_80ep", "proxy_15ep", "subsample", "combined"]:
        vals = [results[s][cname]["full_eval_val"] for s in ITER_SEEDS]
        times = [results[s][cname]["search_time"] for s in ITER_SEEDS]
        summary[cname] = {
            "full_eval_mean": float(np.mean(vals)),
            "full_eval_std": float(np.std(vals)),
            "mean_search_time": float(np.mean(times)),
            "speedup_vs_full": float(np.mean(times))
            / max(np.mean([results[s]["full_80ep"]["search_time"] for s in ITER_SEEDS]), 1),
        }
    results["summary"] = summary
    return results


def experiment_ensemble_lean(strategy, n_train):
    """Quick test: depth=3 vs depth=1. Skip depth=2,5 since we know
    ensemble ≈ single run from existing data."""
    import optuna

    from scripts.optuna_legnet_scaling import load_pool_data

    # Run 3 independent searches
    all_runs = []
    seqs_per_seed = {}
    labels_per_seed = {}
    for seed in ITER_SEEDS:
        seqs, labels = load_pool_data(strategy, n_train, seed)
        seqs_per_seed[seed] = seqs
        labels_per_seed[seed] = labels
        sampler = optuna.samplers.TPESampler(seed=seed)
        res = run_search(sampler, 20, seqs, labels, seed, strategy, n_train)
        all_runs.append(res)
        logger.info(f"  Run seed={seed}: val={res['best_val']:.4f}")

    results = {}
    for depth in [1, 3]:
        runs = all_runs[:depth]
        if depth == 1:
            best_hp = runs[0]["best_params"]
        else:
            # Cross-eval top-3 from each run
            candidates = []
            for r in runs:
                top3 = sorted(
                    [t for t in r["all_trials"] if t["val"] is not None], key=lambda t: -t["val"]
                )[:3]
                candidates.extend([t["params"] for t in top3])

            hp_scores = []
            for hp in candidates:
                scores = []
                for eval_seed in ITER_SEEDS:
                    val = train_eval(
                        seqs_per_seed[eval_seed],
                        labels_per_seed[eval_seed],
                        hp["lr"],
                        hp["batch_size"],
                        hp["weight_decay"],
                        eval_seed,
                    )
                    scores.append(val)
                hp_scores.append((np.mean(scores), hp))
            hp_scores.sort(key=lambda x: -x[0])
            best_hp = hp_scores[0][1]

        # Evaluate on all subsamples
        eval_vals = []
        for eval_seed in ITER_SEEDS:
            val = train_eval(
                seqs_per_seed[eval_seed],
                labels_per_seed[eval_seed],
                best_hp["lr"],
                best_hp["batch_size"],
                best_hp["weight_decay"],
                eval_seed,
            )
            eval_vals.append(val)

        results[depth] = {
            "depth": depth,
            "hp": best_hp,
            "eval_mean": float(np.mean(eval_vals)),
            "eval_std": float(np.std(eval_vals)),
        }
        logger.info(f"  depth={depth}: eval={np.mean(eval_vals):.4f}+/-{np.std(eval_vals):.4f}")

    return results


def experiment_pruning(strategy, n_train):
    """Test Optuna pruning: kill bad trials early, let good ones train fully.

    Compares:
    a) Standard 20-trial (80ep, patience=10) — baseline
    b) MedianPruner 30-trial (80ep, prune after epoch 5) — more trials, bad ones killed
    c) HyperbandPruner 30-trial — multi-fidelity, automatic bracket scheduling
    d) MedianPruner 50-trial (80ep, prune after epoch 5) — even more trials

    Pruning lets us run MORE trials in the same wall time by killing losers early.
    """
    import optuna

    from scripts.optuna_legnet_scaling import get_chr_val, load_pool_data

    _ = get_chr_val()
    val_seqs, val_labels = get_chr_val()

    def make_pruning_objective(seqs, labels, seed, trial_ref):
        """Create objective that reports intermediate vals for pruning."""
        from scipy.stats import pearsonr

        from models.legnet_student import LegNetStudent, TrainConfig

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

            def epoch_cb(epoch, val_metrics):
                val_r = val_metrics.get("pearson_r", 0)
                trial.report(val_r, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                return False

            try:
                model.fit(
                    sequences=seqs,
                    labels=labels,
                    val_sequences=val_seqs,
                    val_labels=val_labels,
                    epoch_callback=epoch_cb,
                )
                preds = model.predict(val_seqs)
                val_r, _ = pearsonr(val_labels, preds)
                return float(val_r)
            except optuna.TrialPruned:
                raise

        return objective

    configs = {
        "standard_20": {"n_trials": 20, "pruner": None},
        "median_30": {
            "n_trials": 30,
            "pruner": lambda: optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        },
        "hyperband_30": {
            "n_trials": 30,
            "pruner": lambda: optuna.pruners.HyperbandPruner(
                min_resource=5, max_resource=80, reduction_factor=3
            ),
        },
        "median_50": {
            "n_trials": 50,
            "pruner": lambda: optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        },
    }

    results = {}
    for config_name, cfg in configs.items():
        logger.info(f"--- {config_name} ---")
        iters = []
        for seed in ITER_SEEDS:
            seqs, labels = load_pool_data(strategy, n_train, seed)
            sampler = optuna.samplers.TPESampler(seed=seed)
            pruner = cfg["pruner"]() if cfg["pruner"] else optuna.pruners.NopPruner()

            study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
            for wc in WARM_CONFIGS:
                study.enqueue_trial(wc)

            if cfg["pruner"] is not None:
                obj = make_pruning_objective(seqs, labels, seed, None)
            else:
                # Standard objective (no pruning callbacks)
                from scripts.optuna_legnet_scaling import get_chr_val as _gcv

                _gcv()

                def obj(trial, _seqs=seqs, _labels=labels, _seed=seed):
                    lr = trial.suggest_float("lr", 5e-5, 5e-2, log=True)
                    bs = trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048, 4096])
                    wd = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
                    return train_eval(_seqs, _labels, lr, bs, wd, _seed)

            t0 = time.time()
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(obj, n_trials=cfg["n_trials"])
            wall_time = time.time() - t0

            best = study.best_trial
            n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
            n_complete = len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            )

            iters.append(
                {
                    "best_val": float(best.value),
                    "best_params": best.params,
                    "wall_time_sec": wall_time,
                    "n_complete": n_complete,
                    "n_pruned": n_pruned,
                }
            )
            logger.info(
                f"  seed={seed}: val={best.value:.4f} "
                f"time={wall_time:.0f}s pruned={n_pruned}/{cfg['n_trials']}"
            )

        vals = [r["best_val"] for r in iters]
        times = [r["wall_time_sec"] for r in iters]
        results[config_name] = {
            "config": config_name,
            "n_trials": cfg["n_trials"],
            "has_pruner": cfg["pruner"] is not None,
            "val_mean": float(np.mean(vals)),
            "val_std": float(np.std(vals)),
            "mean_wall_time": float(np.mean(times)),
            "mean_pruned": float(np.mean([r["n_pruned"] for r in iters])),
            "per_seed": iters,
        }
        logger.info(
            f"  {config_name}: val={np.mean(vals):.4f}+/-{np.std(vals):.4f} "
            f"time={np.mean(times):.0f}s"
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["trial_extension", "sampler_alt", "cost_proxy", "ensemble_lean", "pruning"],
    )
    parser.add_argument("--strategy", required=True, choices=["random", "genomic", "motif_grammar"])
    parser.add_argument("--size", type=int, required=True)
    args = parser.parse_args()

    out_dir = REPO / "outputs" / "hp_lean" / args.strategy / f"n{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.experiment}.json"

    if out_file.exists():
        logger.info(f"SKIP: {out_file} exists")
        return

    from scripts.optuna_legnet_scaling import get_chr_val

    _ = get_chr_val()

    logger.info(f"=== {args.experiment} | {args.strategy} n={args.size} ===")

    if args.experiment == "trial_extension":
        results = experiment_trial_extension(args.strategy, args.size)
    elif args.experiment == "sampler_alt":
        results = experiment_sampler_alt(args.strategy, args.size)
    elif args.experiment == "cost_proxy":
        results = experiment_cost_proxy(args.strategy, args.size)
    elif args.experiment == "ensemble_lean":
        results = experiment_ensemble_lean(args.strategy, args.size)
    elif args.experiment == "pruning":
        results = experiment_pruning(args.strategy, args.size)

    with open(out_file, "w") as f:
        json.dump(
            {
                "experiment": args.experiment,
                "strategy": args.strategy,
                "n_train": args.size,
                "results": results,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
