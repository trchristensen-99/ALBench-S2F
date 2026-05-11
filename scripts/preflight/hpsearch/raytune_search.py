"""Ray Tune HP search driver — 5 strategies.

Strategies:
    random  — basic_variant (uniform sampling); baseline reference
    optuna  — TPE via OptunaSearch
    hyperopt — TPE via HyperOptSearch
    bohb    — Bayesian + HyperBand bandit (via TuneBOHB + HyperBandForBOHB)
    pbt     — Population-Based Training (mutate-during-training)

All wrapped with ASHAScheduler (early termination) except PBT, which has its
own scheduler. Each trial reports per-epoch val_loss via tune.report (see
trainable.py) so ASHA can stop bad trials.

Usage (typically inside a SLURM job):
    python -m scripts.preflight.hpsearch.raytune_search \\
        --arch legnet --d_train 5000 --strategy optuna \\
        --n_trials 50 --max_epochs 60 --gpus 4 --trials_per_gpu 6 \\
        --output_dir results/preflight/hpsearch/optuna_legnet_d5k
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def _make_searcher(strategy: str, arch: str, metric: str, mode: str):
    """Build a Ray Tune search algorithm + scheduler for the given strategy.

    Returns (search_alg, scheduler) tuple. Either can be None.
    """
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    from scripts.preflight.hpsearch.hp_space import to_ray_space

    space = to_ray_space(arch)  # noqa: F841  (used by callers via param_space)

    asha = ASHAScheduler(
        metric=metric,
        mode=mode,
        max_t=60,
        grace_period=8,
        reduction_factor=3,
    )

    if strategy == "random":
        return None, asha
    if strategy == "optuna":
        from ray.tune.search.optuna import OptunaSearch

        return OptunaSearch(metric=metric, mode=mode), asha
    if strategy == "hyperopt":
        from ray.tune.search.hyperopt import HyperOptSearch

        return HyperOptSearch(metric=metric, mode=mode), asha
    if strategy == "bohb":
        from ray.tune.schedulers import HyperBandForBOHB
        from ray.tune.search.bohb import TuneBOHB

        bohb_search = TuneBOHB(metric=metric, mode=mode)
        bohb_sched = HyperBandForBOHB(
            time_attr="training_iteration",
            metric=metric,
            mode=mode,
            max_t=60,
            reduction_factor=3,
        )
        return bohb_search, bohb_sched
    if strategy == "pbt":
        from ray.tune.schedulers import PopulationBasedTraining

        # PBT mutates HPs during training; needs hyperparam_mutations
        pbt_sched = PopulationBasedTraining(
            time_attr="training_iteration",
            metric=metric,
            mode=mode,
            perturbation_interval=5,
            hyperparam_mutations={
                "lr": tune.loguniform(1e-5, 1e-2),
                "weight_decay": tune.loguniform(1e-6, 1e-1),
                "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            },
        )
        return None, pbt_sched
    raise ValueError(f"Unknown strategy: {strategy}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--strategy",
        required=True,
        choices=["random", "optuna", "hyperopt", "bohb", "pbt"],
    )
    ap.add_argument("--arch", required=True, choices=["legnet", "dream_rnn", "dream_attn"])
    ap.add_argument("--d_train", type=int, required=True)
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--max_epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument(
        "--aug",
        type=str,
        default="rev_complement",
        choices=["none", "rev_complement", "rc_shift"],
    )
    ap.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs available to Ray (CUDA_VISIBLE_DEVICES respected).",
    )
    ap.add_argument(
        "--trials_per_gpu",
        type=int,
        default=6,
        help="How many trials share each GPU (fractional gpu allocation).",
    )
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import ray
    from ray import tune

    from scripts.preflight.hpsearch.hp_space import to_ray_space
    from scripts.preflight.hpsearch.trainable import trainable

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save run config for reproducibility
    (out / "search_config.json").write_text(json.dumps(vars(args), indent=2))

    # Init ray with the requested GPU count
    num_cpus = max(1, os.cpu_count() or 4)
    ray.init(
        num_gpus=args.gpus,
        num_cpus=num_cpus,
        log_to_driver=True,
        include_dashboard=False,
        ignore_reinit_error=True,
    )

    metric = "val_loss"
    mode = "min"
    search_alg, scheduler = _make_searcher(args.strategy, args.arch, metric, mode)

    # Fixed params (not searched) — passed as constants
    param_space = to_ray_space(args.arch)
    fixed = {
        "arch": args.arch,
        "d_train": args.d_train,
        "seed": args.seed,
        "epochs": args.max_epochs,
        "patience": args.patience,
        "aug": args.aug,
        "strategy": args.strategy,
        "label_source": "ag_oracle",
    }
    for k, v in fixed.items():
        param_space[k] = v

    gpu_per_trial = 1.0 / args.trials_per_gpu

    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"gpu": gpu_per_trial, "cpu": 2}),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=args.n_trials,
            metric=metric,
            mode=mode,
            max_concurrent_trials=args.gpus * args.trials_per_gpu,
        ),
        run_config=tune.RunConfig(
            name=f"{args.strategy}_{args.arch}_d{args.d_train}",
            storage_path=str(out.absolute()),
            verbose=1,
        ),
    )
    results = tuner.fit()

    # Persist a clean summary
    rows = []
    for r in results:
        cfg = r.config
        m = r.metrics or {}
        rows.append(
            {
                "trial_id": r.path.split("/")[-1] if r.path else None,
                "strategy": args.strategy,
                "arch": args.arch,
                "d_train": args.d_train,
                "val_loss": m.get("val_loss"),
                "test_loss": m.get("test_loss"),
                "best_val": m.get("best_val"),
                "epoch": m.get("epoch"),
                "n_params": m.get("n_params"),
                "gpu_hrs": m.get("gpu_hrs"),
                "config": {
                    k: cfg.get(k)
                    for k in ("lr", "batch_size", "weight_decay", "dropout", "width", "depth")
                },
            }
        )
    rows.sort(key=lambda r: r["val_loss"] or float("inf"))
    (out / "trial_summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\n=== Search done. Best 3 trials by val_loss:")
    for r in rows[:3]:
        print(f"  val={r['val_loss']:.4f} test={r['test_loss']:.4f}  cfg={r['config']}")


if __name__ == "__main__":
    main()
