"""Ray Tune Trainable wrapping run_single.train() with per-epoch reports.

Uses function_trainable. Each trial:
  1. Sets seed, builds args namespace from Ray config
  2. Translates abstract HP (width/depth/etc) to run_single overrides
  3. Calls run_single.train() with epoch_callback that calls tune.report
  4. Final result.json written by train()

ASHA can early-stop trials based on the per-epoch heartbeat val_loss.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def _build_args(config: dict[str, Any], trial_dir: str) -> argparse.Namespace:
    """Build the argparse.Namespace that run_single.train() expects."""
    return argparse.Namespace(
        arch=config["arch"],
        d_train=int(config["d_train"]),
        seed=int(config["seed"]),
        epochs=int(config.get("epochs", 60)),
        augmentations=config.get("aug", "rev_complement"),
        gpu=0,
        num_workers=int(config.get("num_workers", 2)),
        use_amp=True,
        output_dir=trial_dir,
        label_source=config.get("label_source", "ag_oracle"),
        report_min_val_in_final_pct=0.1,
        early_stop_patience=int(config.get("patience", 15)),
        evoaug_intensity=config.get("evoaug_intensity"),
        evoaug_prob=float(config.get("evoaug_prob", 0.5)),
        sweep_name=config.get("sweep_name"),
        hp=[],
    )


def trainable(config: dict[str, Any]):
    """Ray Tune function trainable. Runs one trial to completion (or until ASHA stops it)."""
    from ray import tune

    import wandb
    from scripts.preflight.hpsearch.hp_space import to_run_single_overrides
    from scripts.preflight.run_single import ARCH_PRIORS, train

    trial_dir = tune.get_context().get_trial_dir()
    args = _build_args(config, trial_dir)

    # Build hp dict: start from arch priors, apply abstract HPs via translator.
    arch = config["arch"]
    hp = dict(ARCH_PRIORS[arch])
    # Translate abstract HPs to overrides via the same shared logic used everywhere
    abstract_hp = {
        "lr": config["lr"],
        "batch_size": config["batch_size"],
        "weight_decay": config["weight_decay"],
        "dropout": config["dropout"],
        "width": config["width"],
        "depth": config["depth"],
    }
    overrides_strings = to_run_single_overrides(arch, abstract_hp)
    # Parse override strings (they're "k=v" form)
    from scripts.preflight.run_single import parse_overrides

    hp.update(parse_overrides(overrides_strings))

    # W&B init for this trial. WANDB_API_KEY in env (set via SLURM script).
    strategy = config.get("strategy", "unknown")
    wandb_run = None
    try:
        wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "albench-s2f-hpsearch"),
            entity=os.environ.get("WANDB_ENTITY"),
            name=f"{strategy}_{arch}_d{config['d_train']}_t{tune.get_context().get_trial_id()}",
            tags=[
                f"arch={arch}",
                f"d={config['d_train']}",
                f"strategy={strategy}",
                "phase=hpsearch",
            ],
            config={**config, **abstract_hp, "hp_resolved": hp},
            reinit=True,
            mode=os.environ.get("WANDB_MODE", "online"),
        )
    except Exception as e:
        print(f"  W&B init failed: {e}")

    # Heartbeat callback: reports per-epoch metrics to Ray Tune (for ASHA)
    def _epoch_cb(metrics: dict[str, Any]):
        tune.report(
            {
                "epoch": metrics["epoch"],
                "val_loss": metrics["val_loss"],
                "train_loss": metrics["train_loss"],
                "test_loss": metrics["test_loss"],
                "best_val": metrics["best_val_so_far"],
            }
        )

    try:
        summary = train(args, hp, epoch_callback=_epoch_cb)
        # Final report so ASHA + searcher see the best metric
        tune.report(
            {
                "epoch": summary["best_epoch"] + 1,
                "val_loss": summary["best_val_mse"],
                "test_loss": summary["test_mse_at_best_val"],
                "best_val": summary["best_val_mse"],
                "n_params": summary["n_params"],
                "gpu_hrs": summary["gpu_hrs"],
                "done": True,
            }
        )
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass
