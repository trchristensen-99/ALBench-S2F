#!/usr/bin/env python
"""
Experiment 0 — Yeast random downsampling scaling curve.

Trains DREAM-RNN on random subsets of the full yeast training data at
varying fractions, evaluating on a held-out validation set.  This establishes
the baseline data-efficiency curve WITHOUT active learning.

Usage (single fraction):
    python experiments/exp0_yeast_scaling.py --fraction 0.01

Usage (all fractions):
    python experiments/exp0_yeast_scaling.py

Usage (specific GPU):
    CUDA_VISIBLE_DEVICES=2 python experiments/exp0_yeast_scaling.py --fraction 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Subset

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from albench.data.yeast import YeastDataset
from albench.models.dream_rnn import create_dream_rnn
from albench.models.loss_utils import YeastKLLoss
from albench.models.training import train_model_optimized
from albench.models.training_base import create_optimizer_and_scheduler

# ── Default config ──────────────────────────────────────────────────────────

DEFAULT_FRACTIONS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

CONFIG = {
    "data_path": "./data/yeast",
    "batch_size": 1024,
    "num_workers": 0,  # 0 avoids DataLoader RuntimeError on some hosts
    "pin_memory": True,
    # Model
    "hidden_dim": 320,
    "cnn_filters": 160,
    "dropout_cnn": 0.1,  # MC-dropout for active-learning uncertainty
    "dropout_lstm": 0.1,
    # Training
    "num_epochs": 80,
    "lr": 0.005,
    "lr_lstm": 0.001,
    "weight_decay": 0.01,
    "pct_start": 0.3,
    "use_amp": True,
    "use_compile": False,
    "use_reverse_complement": True,
    "early_stopping_patience": None,  # Train for full 80 epochs
    "metric_for_best": "pearson_r",
    # Reproducibility
    "seed": 42,
}


# ── Helpers ─────────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_subset_indices(dataset_size: int, fraction: float, seed: int) -> np.ndarray:
    """Create deterministic random subset indices."""
    rng = np.random.RandomState(seed)
    num_samples = max(1, int(dataset_size * fraction))
    return rng.choice(dataset_size, size=num_samples, replace=False)


# ── Main ────────────────────────────────────────────────────────────────────


def run_fraction(
    fraction: float,
    train_dataset: YeastDataset,
    val_loader: DataLoader,
    device: torch.device,
    output_root: Path,
    wandb_run: bool = True,
) -> dict:
    """Train DREAM-RNN on a random subset at a given fraction."""
    n_total = len(train_dataset)
    n_samples = max(1, int(n_total * fraction))

    print(f"\n{'=' * 80}")
    print(f"FRACTION {fraction:.3f} — {n_samples:,} / {n_total:,} sequences")
    print(f"{'=' * 80}")

    # Deterministic subset — seed encodes the fraction so each gets a unique
    # but reproducible subset.
    subset_seed = CONFIG["seed"] + int(fraction * 100_000)
    indices = create_subset_indices(n_total, fraction, subset_seed)
    train_subset = Subset(train_dataset, indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
    )

    # Create model
    model = create_dream_rnn(
        input_channels=6,
        sequence_length=150,
        task_mode="yeast",
        hidden_dim=CONFIG["hidden_dim"],
        cnn_filters=CONFIG["cnn_filters"],
        dropout_cnn=CONFIG["dropout_cnn"],
        dropout_lstm=CONFIG["dropout_lstm"],
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss = KL divergence for yeast 18-bin classification
    criterion = YeastKLLoss(reduction="batchmean")

    # Optimizer + OneCycleLR scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        train_loader=train_loader,
        num_epochs=CONFIG["num_epochs"],
        lr=CONFIG["lr"],
        lr_lstm=CONFIG["lr_lstm"],
        weight_decay=CONFIG["weight_decay"],
        pct_start=CONFIG["pct_start"],
    )

    # Checkpoint directory
    checkpoint_dir = output_root / f"fraction_{fraction:.4f}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    history = train_model_optimized(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=CONFIG["num_epochs"],
        device=device,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
        use_reverse_complement=CONFIG["use_reverse_complement"],
        early_stopping_patience=CONFIG["early_stopping_patience"],
        metric_for_best=CONFIG["metric_for_best"],
        use_amp=CONFIG["use_amp"],
        use_compile=CONFIG["use_compile"],
    )

    elapsed = time.time() - start

    best_val_pearson = max(history["val_pearson_r"]) if history["val_pearson_r"] else 0.0
    best_val_spearman = max(history["val_spearman_r"]) if history["val_spearman_r"] else 0.0
    best_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")

    result = {
        "fraction": fraction,
        "n_samples": n_samples,
        "n_total": n_total,
        "training_time_seconds": elapsed,
        "best_val_pearson_r": best_val_pearson,
        "best_val_spearman_r": best_val_spearman,
        "best_val_loss": best_val_loss,
        "num_epochs_run": len(history["val_loss"]),
    }

    # Save per-fraction results
    with (checkpoint_dir / "result.json").open("w") as f:
        json.dump(result, f, indent=2)

    print(f"\nFraction {fraction:.3f} done in {elapsed / 60:.1f} min")
    print(f"  Best val Pearson R:  {best_val_pearson:.4f}")
    print(f"  Best val Spearman R: {best_val_spearman:.4f}")

    return result


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Exp 0: Yeast scaling curve")
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Run a single fraction (e.g. 0.01). Omit to run all.",
    )
    parser.add_argument("--fractions", type=float, nargs="+", default=None)
    parser.add_argument("--data-path", type=str, default=CONFIG["data_path"])
    parser.add_argument("--output-dir", type=str, default="./outputs/exp0_yeast_scaling")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument(
        "--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"]
    )
    args = parser.parse_args()

    # Seed
    CONFIG["seed"] = args.seed
    set_seed(args.seed)
    if args.epochs is not None:
        CONFIG["num_epochs"] = args.epochs
    CONFIG["data_path"] = args.data_path

    # Fractions
    if args.fraction is not None:
        fractions = [args.fraction]
    elif args.fractions is not None:
        fractions = sorted(args.fractions)
    else:
        fractions = DEFAULT_FRACTIONS

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Output - use deterministic path to allow resuming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_root = Path(args.output_dir) / f"run_{timestamp}_seed{args.seed}"
    output_root = Path(args.output_dir) / f"seed_{args.seed}"
    output_root.mkdir(parents=True, exist_ok=True)

    # Save config
    with (output_root / "config.json").open("w") as f:
        json.dump({**CONFIG, "fractions": fractions, "device": str(device)}, f, indent=2)

    # W&B
    wandb.init(
        project="albench-s2f",
        name=f"exp0_yeast_scaling_seed{args.seed}_{timestamp}",
        config={**CONFIG, "fractions": fractions},
        tags=["exp0", "yeast", "scaling"],
        mode=args.wandb_mode,
    )

    # ── Load datasets ───────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_dataset = YeastDataset(data_path=CONFIG["data_path"], split="train")
    val_dataset = YeastDataset(data_path=CONFIG["data_path"], split="val")

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
    )

    print(f"Full training set: {len(train_dataset):,} sequences")
    print(f"Validation set:    {len(val_dataset):,} sequences")

    # ── Run fractions ───────────────────────────────────────────────────
    all_results: list[dict] = []
    for frac in fractions:
        # Check if already done
        fraction_dir = output_root / f"fraction_{frac:.4f}"
        result_json = fraction_dir / "result.json"

        if result_json.exists():
            print(f"\nSkipping fraction {frac} (already complete found at {result_json})")
            with result_json.open() as f:
                result = json.load(f)
            all_results.append(result)
            continue

        result = run_fraction(frac, train_dataset, val_loader, device, output_root)
        all_results.append(result)

        # Log summary to W&B
        wandb.log(
            {
                "fraction": frac,
                "n_samples": result["n_samples"],
                "best_val_pearson_r": result["best_val_pearson_r"],
                "best_val_spearman_r": result["best_val_spearman_r"],
                "best_val_loss": result["best_val_loss"],
                "training_time_minutes": result["training_time_seconds"] / 60,
            }
        )

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("EXPERIMENT 0 COMPLETE — YEAST SCALING CURVE")
    print(f"{'=' * 80}")
    print(f"{'Fraction':<10} {'N':<10} {'Time (min)':<12} {'Pearson R':<12} {'Spearman R':<12}")
    print("-" * 56)
    for r in all_results:
        print(
            f"{r['fraction']:<10.4f} "
            f"{r['n_samples']:<10,} "
            f"{r['training_time_seconds'] / 60:<12.1f} "
            f"{r['best_val_pearson_r']:<12.4f} "
            f"{r['best_val_spearman_r']:<12.4f}"
        )

    # Save final results
    with (output_root / "scaling_curve.json").open("w") as f:
        json.dump(all_results, f, indent=2)

    wandb.finish()
    print(f"\nResults saved to: {output_root}")


if __name__ == "__main__":
    main()
