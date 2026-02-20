#!/usr/bin/env python
"""
Oracle Training — DREAM-RNN on full Yeast dataset.

Trains DREAM-RNN on the full ~6.06M yeast training sequences.
Establishes the upper bound performance for this architecture.
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
from torch.utils.data import DataLoader

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from albench.data.yeast import YeastDataset
from albench.models.dream_rnn import create_dream_rnn
from albench.models.loss_utils import YeastKLLoss
from albench.models.training import train_model_optimized
from albench.models.training_base import create_optimizer_and_scheduler

# ── Config ──────────────────────────────────────────────────────────────────

CONFIG = {
    "data_path": "./data/yeast",
    "batch_size": 1024,
    "num_workers": 4,
    "pin_memory": True,
    # Model
    "hidden_dim": 320,
    "cnn_filters": 160,
    "dropout_cnn": 0.1,
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
    "early_stopping_patience": None,
    "metric_for_best": "pearson_r",
    # Reproducibility
    "seed": 42,
}


def set_seed(seed: int) -> None:
    """Set all random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Oracle: DREAM-RNN on full yeast data")
    parser.add_argument("--data-path", type=str, default=CONFIG["data_path"])
    parser.add_argument("--output-dir", type=str, default="./outputs/oracle_dream_rnn_yeast")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=CONFIG["num_epochs"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument(
        "--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"]
    )
    args = parser.parse_args()

    # Setup
    CONFIG["seed"] = args.seed
    CONFIG["num_epochs"] = args.epochs
    CONFIG["data_path"] = args.data_path
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Output directory
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # W&B
    wandb.init(
        project="albench-s2f",
        name=f"oracle_dream_rnn_yeast_seed{args.seed}",
        config=CONFIG,
        tags=["oracle", "yeast", "dream_rnn"],
        mode=args.wandb_mode,
        job_type="oracle_training",
    )

    # Load Data
    print("\nLoading datasets...")
    # split='train' with subset_size=None -> loads full data
    train_dataset = YeastDataset(data_path=args.data_path, split="train", subset_size=None)
    val_dataset = YeastDataset(data_path=args.data_path, split="val")

    print(f"Training set:   {len(train_dataset):,} sequences")
    print(f"Validation set: {len(val_dataset):,} sequences")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
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

    # Loss
    criterion = YeastKLLoss(reduction="batchmean")

    # Optimizer
    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        train_loader=train_loader,
        num_epochs=CONFIG["num_epochs"],
        lr=CONFIG["lr"],
        lr_lstm=CONFIG["lr_lstm"],
        weight_decay=CONFIG["weight_decay"],
        pct_start=CONFIG["pct_start"],
    )

    # Train
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
        checkpoint_dir=output_root,  # Saves last_model.pt here
        use_reverse_complement=CONFIG["use_reverse_complement"],
        early_stopping_patience=CONFIG["early_stopping_patience"],
        metric_for_best=CONFIG["metric_for_best"],
        use_amp=CONFIG["use_amp"],
        use_compile=CONFIG["use_compile"],
    )

    elapsed = time.time() - start
    print(f"\nTraining done in {elapsed / 60:.1f} min")

    if history["val_pearson_r"]:
        print(f"Best val Pearson R:  {max(history['val_pearson_r']):.4f}")
        print(f"Best val Spearman R: {max(history['val_spearman_r']):.4f}")

    # Save summary
    summary = {
        "best_val_pearson_r": max(history["val_pearson_r"]) if history["val_pearson_r"] else 0.0,
        "best_val_spearman_r": max(history["val_spearman_r"]) if history["val_spearman_r"] else 0.0,
        "best_val_loss": min(history["val_loss"]) if history["val_loss"] else float("inf"),
        "training_time_seconds": elapsed,
        "epochs_run": len(history["val_loss"]),
    }

    with (output_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    wandb.finish()


if __name__ == "__main__":
    main()
