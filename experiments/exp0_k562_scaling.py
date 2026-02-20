#!/usr/bin/env python
"""Experiment 0 (K562): random downsampling scaling curve on HashFrag train+pool."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from torch.utils.data import ConcatDataset, DataLoader, Subset

from albench.data.k562 import K562Dataset
from albench.models.dream_rnn import create_dream_rnn
from albench.models.training import train_model_optimized
from albench.models.training_base import create_optimizer_and_scheduler

DEFAULT_FRACTIONS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]

CONFIG = {
    "data_path": "./data/k562",
    "batch_size": 1024,
    "num_workers": 4,
    "pin_memory": True,
    "hidden_dim": 320,
    "cnn_filters": 160,
    "dropout_cnn": 0.1,
    "dropout_lstm": 0.1,
    "num_epochs": 80,
    "lr": 0.005,
    "lr_lstm": 0.005,
    "weight_decay": 0.01,
    "pct_start": 0.3,
    "use_amp": True,
    "use_compile": False,
    "use_reverse_complement": True,
    "early_stopping_patience": None,
    "metric_for_best": "pearson_r",
    "seed": None,
}


def set_seed(seed: int | None) -> int:
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big") % (2**31)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def create_subset_indices(n_total: int, fraction: float, seed: int | None) -> np.ndarray:
    n = max(1, int(n_total * fraction))
    if seed is None:
        return np.random.choice(n_total, size=n, replace=False)
    rng = np.random.RandomState(seed)
    return rng.choice(n_total, size=n, replace=False)


def run_fraction(
    fraction: float,
    full_train: ConcatDataset,
    val_loader: DataLoader,
    device: torch.device,
    output_root: Path,
) -> dict:
    n_total = len(full_train)
    n_samples = max(1, int(n_total * fraction))

    subset_seed = None if CONFIG["seed"] is None else int(CONFIG["seed"]) + int(fraction * 100_000)
    indices = create_subset_indices(n_total, fraction, subset_seed)
    train_subset = Subset(full_train, indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
    )

    model = create_dream_rnn(
        input_channels=5,
        sequence_length=200,
        task_mode="k562",
        hidden_dim=CONFIG["hidden_dim"],
        cnn_filters=CONFIG["cnn_filters"],
        dropout_cnn=CONFIG["dropout_cnn"],
        dropout_lstm=CONFIG["dropout_lstm"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        train_loader=train_loader,
        num_epochs=CONFIG["num_epochs"],
        lr=CONFIG["lr"],
        lr_lstm=CONFIG["lr_lstm"],
        weight_decay=CONFIG["weight_decay"],
        pct_start=CONFIG["pct_start"],
    )

    fraction_dir = output_root / f"fraction_{fraction:.4f}"
    fraction_dir.mkdir(parents=True, exist_ok=True)

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
        checkpoint_dir=fraction_dir,
        use_reverse_complement=CONFIG["use_reverse_complement"],
        early_stopping_patience=CONFIG["early_stopping_patience"],
        metric_for_best=CONFIG["metric_for_best"],
        use_amp=CONFIG["use_amp"],
        use_compile=CONFIG["use_compile"],
    )
    elapsed = time.time() - start

    result = {
        "fraction": fraction,
        "n_samples": n_samples,
        "n_total": n_total,
        "training_time_seconds": elapsed,
        "best_val_pearson_r": max(history["val_pearson_r"]) if history["val_pearson_r"] else 0.0,
        "best_val_spearman_r": max(history["val_spearman_r"]) if history["val_spearman_r"] else 0.0,
        "best_val_loss": min(history["val_loss"]) if history["val_loss"] else float("inf"),
        "num_epochs_run": len(history["val_loss"]),
    }
    (fraction_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Exp0 K562 scaling")
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--fractions", type=float, nargs="+", default=None)
    parser.add_argument("--data-path", type=str, default=CONFIG["data_path"])
    parser.add_argument("--output-dir", type=str, default="./outputs/exp0_k562_scaling")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--wandb-mode", type=str, default="offline", choices=["online", "offline", "disabled"]
    )
    args = parser.parse_args()

    used_seed = set_seed(args.seed)
    CONFIG["seed"] = args.seed
    if args.epochs is not None:
        CONFIG["num_epochs"] = args.epochs
    CONFIG["data_path"] = args.data_path

    if args.fraction is not None:
        fractions = [args.fraction]
    elif args.fractions is not None:
        fractions = sorted(args.fractions)
    else:
        fractions = DEFAULT_FRACTIONS

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    output_root = Path(args.output_dir) / f"seed_{used_seed}"
    output_root.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project="albench-s2f",
        name=f"exp0_k562_scaling_seed{used_seed}",
        config={**CONFIG, "fractions": fractions},
        tags=["exp0", "k562", "scaling"],
        mode=args.wandb_mode,
    )

    ds_train = K562Dataset(data_path=args.data_path, split="train")
    ds_pool = K562Dataset(data_path=args.data_path, split="pool")
    ds_val = K562Dataset(data_path=args.data_path, split="val")

    full_train = ConcatDataset([ds_train, ds_pool])
    val_loader = DataLoader(
        ds_val,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
    )

    all_results = []
    for frac in fractions:
        result_json = output_root / f"fraction_{frac:.4f}" / "result.json"
        if result_json.exists():
            all_results.append(json.loads(result_json.read_text()))
            continue

        res = run_fraction(frac, full_train, val_loader, device, output_root)
        all_results.append(res)
        wandb.log(
            {
                "fraction": frac,
                "n_samples": res["n_samples"],
                "best_val_pearson_r": res["best_val_pearson_r"],
                "best_val_spearman_r": res["best_val_spearman_r"],
                "best_val_loss": res["best_val_loss"],
                "training_time_minutes": res["training_time_seconds"] / 60.0,
            }
        )

    (output_root / "scaling_curve.json").write_text(json.dumps(all_results, indent=2))
    wandb.finish()


if __name__ == "__main__":
    main()
