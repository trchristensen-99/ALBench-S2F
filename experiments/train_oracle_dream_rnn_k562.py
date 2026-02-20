#!/usr/bin/env python
"""Oracle training: DREAM-RNN on full K562 HashFrag train+pool set."""

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
from torch.utils.data import ConcatDataset, DataLoader

from albench.data.k562 import K562Dataset
from albench.models.dream_rnn import create_dream_rnn
from albench.models.training import train_model_optimized
from albench.models.training_base import create_optimizer_and_scheduler

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


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Oracle: DREAM-RNN on full K562 HashFrag train+pool"
    )
    parser.add_argument("--data-path", type=str, default=CONFIG["data_path"])
    parser.add_argument("--output-dir", type=str, default="./outputs/oracle_dream_rnn_k562")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=CONFIG["num_epochs"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--wandb-mode", type=str, default="offline", choices=["online", "offline", "disabled"]
    )
    args = parser.parse_args()

    used_seed = set_seed(args.seed)
    CONFIG["seed"] = args.seed
    CONFIG["num_epochs"] = args.epochs
    CONFIG["data_path"] = args.data_path

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project="albench-s2f",
        name=f"oracle_dream_rnn_k562_seed{used_seed}",
        config=CONFIG,
        tags=["oracle", "k562", "dream_rnn"],
        mode=args.wandb_mode,
        job_type="oracle_training",
    )

    ds_train = K562Dataset(data_path=args.data_path, split="train")
    ds_pool = K562Dataset(data_path=args.data_path, split="pool")
    ds_val = K562Dataset(data_path=args.data_path, split="val")

    full_train = ConcatDataset([ds_train, ds_pool])

    train_loader = DataLoader(
        full_train,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
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
        checkpoint_dir=output_root,
        use_reverse_complement=CONFIG["use_reverse_complement"],
        early_stopping_patience=CONFIG["early_stopping_patience"],
        metric_for_best=CONFIG["metric_for_best"],
        use_amp=CONFIG["use_amp"],
        use_compile=CONFIG["use_compile"],
    )
    elapsed = time.time() - start

    summary = {
        "best_val_pearson_r": max(history["val_pearson_r"]) if history["val_pearson_r"] else 0.0,
        "best_val_spearman_r": max(history["val_spearman_r"]) if history["val_spearman_r"] else 0.0,
        "best_val_loss": min(history["val_loss"]) if history["val_loss"] else float("inf"),
        "training_time_seconds": elapsed,
        "epochs_run": len(history["val_loss"]),
        "n_train_total": len(full_train),
        "n_val": len(ds_val),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    wandb.finish()


if __name__ == "__main__":
    main()
