#!/usr/bin/env python
"""Oracle training: DREAM-RNN on full K562 HashFrag train+pool set."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader

from data.k562 import K562Dataset
from models.dream_rnn import create_dream_rnn
from models.training import train_model_optimized
from models.training_base import create_optimizer_and_scheduler


def set_seed(seed: int | None) -> int:
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big") % (2**31)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="oracle_dream_rnn_k562",
)
def main(cfg: DictConfig) -> None:
    load_dotenv()

    used_seed = set_seed(int(cfg.seed) if cfg.seed is not None else None)
    device = torch.device(f"cuda:{int(cfg.gpu)}" if torch.cuda.is_available() else "cpu")

    output_root = Path(str(cfg.output_dir))
    output_root.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project="albench-s2f",
        name=f"oracle_dream_rnn_k562_seed{used_seed}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=["oracle", "k562", "dream_rnn"],
        mode=str(cfg.wandb_mode),
        job_type="oracle_training",
    )

    ds_train = K562Dataset(data_path=str(cfg.data_path), split="train")
    ds_pool = K562Dataset(data_path=str(cfg.data_path), split="pool")
    ds_val = K562Dataset(data_path=str(cfg.data_path), split="val")

    full_train = ConcatDataset([ds_train, ds_pool])

    train_loader = DataLoader(
        full_train,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
    )

    model = create_dream_rnn(
        input_channels=5,
        sequence_length=200,
        task_mode="k562",
        hidden_dim=int(cfg.hidden_dim),
        cnn_filters=int(cfg.cnn_filters),
        dropout_cnn=float(cfg.dropout_cnn),
        dropout_lstm=float(cfg.dropout_lstm),
    ).to(device)

    criterion = nn.MSELoss()
    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        train_loader=train_loader,
        num_epochs=int(cfg.epochs),
        lr=float(cfg.lr),
        lr_lstm=float(cfg.lr_lstm),
        weight_decay=float(cfg.weight_decay),
        pct_start=float(cfg.pct_start),
    )

    start = time.time()
    history = train_model_optimized(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=int(cfg.epochs),
        device=device,
        scheduler=scheduler,
        checkpoint_dir=output_root,
        use_reverse_complement=bool(cfg.use_reverse_complement),
        early_stopping_patience=cfg.early_stopping_patience,
        metric_for_best=str(cfg.metric_for_best),
        use_amp=bool(cfg.use_amp),
        use_compile=bool(cfg.use_compile),
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
