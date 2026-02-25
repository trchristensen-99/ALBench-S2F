#!/usr/bin/env python
"""Oracle training: DREAM-RNN on full yeast training split."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader

from data.yeast import YeastDataset
from models.dream_rnn import create_dream_rnn
from models.loss_utils import YeastKLLoss
from models.training import train_model_optimized
from models.training_base import create_optimizer_and_scheduler


def set_seed(seed: int | None) -> int:
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big") % (2**31)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="oracle_dream_rnn_yeast",
)
def main(cfg: DictConfig) -> None:
    load_dotenv()

    used_seed = set_seed(int(cfg.seed) if cfg.seed is not None else None)
    device = torch.device(f"cuda:{int(cfg.gpu)}" if torch.cuda.is_available() else "cpu")

    output_root = Path(str(cfg.output_dir))
    output_root.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project="albench-s2f",
        name=f"oracle_dream_rnn_yeast_seed{used_seed}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=["oracle", "yeast", "dream_rnn"],
        mode=str(cfg.wandb_mode),
        job_type="oracle_training",
    )

    print("\nLoading datasets...")
    ds_train = YeastDataset(
        data_path=str(cfg.data_path),
        split="train",
        subset_size=None,
        context_mode=str(cfg.context_mode),
    )
    if bool(cfg.include_pool):
        ds_pool = YeastDataset(
            data_path=str(cfg.data_path),
            split="pool",
            subset_size=None,
            context_mode=str(cfg.context_mode),
        )
        train_dataset = ConcatDataset([ds_train, ds_pool])
    else:
        train_dataset = ds_train
    val_dataset = YeastDataset(
        data_path=str(cfg.data_path),
        split="val",
        context_mode=str(cfg.context_mode),
    )

    print(f"Training set:   {len(train_dataset):,} sequences")
    print(f"Validation set: {len(val_dataset):,} sequences")

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
    )

    seq_len = ds_train.get_sequence_length()
    model = create_dream_rnn(
        input_channels=6,
        sequence_length=seq_len,
        task_mode="yeast",
        hidden_dim=int(cfg.hidden_dim),
        cnn_filters=int(cfg.cnn_filters),
        dropout_cnn=float(cfg.dropout_cnn),
        dropout_lstm=float(cfg.dropout_lstm),
    ).to(device)

    criterion = YeastKLLoss(reduction="batchmean")
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
    }

    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    wandb.finish()


if __name__ == "__main__":
    main()
