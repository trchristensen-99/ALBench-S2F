#!/usr/bin/env python
"""Oracle training: DREAM-RNN on yeast with k-fold CV."""

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
from torch.utils.data import DataLoader, Subset

from data.yeast import YeastDataset
from evaluation.yeast_testsets import (
    evaluate_yeast_test_subsets,
    load_yeast_test_subsets,
)
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


def _kfold_indices(n: int, n_folds: int, fold_id: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if not (0 <= fold_id < n_folds):
        raise ValueError(f"fold_id must be in [0, {n_folds - 1}], got {fold_id}")

    perm = np.random.RandomState(seed).permutation(n)
    fold_sizes = np.full(n_folds, n // n_folds, dtype=np.int64)
    fold_sizes[: n % n_folds] += 1
    start = int(np.sum(fold_sizes[:fold_id]))
    end = start + int(fold_sizes[fold_id])
    val_idx = perm[start:end]
    train_idx = np.concatenate([perm[:start], perm[end:]])
    return train_idx, val_idx


def _predict_test(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_reverse_complement: bool,
) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            yhat = model.predict(xb, use_reverse_complement=use_reverse_complement)
            preds.append(yhat.detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds, axis=0)


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

    n_folds = int(cfg.n_folds)
    fold_id = int(cfg.fold_id)
    fold_split_seed = int(cfg.fold_split_seed)
    train_idx, val_idx = _kfold_indices(len(ds_train), n_folds, fold_id, fold_split_seed)
    train_dataset = Subset(ds_train, train_idx)
    val_dataset = Subset(ds_train, val_idx)

    print(
        f"Oracle k-fold split: fold {fold_id}/{n_folds} | "
        f"train={len(train_dataset):,} val={len(val_dataset):,} "
        f"(from {len(ds_train):,} total)"
    )

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
        "fold_id": fold_id,
        "n_folds": n_folds,
        "fold_split_seed": fold_split_seed,
        "n_train_samples": len(train_dataset),
        "n_val_samples": len(val_dataset),
        "n_total_train": len(ds_train),
        "best_val_pearson_r": max(history["val_pearson_r"]) if history["val_pearson_r"] else 0.0,
        "best_val_spearman_r": max(history["val_spearman_r"]) if history["val_spearman_r"] else 0.0,
        "best_val_loss": min(history["val_loss"]) if history["val_loss"] else float("inf"),
        "training_time_seconds": elapsed,
        "epochs_run": len(history["val_loss"]),
    }

    # Evaluate best checkpoint on fixed yeast test subsets (random/genomic/snv/snv_abs).
    best_ckpt = output_root / "best_model.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device)

        test_dataset = YeastDataset(
            data_path=str(cfg.data_path),
            split="test",
            context_mode=str(cfg.context_mode),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=int(cfg.num_workers),
            pin_memory=bool(cfg.pin_memory),
        )

        default_subset_dir = Path(str(cfg.data_path)) / "test_subset_ids"
        subset_dir = (
            Path(str(cfg.test_subset_dir))
            if cfg.test_subset_dir is not None
            else default_subset_dir
        )
        if subset_dir.exists():
            public_dir = (
                str(cfg.public_leaderboard_dir) if cfg.public_leaderboard_dir is not None else None
            )
            subsets = load_yeast_test_subsets(
                subset_dir=subset_dir,
                public_dir=public_dir,
                use_private_only=bool(cfg.private_only_test),
            )
            preds = _predict_test(
                model,
                test_loader,
                device,
                use_reverse_complement=bool(cfg.use_reverse_complement),
            )
            summary["test_metrics"] = evaluate_yeast_test_subsets(
                predictions=preds,
                labels=test_dataset.labels.astype(np.float32),
                subsets=subsets,
            )

    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    wandb.finish()


if __name__ == "__main__":
    main()
