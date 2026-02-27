#!/usr/bin/env python
"""Experiment 0 — Yeast random downsampling scaling curve."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from data.yeast import YeastDataset
from evaluation.yeast_testsets import (
    evaluate_yeast_test_subsets,
    load_yeast_test_subsets,
)
from models.dream_rnn import create_dream_rnn
from models.loss_utils import YeastKLLoss
from models.training import train_model_optimized
from models.training_base import create_optimizer_and_scheduler

CONFIG: dict[str, object] = {}


def set_seed(seed: int | None) -> int:
    """Set seed if provided, otherwise sample one from system entropy."""
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big") % (2**31)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def create_subset_indices(dataset_size: int, fraction: float, seed: int | None) -> np.ndarray:
    """Create subset indices (deterministic only when seed is provided)."""
    num_samples = max(1, int(dataset_size * fraction))
    if seed is None:
        return np.random.choice(dataset_size, size=num_samples, replace=False)
    rng = np.random.RandomState(seed)
    return rng.choice(dataset_size, size=num_samples, replace=False)


def run_fraction(
    fraction: float,
    train_dataset: Dataset,
    val_loader: DataLoader,
    device: torch.device,
    output_root: Path,
    seq_len: int,
    test_loader: DataLoader | None = None,
    test_labels: np.ndarray | None = None,
    test_subsets: dict[str, np.ndarray] | None = None,
) -> dict:
    """Train DREAM-RNN on a random subset at a given fraction of the full 6M pool."""
    n_total = len(train_dataset)
    n_samples = max(1, int(n_total * fraction))

    subset_seed = None if CONFIG["seed"] is None else int(CONFIG["seed"]) + int(fraction * 100_000)
    indices = create_subset_indices(n_total, fraction, subset_seed)
    train_subset = Subset(train_dataset, indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=int(CONFIG["batch_size"]),
        shuffle=True,
        num_workers=int(CONFIG["num_workers"]),
        pin_memory=bool(CONFIG["pin_memory"]),
    )

    model = create_dream_rnn(
        input_channels=6,
        sequence_length=seq_len,
        task_mode="yeast",
        hidden_dim=int(CONFIG["hidden_dim"]),
        cnn_filters=int(CONFIG["cnn_filters"]),
        dropout_cnn=float(CONFIG["dropout_cnn"]),
        dropout_lstm=float(CONFIG["dropout_lstm"]),
    ).to(device)

    criterion = YeastKLLoss(reduction="batchmean")

    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        train_loader=train_loader,
        num_epochs=int(CONFIG["epochs"]),
        lr=float(CONFIG["lr"]),
        lr_lstm=float(CONFIG["lr_lstm"]),
        weight_decay=float(CONFIG["weight_decay"]),
        pct_start=float(CONFIG["pct_start"]),
    )

    checkpoint_dir = output_root / f"fraction_{fraction:.4f}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    history = train_model_optimized(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=int(CONFIG["epochs"]),
        device=device,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
        use_reverse_complement=bool(CONFIG["use_reverse_complement"]),
        early_stopping_patience=CONFIG["early_stopping_patience"],
        metric_for_best=str(CONFIG["metric_for_best"]),
        use_amp=bool(CONFIG["use_amp"]),
        use_compile=bool(CONFIG["use_compile"]),
    )
    elapsed = time.time() - start

    best_val_pearson = max(history["val_pearson_r"]) if history["val_pearson_r"] else 0.0
    best_val_spearman = max(history["val_spearman_r"]) if history["val_spearman_r"] else 0.0
    best_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")

    test_metrics: dict[str, dict[str, float]] = {}
    if test_loader is not None and test_labels is not None and test_subsets is not None:
        model.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device, non_blocking=True)
                yhat = model.predict(
                    xb, use_reverse_complement=bool(CONFIG["use_reverse_complement"])
                )
                preds.append(yhat.detach().cpu().numpy().reshape(-1))
        test_predictions = np.concatenate(preds, axis=0)
        test_metrics = evaluate_yeast_test_subsets(
            predictions=test_predictions,
            labels=test_labels,
            subsets=test_subsets,
        )

    result = {
        "fraction": fraction,
        "n_samples": n_samples,
        "n_total": n_total,
        "training_time_seconds": elapsed,
        "best_val_pearson_r": best_val_pearson,
        "best_val_spearman_r": best_val_spearman,
        "best_val_loss": best_val_loss,
        "num_epochs_run": len(history["val_loss"]),
        "test_metrics": test_metrics,
    }

    with (checkpoint_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    return result


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="exp0_yeast_scaling",
)
def main(cfg: DictConfig) -> None:
    load_dotenv()
    global CONFIG
    CONFIG = OmegaConf.to_container(cfg, resolve=True)

    used_seed = set_seed(int(cfg.seed) if cfg.seed is not None else None)
    CONFIG["seed"] = cfg.seed

    if cfg.fraction is not None:
        fractions = [float(cfg.fraction)]
    elif cfg.fractions is not None:
        fractions = sorted([float(x) for x in cfg.fractions])
    else:
        fractions = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    device = torch.device(f"cuda:{int(cfg.gpu)}" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(str(cfg.output_dir)) / f"seed_{used_seed}"
    output_root.mkdir(parents=True, exist_ok=True)

    with (output_root / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({**CONFIG, "fractions": fractions, "device": str(device)}, handle, indent=2)

    wandb.init(
        project="albench-s2f",
        name=f"exp0_yeast_scaling_seed{used_seed}_{timestamp}",
        config={**CONFIG, "fractions": fractions},
        tags=["exp0", "yeast", "scaling"],
        mode=str(cfg.wandb_mode),
    )

    # Use train+pool (the full ~6M random sequences) as the base scaling pool.
    # The train split alone is only 100K sequences (FIXED_TRAIN_SIZE), which
    # would give inflated-looking fractions and depressed performance.
    ds_train = YeastDataset(
        data_path=str(cfg.data_path),
        split="train",
        context_mode=str(cfg.context_mode),
    )
    ds_pool = YeastDataset(
        data_path=str(cfg.data_path),
        split="pool",
        context_mode=str(cfg.context_mode),
    )
    train_dataset: Dataset = ConcatDataset([ds_train, ds_pool])
    seq_len = ds_train.get_sequence_length()
    val_dataset = YeastDataset(
        data_path=str(cfg.data_path),
        split="val",
        context_mode=str(cfg.context_mode),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
    )

    test_loader: DataLoader | None = None
    test_labels: np.ndarray | None = None
    test_subsets: dict[str, np.ndarray] | None = None
    default_subset_dir = Path(str(cfg.data_path)) / "test_subset_ids"
    subset_dir = Path(str(cfg.test_subset_dir)) if cfg.test_subset_dir else default_subset_dir

    if subset_dir.exists():
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
        test_labels = test_dataset.labels.astype(np.float32)
        public_dir = str(cfg.public_leaderboard_dir) if cfg.public_leaderboard_dir else None
        test_subsets = load_yeast_test_subsets(
            subset_dir=subset_dir,
            public_dir=public_dir,
            use_private_only=bool(cfg.private_only_test),
        )

    all_results: list[dict] = []
    for frac in fractions:
        fraction_dir = output_root / f"fraction_{frac:.4f}"
        result_json = fraction_dir / "result.json"

        if result_json.exists():
            with result_json.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
            all_results.append(result)
            continue

        result = run_fraction(
            frac,
            train_dataset,
            val_loader,
            device,
            output_root,
            seq_len=seq_len,
            test_loader=test_loader,
            test_labels=test_labels,
            test_subsets=test_subsets,
        )
        all_results.append(result)

        wandb.log(
            {
                "fraction": frac,
                "n_samples": result["n_samples"],
                "best_val_pearson_r": result["best_val_pearson_r"],
                "best_val_spearman_r": result["best_val_spearman_r"],
                "best_val_loss": result["best_val_loss"],
                "training_time_minutes": result["training_time_seconds"] / 60,
                "test/random/pearson_r": result.get("test_metrics", {})
                .get("random", {})
                .get("pearson_r", 0.0),
                "test/snv/pearson_r": result.get("test_metrics", {})
                .get("snv", {})
                .get("pearson_r", 0.0),
                "test/snv_abs/pearson_r": result.get("test_metrics", {})
                .get("snv_abs", {})
                .get("pearson_r", 0.0),
                "test/genomic/pearson_r": result.get("test_metrics", {})
                .get("genomic", {})
                .get("pearson_r", 0.0),
            }
        )

    with (output_root / "scaling_curve.json").open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)

    wandb.finish()


if __name__ == "__main__":
    main()
