#!/usr/bin/env python
"""Experiment 0 (K562): random downsampling scaling curve on HashFrag train+pool.

Trains DREAM-RNN at the requested fractions and evaluates each run on:
- in-domain hashfrag test set
- SNV delta test set and raw SNV expression set
- OOD CRE test set
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from data.k562 import K562Dataset
from data.utils import one_hot_encode
from models.dream_rnn import create_dream_rnn
from models.training import train_model_optimized
from models.training_base import create_optimizer_and_scheduler

CONFIG: dict[str, object] = {}


class EncodedK562Dataset(Dataset):
    """Tensor-backed dataset for encoded K562 sequences."""

    def __init__(self, sequences: list[str]):
        self.x = torch.from_numpy(
            np.stack([_encode_k562_sequence(s) for s in sequences], axis=0)
        ).float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


def _safe_corr(pred: np.ndarray, target: np.ndarray, fn: object) -> float:
    if pred.size < 2 or target.size < 2:
        return 0.0
    if np.std(pred) == 0.0 or np.std(target) == 0.0:
        return 0.0
    return float(fn(pred, target)[0])


def _encode_k562_sequence(sequence: str) -> np.ndarray:
    sequence = _standardize_to_200bp(sequence)
    one_hot = one_hot_encode(sequence, add_singleton_channel=False)
    rc = np.zeros((1, one_hot.shape[1]), dtype=np.float32)
    return np.concatenate([one_hot, rc], axis=0)


def _standardize_to_200bp(sequence: str) -> str:
    """Center-pad/truncate to 200bp to match K562Dataset preprocessing."""
    target_len = 200
    curr_len = len(sequence)
    if curr_len == target_len:
        return sequence
    if curr_len < target_len:
        pad_needed = target_len - curr_len
        left_pad = pad_needed // 2
        right_pad = pad_needed - left_pad
        return "N" * left_pad + sequence + "N" * right_pad
    start = (curr_len - target_len) // 2
    return sequence[start : start + target_len]


def _predict_sequences(
    model: torch.nn.Module, sequences: list[str], device: torch.device
) -> np.ndarray:
    if not sequences:
        return np.asarray([], dtype=np.float32)

    ds = EncodedK562Dataset(sequences)
    loader = DataLoader(
        ds,
        batch_size=int(CONFIG["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(CONFIG["pin_memory"]),
    )

    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = model.predict(xb, use_reverse_complement=bool(CONFIG["use_reverse_complement"]))
            preds.append(yb.detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds, axis=0)


def _evaluate_k562_test_sets(
    model: torch.nn.Module,
    device: torch.device,
    test_set_dir: Path,
) -> dict[str, dict[str, float]]:
    in_path = test_set_dir / "test_in_distribution_hashfrag.tsv"
    snv_path = test_set_dir / "test_snv_pairs_hashfrag.tsv"
    ood_path = test_set_dir / "test_ood_cre.tsv"

    missing = [str(p) for p in [in_path, snv_path, ood_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing K562 test-set files. Generate them first with: "
            "python scripts/create_k562_test_sets.py --data-root data/k562\n"
            f"Missing: {missing}"
        )

    metrics: dict[str, dict[str, float]] = {}

    in_df = pd.read_csv(in_path, sep="\t")
    in_pred = _predict_sequences(model, in_df["sequence"].astype(str).tolist(), device)
    in_true = in_df["K562_log2FC"].to_numpy(dtype=np.float32)
    metrics["in_distribution"] = {
        "pearson_r": _safe_corr(in_pred, in_true, pearsonr),
        "spearman_r": _safe_corr(in_pred, in_true, spearmanr),
        "mse": float(np.mean((in_pred - in_true) ** 2)),
    }

    snv_df = pd.read_csv(snv_path, sep="\t")
    if len(snv_df) > 0:
        ref_pred = _predict_sequences(model, snv_df["sequence_ref"].astype(str).tolist(), device)
        alt_pred = _predict_sequences(model, snv_df["sequence_alt"].astype(str).tolist(), device)

        # SNV absolute expression: alt-allele predictions vs alt truth only.
        # Ref sequences overlap the in-distribution test set; using alt-only avoids inflation.
        alt_true = snv_df["K562_log2FC_alt"].to_numpy(dtype=np.float32)
        metrics["snv_abs"] = {
            "pearson_r": _safe_corr(alt_pred, alt_true, pearsonr),
            "spearman_r": _safe_corr(alt_pred, alt_true, spearmanr),
            "mse": float(np.mean((alt_pred - alt_true) ** 2)),
        }

        # SNV variant-effect metric (delta alt-ref), retained for compatibility.
        delta_pred = alt_pred - ref_pred
        delta_true = snv_df["delta_log2FC"].to_numpy(dtype=np.float32)
        metrics["snv_delta"] = {
            "pearson_r": _safe_corr(delta_pred, delta_true, pearsonr),
            "spearman_r": _safe_corr(delta_pred, delta_true, spearmanr),
            "mse": float(np.mean((delta_pred - delta_true) ** 2)),
        }
    else:
        metrics["snv_abs"] = {"pearson_r": 0.0, "spearman_r": 0.0, "mse": float("inf")}
        metrics["snv_delta"] = {"pearson_r": 0.0, "spearman_r": 0.0, "mse": float("inf")}

    ood_df = pd.read_csv(ood_path, sep="\t")
    ood_pred = _predict_sequences(model, ood_df["sequence"].astype(str).tolist(), device)
    ood_true = ood_df["K562_log2FC"].to_numpy(dtype=np.float32)
    metrics["ood"] = {
        "pearson_r": _safe_corr(ood_pred, ood_true, pearsonr),
        "spearman_r": _safe_corr(ood_pred, ood_true, spearmanr),
        "mse": float(np.mean((ood_pred - ood_true) ** 2)),
    }

    return metrics


def _load_best_checkpoint(model: torch.nn.Module, checkpoint_dir: Path) -> None:
    best = checkpoint_dir / "best_model.pt"
    final = checkpoint_dir / "final_model.pt"
    if best.exists():
        ckpt = torch.load(best, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
    elif final.exists():
        ckpt = torch.load(final, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])


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
    test_set_dir: Path,
) -> dict:
    n_total = len(full_train)
    n_samples = max(1, int(n_total * fraction))

    subset_seed = None if CONFIG["seed"] is None else int(CONFIG["seed"]) + int(fraction * 100_000)
    indices = create_subset_indices(n_total, fraction, subset_seed)
    train_subset = Subset(full_train, indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=int(CONFIG["batch_size"]),
        shuffle=True,
        num_workers=int(CONFIG["num_workers"]),
        pin_memory=bool(CONFIG["pin_memory"]),
    )

    model = create_dream_rnn(
        input_channels=5,
        sequence_length=200,
        task_mode="k562",
        hidden_dim=int(CONFIG["hidden_dim"]),
        cnn_filters=int(CONFIG["cnn_filters"]),
        dropout_cnn=float(CONFIG["dropout_cnn"]),
        dropout_lstm=float(CONFIG["dropout_lstm"]),
    ).to(device)

    criterion = nn.MSELoss()
    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        train_loader=train_loader,
        num_epochs=int(CONFIG["epochs"]),
        lr=float(CONFIG["lr"]),
        lr_lstm=float(CONFIG["lr_lstm"]),
        weight_decay=float(CONFIG["weight_decay"]),
        pct_start=float(CONFIG["pct_start"]),
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
        num_epochs=int(CONFIG["epochs"]),
        device=device,
        scheduler=scheduler,
        checkpoint_dir=fraction_dir,
        use_reverse_complement=bool(CONFIG["use_reverse_complement"]),
        early_stopping_patience=CONFIG["early_stopping_patience"],
        metric_for_best=str(CONFIG["metric_for_best"]),
        use_amp=bool(CONFIG["use_amp"]),
        use_compile=bool(CONFIG["use_compile"]),
    )
    elapsed = time.time() - start

    _load_best_checkpoint(model, fraction_dir)
    model = model.to(device)
    test_metrics = _evaluate_k562_test_sets(model=model, device=device, test_set_dir=test_set_dir)

    result = {
        "fraction": fraction,
        "n_samples": n_samples,
        "n_total": n_total,
        "training_time_seconds": elapsed,
        "best_val_pearson_r": max(history["val_pearson_r"]) if history["val_pearson_r"] else 0.0,
        "best_val_spearman_r": max(history["val_spearman_r"]) if history["val_spearman_r"] else 0.0,
        "best_val_loss": min(history["val_loss"]) if history["val_loss"] else float("inf"),
        "num_epochs_run": len(history["val_loss"]),
        "test_metrics": test_metrics,
    }
    (fraction_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="exp0_k562_scaling",
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
        fractions = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]

    device = torch.device(f"cuda:{int(cfg.gpu)}" if torch.cuda.is_available() else "cpu")

    output_root = Path(str(cfg.output_dir)) / f"seed_{used_seed}"
    output_root.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project="albench-s2f",
        name=f"exp0_k562_scaling_seed{used_seed}",
        config={**CONFIG, "fractions": fractions},
        tags=["exp0", "k562", "scaling"],
        mode=str(cfg.wandb_mode),
    )

    ds_train = K562Dataset(data_path=str(cfg.data_path), split="train")
    ds_pool = K562Dataset(data_path=str(cfg.data_path), split="pool")
    ds_val = K562Dataset(data_path=str(cfg.data_path), split="val")

    full_train = ConcatDataset([ds_train, ds_pool])
    val_loader = DataLoader(
        ds_val,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
    )

    test_set_dir = Path(str(cfg.data_path)) / "test_sets"

    all_results = []
    for frac in fractions:
        result_json = output_root / f"fraction_{frac:.4f}" / "result.json"
        if result_json.exists():
            all_results.append(json.loads(result_json.read_text()))
            continue

        res = run_fraction(frac, full_train, val_loader, device, output_root, test_set_dir)
        all_results.append(res)
        wandb.log(
            {
                "fraction": frac,
                "n_samples": res["n_samples"],
                "best_val_pearson_r": res["best_val_pearson_r"],
                "best_val_spearman_r": res["best_val_spearman_r"],
                "best_val_loss": res["best_val_loss"],
                "training_time_minutes": res["training_time_seconds"] / 60.0,
                "test/in_distribution/pearson_r": res["test_metrics"]["in_distribution"][
                    "pearson_r"
                ],
                "test/snv_abs/pearson_r": res["test_metrics"]["snv_abs"]["pearson_r"],
                "test/snv_delta/pearson_r": res["test_metrics"]["snv_delta"]["pearson_r"],
                "test/ood/pearson_r": res["test_metrics"]["ood"]["pearson_r"],
            }
        )

    (output_root / "scaling_curve.json").write_text(json.dumps(all_results, indent=2))
    wandb.finish()


if __name__ == "__main__":
    main()
