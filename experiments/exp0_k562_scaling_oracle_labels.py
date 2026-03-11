#!/usr/bin/env python
"""Experiment 0 (K562): DREAM-RNN oracle-label scaling curve on hashFrag.

Trains DREAM-RNN on oracle ensemble pseudolabels (instead of true labels)
at various downsampling fractions, evaluating on both oracle and real test labels.
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
from torch.utils.data import DataLoader, Dataset, Subset

from data.k562 import K562Dataset
from data.utils import one_hot_encode
from models.dream_rnn import create_dream_rnn
from models.training import train_model_optimized
from models.training_base import create_optimizer_and_scheduler

CONFIG: dict[str, object] = {}


class OracleLabelK562Dataset(Dataset):
    """Wraps a K562Dataset but replaces labels with oracle pseudolabels."""

    def __init__(self, base_dataset: K562Dataset, oracle_labels: np.ndarray):
        assert len(base_dataset) == len(oracle_labels), (
            f"Dataset size {len(base_dataset)} != oracle labels {len(oracle_labels)}"
        )
        self.base_dataset = base_dataset
        self.oracle_labels = oracle_labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        seq_tensor, _ = self.base_dataset[idx]
        label_tensor = torch.tensor(self.oracle_labels[idx], dtype=torch.float32)
        return seq_tensor, label_tensor


class EncodedK562Dataset(Dataset):
    """Tensor-backed dataset for encoded K562 sequences (test set eval)."""

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
    metrics: dict[str, dict[str, float]] = {}

    in_path = test_set_dir / "test_in_distribution_hashfrag.tsv"
    if in_path.exists():
        in_df = pd.read_csv(in_path, sep="\t")
        in_pred = _predict_sequences(model, in_df["sequence"].astype(str).tolist(), device)
        in_true = in_df["K562_log2FC"].to_numpy(dtype=np.float32)
        metrics["in_distribution"] = {
            "pearson_r": _safe_corr(in_pred, in_true, pearsonr),
            "spearman_r": _safe_corr(in_pred, in_true, spearmanr),
            "mse": float(np.mean((in_pred - in_true) ** 2)),
        }

    snv_path = test_set_dir / "test_snv_pairs_hashfrag.tsv"
    if snv_path.exists():
        snv_df = pd.read_csv(snv_path, sep="\t")
        if len(snv_df) > 0:
            ref_pred = _predict_sequences(
                model, snv_df["sequence_ref"].astype(str).tolist(), device
            )
            alt_pred = _predict_sequences(
                model, snv_df["sequence_alt"].astype(str).tolist(), device
            )
            alt_true = snv_df["K562_log2FC_alt"].to_numpy(dtype=np.float32)
            metrics["snv_abs"] = {
                "pearson_r": _safe_corr(alt_pred, alt_true, pearsonr),
                "spearman_r": _safe_corr(alt_pred, alt_true, spearmanr),
                "mse": float(np.mean((alt_pred - alt_true) ** 2)),
            }
            delta_pred = alt_pred - ref_pred
            delta_true = snv_df["delta_log2FC"].to_numpy(dtype=np.float32)
            metrics["snv_delta"] = {
                "pearson_r": _safe_corr(delta_pred, delta_true, pearsonr),
                "spearman_r": _safe_corr(delta_pred, delta_true, spearmanr),
                "mse": float(np.mean((delta_pred - delta_true) ** 2)),
            }

    ood_path = test_set_dir / "test_ood_designed_k562.tsv"
    if ood_path.exists():
        ood_df = pd.read_csv(ood_path, sep="\t")
        ood_pred = _predict_sequences(model, ood_df["sequence"].astype(str).tolist(), device)
        ood_true = ood_df["K562_log2FC"].to_numpy(dtype=np.float32)
        metrics["ood"] = {
            "pearson_r": _safe_corr(ood_pred, ood_true, pearsonr),
            "spearman_r": _safe_corr(ood_pred, ood_true, spearmanr),
            "mse": float(np.mean((ood_pred - ood_true) ** 2)),
        }

    return metrics


def _evaluate_k562_test_sets_oracle(
    model: torch.nn.Module,
    device: torch.device,
    test_set_dir: Path,
    pseudolabel_dir: Path,
) -> dict[str, dict[str, float]]:
    """Evaluate against oracle pseudolabel targets on K562 test sets."""
    metrics: dict[str, dict[str, float]] = {}

    # In-distribution: predict sequences, compare to oracle_mean
    in_path = test_set_dir / "test_in_distribution_hashfrag.tsv"
    in_oracle = pseudolabel_dir / "test_in_dist_oracle_labels.npz"
    if in_path.exists() and in_oracle.exists():
        in_df = pd.read_csv(in_path, sep="\t")
        in_pred = _predict_sequences(model, in_df["sequence"].astype(str).tolist(), device)
        oracle = np.load(in_oracle)
        in_true = oracle["oracle_mean"].astype(np.float32)
        metrics["in_distribution"] = {
            "pearson_r": _safe_corr(in_pred, in_true, pearsonr),
            "spearman_r": _safe_corr(in_pred, in_true, spearmanr),
            "mse": float(np.mean((in_pred - in_true) ** 2)),
        }

    # SNV: compare to oracle ref/alt/delta
    snv_path = test_set_dir / "test_snv_pairs_hashfrag.tsv"
    snv_oracle = pseudolabel_dir / "test_snv_oracle_labels.npz"
    if snv_path.exists() and snv_oracle.exists():
        snv_df = pd.read_csv(snv_path, sep="\t")
        if len(snv_df) > 0:
            ref_pred = _predict_sequences(
                model, snv_df["sequence_ref"].astype(str).tolist(), device
            )
            alt_pred = _predict_sequences(
                model, snv_df["sequence_alt"].astype(str).tolist(), device
            )
            oracle = np.load(snv_oracle)
            alt_true = oracle["alt_oracle_mean"].astype(np.float32)
            metrics["snv_abs"] = {
                "pearson_r": _safe_corr(alt_pred, alt_true, pearsonr),
                "spearman_r": _safe_corr(alt_pred, alt_true, spearmanr),
                "mse": float(np.mean((alt_pred - alt_true) ** 2)),
            }
            delta_pred = alt_pred - ref_pred
            delta_true = oracle["delta_oracle_mean"].astype(np.float32)
            metrics["snv_delta"] = {
                "pearson_r": _safe_corr(delta_pred, delta_true, pearsonr),
                "spearman_r": _safe_corr(delta_pred, delta_true, spearmanr),
                "mse": float(np.mean((delta_pred - delta_true) ** 2)),
            }

    # OOD
    ood_path = test_set_dir / "test_ood_designed_k562.tsv"
    ood_oracle = pseudolabel_dir / "test_ood_oracle_labels.npz"
    if ood_path.exists() and ood_oracle.exists():
        ood_df = pd.read_csv(ood_path, sep="\t")
        ood_pred = _predict_sequences(model, ood_df["sequence"].astype(str).tolist(), device)
        oracle = np.load(ood_oracle)
        ood_true = oracle["oracle_mean"].astype(np.float32)
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


def run_fraction(
    fraction: float,
    train_dataset: Dataset,
    val_loader: DataLoader,
    device: torch.device,
    output_root: Path,
    test_set_dir: Path,
    pseudolabel_dir: Path | None = None,
) -> dict:
    """Train DREAM-RNN on oracle-labeled subset at a given fraction."""
    n_total = len(train_dataset)
    n_samples = max(1, int(n_total * fraction))

    indices = np.random.choice(n_total, size=n_samples, replace=False)
    train_subset = Subset(train_dataset, indices)

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
    test_metrics_oracle: dict[str, dict[str, float]] = {}
    if pseudolabel_dir is not None:
        test_metrics_oracle = _evaluate_k562_test_sets_oracle(
            model=model,
            device=device,
            test_set_dir=test_set_dir,
            pseudolabel_dir=pseudolabel_dir,
        )

    result = {
        "fraction": fraction,
        "n_samples": n_samples,
        "n_total": n_total,
        "label_source": "oracle_pseudolabel",
        "training_time_seconds": elapsed,
        "best_val_pearson_r": max(history["val_pearson_r"]) if history["val_pearson_r"] else 0.0,
        "best_val_spearman_r": max(history["val_spearman_r"]) if history["val_spearman_r"] else 0.0,
        "best_val_loss": min(history["val_loss"]) if history["val_loss"] else float("inf"),
        "num_epochs_run": len(history["val_loss"]),
        "test_metrics": test_metrics,
        "test_metrics_oracle": test_metrics_oracle,
    }
    (fraction_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="exp0_k562_scaling_oracle_labels",
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
        fractions = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]

    device = torch.device(f"cuda:{int(cfg.gpu)}" if torch.cuda.is_available() else "cpu")

    pseudolabel_dir = Path(str(cfg.pseudolabel_dir)).expanduser().resolve()

    # Load dataset.
    ds_train = K562Dataset(data_path=str(cfg.data_path), split="train")

    # Load oracle pseudolabels.
    train_pl = np.load(pseudolabel_dir / "train_oracle_labels.npz")
    train_oracle_mean = train_pl["oracle_mean"]

    # Replace labels with oracle pseudolabels.
    train_dataset = OracleLabelK562Dataset(ds_train, train_oracle_mean)
    print(f"Oracle-labeled train: {len(train_dataset):,} sequences")

    # Validation uses oracle labels too (for loss/early stopping).
    val_pl = np.load(pseudolabel_dir / "val_oracle_labels.npz")
    ds_val = K562Dataset(data_path=str(cfg.data_path), split="val")
    val_dataset = OracleLabelK562Dataset(ds_val, val_pl["oracle_mean"])

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
    )

    test_set_dir = Path(str(cfg.data_path)) / "test_sets"

    output_root = Path(str(cfg.output_dir)) / f"seed_{used_seed}"
    output_root.mkdir(parents=True, exist_ok=True)

    with (output_root / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({**CONFIG, "fractions": fractions, "device": str(device)}, handle, indent=2)

    wandb.init(
        project="albench-s2f",
        name=f"exp0_k562_oracle_scaling_seed{used_seed}",
        config={**CONFIG, "fractions": fractions},
        tags=["exp0", "k562", "scaling", "oracle_labels"],
        mode=str(cfg.wandb_mode),
    )

    all_results: list[dict] = []
    for frac in fractions:
        result_json = output_root / f"fraction_{frac:.4f}" / "result.json"
        if result_json.exists():
            all_results.append(json.loads(result_json.read_text()))
            continue

        res = run_fraction(
            frac,
            train_dataset,
            val_loader,
            device,
            output_root,
            test_set_dir,
            pseudolabel_dir,
        )
        all_results.append(res)
        wandb.log(
            {
                "fraction": frac,
                "n_samples": res["n_samples"],
                "best_val_pearson_r": res["best_val_pearson_r"],
                "best_val_spearman_r": res["best_val_spearman_r"],
                "best_val_loss": res["best_val_loss"],
                "training_time_minutes": res["training_time_seconds"] / 60.0,
                "test/in_distribution/pearson_r": res["test_metrics"]
                .get("in_distribution", {})
                .get("pearson_r", 0.0),
                "test/snv_abs/pearson_r": res["test_metrics"]
                .get("snv_abs", {})
                .get("pearson_r", 0.0),
                "test/snv_delta/pearson_r": res["test_metrics"]
                .get("snv_delta", {})
                .get("pearson_r", 0.0),
                "test/ood/pearson_r": res["test_metrics"].get("ood", {}).get("pearson_r", 0.0),
            }
        )

    (output_root / "scaling_curve.json").write_text(json.dumps(all_results, indent=2))
    wandb.finish()


if __name__ == "__main__":
    main()
