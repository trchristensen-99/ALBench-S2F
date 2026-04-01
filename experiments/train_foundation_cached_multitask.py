#!/usr/bin/env python
"""Train a SINGLE multi-task head on cached foundation model embeddings.

Predicts all 3 cell types (K562, HepG2, SKNSH) simultaneously from a shared
MLP trunk with 3 output neurons.

Architecture: LayerNorm -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear(3)

Loss: mean MSE across cell types (NaN-aware -- missing labels are masked out).

Usage::

    python experiments/train_foundation_cached_multitask.py \
        ++model_name=nt \
        ++cache_dir=outputs/nt_k562_cached/embedding_cache \
        ++embed_dim=768 \
        ++output_dir=outputs/nt_k562_multitask

    python experiments/train_foundation_cached_multitask.py \
        ++model_name=enformer \
        ++cache_dir=outputs/enformer_k562_cached/embedding_cache \
        ++embed_dim=3072 \
        ++output_dir=outputs/enformer_k562_multitask
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset

from data.k562 import K562Dataset

# ── Config ───────────────────────────────────────────────────────────────────
CELL_LINES = ["k562", "hepg2", "sknsh"]
LABEL_COLS = {
    "k562": "K562_log2FC",
    "hepg2": "HepG2_log2FC",
    "sknsh": "SKNSH_log2FC",
}

DEFAULT_CONFIG = {
    "model_name": "nt",
    "cache_dir": "outputs/nt_k562_cached/embedding_cache",
    "embed_dim": 768,
    "output_dir": "outputs/nt_k562_multitask",
    "seed": None,
    "epochs": 100,
    "batch_size": 512,
    "lr": 0.001,
    "weight_decay": 1e-6,
    "early_stop_patience": 10,
    "hidden_dim": 512,
    "dropout": 0.1,
    "data_path": "data/k562",
    "num_workers": 4,
    "rc_aug": True,
    "chr_split": False,
    "include_alt_alleles": None,
}


# ── Multi-task MLP Head ─────────────────────────────────────────────────────
class MultiTaskMLPHead(nn.Module):
    """Shared MLP trunk with 3 output neurons (one per cell type).

    LayerNorm -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear(3)
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        n_tasks: int = 3,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, n_tasks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, n_tasks)."""
        return self.head(self.trunk(x))


# ── Multi-task cached dataset ────────────────────────────────────────────────
class MultiTaskCachedEmbeddingDataset(Dataset):
    """Loads pre-computed embeddings and multi-task labels (N, 3)."""

    def __init__(
        self,
        canonical_path: str | Path,
        rc_path: str | Path,
        labels: np.ndarray,
        rc_aug: bool = True,
    ):
        """
        Args:
            canonical_path: Path to canonical embeddings (N, D) .npy
            rc_path: Path to RC embeddings (N, D) .npy
            labels: Array of shape (N, 3) with NaN for missing values
            rc_aug: If True, double dataset with RC embeddings
        """
        self.canonical = np.load(canonical_path, mmap_mode="r")
        self.rc = np.load(rc_path, mmap_mode="r")
        self.labels = labels.astype(np.float32)  # (N, 3), may contain NaN
        self.rc_aug = rc_aug
        assert len(self.canonical) == len(self.labels)

    def __len__(self) -> int:
        if self.rc_aug:
            return len(self.labels) * 2
        return len(self.labels)

    def __getitem__(self, idx: int):
        N = len(self.labels)
        if self.rc_aug and idx >= N:
            real_idx = idx - N
            emb = self.rc[real_idx].astype(np.float32)
        else:
            real_idx = idx if not self.rc_aug else idx
            emb = self.canonical[real_idx].astype(np.float32)
        label = self.labels[real_idx if self.rc_aug else idx]
        return torch.from_numpy(emb), torch.from_numpy(label.copy())


class MultiTaskValCachedDataset(Dataset):
    """Val set: returns canonical + RC for RC-averaged prediction."""

    def __init__(
        self,
        canonical_path: str | Path,
        rc_path: str | Path,
        labels: np.ndarray,
    ):
        self.canonical = np.load(canonical_path, mmap_mode="r")
        self.rc = np.load(rc_path, mmap_mode="r")
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        can = torch.from_numpy(self.canonical[idx].astype(np.float32))
        rc = torch.from_numpy(self.rc[idx].astype(np.float32))
        label = torch.from_numpy(self.labels[idx].copy())
        return can, rc, label


# ── Helpers ──────────────────────────────────────────────────────────────────
def _safe_corr(pred: np.ndarray, target: np.ndarray, fn) -> float:
    if pred.size < 2 or target.size < 2:
        return 0.0
    if np.std(pred) == 0.0 or np.std(target) == 0.0:
        return 0.0
    return float(fn(pred, target)[0])


def _load_multitask_labels(
    data_path: str,
    split: str,
    chr_split: bool = False,
    include_alt_alleles: bool = False,
) -> np.ndarray:
    """Load labels for all 3 cell types and stack into (N, 3).

    Returns:
        labels: np.ndarray of shape (N, 3), columns = [K562, HepG2, SKNSH].
                NaN where a cell type has no valid measurement.
    """
    all_labels = []
    for cell_line in CELL_LINES:
        label_col = LABEL_COLS[cell_line]
        ds_kwargs: dict = {
            "data_path": data_path,
            "label_column": label_col,
            "split": split,
            "include_alt_alleles": include_alt_alleles,
        }
        if chr_split:
            ds_kwargs["use_hashfrag"] = False
            ds_kwargs["use_chromosome_fallback"] = True
        ds = K562Dataset(**ds_kwargs)
        all_labels.append(ds.labels.astype(np.float32))

    # Stack: (N, 3)
    return np.stack(all_labels, axis=1)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss that ignores NaN targets.

    Args:
        pred: (B, 3) predictions
        target: (B, 3) targets, may contain NaN
    Returns:
        Scalar loss (mean over valid entries)
    """
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return torch.nn.functional.mse_loss(pred[mask], target[mask])


# ── Test evaluation ──────────────────────────────────────────────────────────
def evaluate_test_sets_cached_multitask(
    model: nn.Module,
    cache_dir: Path,
    data_path: Path,
    device: torch.device,
    chr_split: bool = False,
    include_alt_alleles: bool = False,
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate on test sets per cell type using cached embeddings (RC-averaged).

    Returns nested dict: {cell_line: {test_set: {metric: value}}}
    """
    import pandas as pd

    test_dir = data_path / "test_sets"
    all_metrics: dict[str, dict[str, dict[str, float]]] = {}

    def _predict_cached(prefix: str) -> np.ndarray:
        """Returns (N, 3) predictions."""
        can = np.load(cache_dir / f"{prefix}_canonical.npy", mmap_mode="r")
        rc = np.load(cache_dir / f"{prefix}_rc.npy", mmap_mode="r")
        preds = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(can), 512):
                end = min(i + 512, len(can))
                can_t = torch.from_numpy(can[i:end].astype(np.float32)).to(device)
                rc_t = torch.from_numpy(rc[i:end].astype(np.float32)).to(device)
                p_can = model(can_t)
                p_rc = model(rc_t)
                preds.append(((p_can + p_rc) / 2.0).cpu().numpy())
        return np.concatenate(preds)  # (N, 3)

    for task_idx, cell_line in enumerate(CELL_LINES):
        fc_col = LABEL_COLS[cell_line]
        cell_metrics: dict[str, dict[str, float]] = {}

        # In-distribution test
        if chr_split:
            test_ds = K562Dataset(
                data_path=str(data_path),
                split="test",
                label_column=fc_col,
                use_hashfrag=False,
                use_chromosome_fallback=True,
                include_alt_alleles=include_alt_alleles,
            )
            in_true = test_ds.labels.astype(np.float32)
            in_pred_all = _predict_cached("test_in_dist")
            in_pred = in_pred_all[:, task_idx]
            if len(in_pred) == len(in_true):
                cell_metrics["in_dist"] = {
                    "pearson_r": _safe_corr(in_pred, in_true, pearsonr),
                    "spearman_r": _safe_corr(in_pred, in_true, spearmanr),
                    "mse": float(np.mean((in_pred - in_true) ** 2)),
                }
        else:
            in_file = test_dir / "test_in_distribution_hashfrag.tsv"
            if in_file.exists():
                in_df = pd.read_csv(in_file, sep="\t")
                if fc_col in in_df.columns:
                    in_pred_all = _predict_cached("test_in_dist")
                    in_pred = in_pred_all[:, task_idx]
                    in_true = in_df[fc_col].to_numpy(dtype=np.float32)
                    cell_metrics["in_distribution"] = {
                        "pearson_r": _safe_corr(in_pred, in_true, pearsonr),
                        "spearman_r": _safe_corr(in_pred, in_true, spearmanr),
                        "mse": float(np.mean((in_pred - in_true) ** 2)),
                    }

        # OOD -- only K562 has OOD labels
        if cell_line == "k562":
            ood_file = test_dir / "test_ood_designed_k562.tsv"
            if ood_file.exists() and (cache_dir / "test_ood_canonical.npy").exists():
                ood_df = pd.read_csv(ood_file, sep="\t")
                if fc_col in ood_df.columns:
                    ood_pred_all = _predict_cached("test_ood")
                    ood_pred = ood_pred_all[:, task_idx]
                    ood_true = ood_df[fc_col].to_numpy(dtype=np.float32)
                    cell_metrics["ood"] = {
                        "pearson_r": _safe_corr(ood_pred, ood_true, pearsonr),
                        "spearman_r": _safe_corr(ood_pred, ood_true, spearmanr),
                        "mse": float(np.mean((ood_pred - ood_true) ** 2)),
                    }

        # SNV pairs
        snv_file = test_dir / "test_snv_pairs_hashfrag.tsv"
        if snv_file.exists():
            snv_df = pd.read_csv(snv_file, sep="\t")
            ref_pred_all = _predict_cached("test_snv_ref")
            alt_pred_all = _predict_cached("test_snv_alt")
            ref_pred = ref_pred_all[:, task_idx]
            alt_pred = alt_pred_all[:, task_idx]

            alt_col = f"{fc_col}_alt"
            if alt_col not in snv_df.columns:
                alt_col = "K562_log2FC_alt"
            if alt_col in snv_df.columns:
                alt_true = snv_df[alt_col].to_numpy(dtype=np.float32)
                cell_metrics["snv_abs"] = {
                    "pearson_r": _safe_corr(alt_pred, alt_true, pearsonr),
                    "spearman_r": _safe_corr(alt_pred, alt_true, spearmanr),
                    "mse": float(np.mean((alt_pred - alt_true) ** 2)),
                }

            delta_pred = alt_pred - ref_pred
            delta_col = f"delta_{fc_col}"
            if delta_col not in snv_df.columns:
                delta_col = "delta_log2FC"
            if delta_col in snv_df.columns:
                delta_true = snv_df[delta_col].to_numpy(dtype=np.float32)
                cell_metrics["snv_delta"] = {
                    "pearson_r": _safe_corr(delta_pred, delta_true, pearsonr),
                    "spearman_r": _safe_corr(delta_pred, delta_true, spearmanr),
                    "mse": float(np.mean((delta_pred - delta_true) ** 2)),
                }

        all_metrics[cell_line] = cell_metrics

    return all_metrics


# ── Training loop ────────────────────────────────────────────────────────────
def train(cfg: dict):
    seed = cfg["seed"]
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big") % (2**31)
    else:
        seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seed: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(cfg["cache_dir"])
    data_path = Path(cfg["data_path"])

    chr_split = cfg.get("chr_split", False)
    include_alt = cfg.get("include_alt_alleles")
    if include_alt is None:
        include_alt = chr_split
    elif isinstance(include_alt, str):
        include_alt = include_alt.lower() in ("true", "1", "yes")

    # Load multi-task labels (N, 3) for train and val
    print("Loading multi-task labels for all 3 cell types...")
    train_labels = _load_multitask_labels(
        str(data_path), "train", chr_split=chr_split, include_alt_alleles=include_alt
    )
    val_labels = _load_multitask_labels(
        str(data_path), "val", chr_split=chr_split, include_alt_alleles=include_alt
    )
    n_train_seqs = len(train_labels)
    n_val_seqs = len(val_labels)

    # Report NaN stats
    for i, cl in enumerate(CELL_LINES):
        n_nan_train = int(np.isnan(train_labels[:, i]).sum())
        n_nan_val = int(np.isnan(val_labels[:, i]).sum())
        print(f"  {cl}: train NaN={n_nan_train}/{n_train_seqs}, val NaN={n_nan_val}/{n_val_seqs}")

    # Create datasets from cached embeddings
    train_dataset = MultiTaskCachedEmbeddingDataset(
        cache_dir / "train_canonical.npy",
        cache_dir / "train_rc.npy",
        train_labels,
        rc_aug=cfg["rc_aug"],
    )
    val_dataset = MultiTaskValCachedDataset(
        cache_dir / "val_canonical.npy",
        cache_dir / "val_rc.npy",
        val_labels,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    embed_dim = cfg["embed_dim"]
    print(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
        f"Embed dim: {embed_dim}, Tasks: {len(CELL_LINES)}"
    )

    # Model
    model = MultiTaskMLPHead(
        embed_dim=embed_dim,
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
        n_tasks=len(CELL_LINES),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Head parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    # Output
    out_dir = Path(cfg["output_dir"]) / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_pearson = -float("inf")
    patience_counter = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        train_losses = []
        for emb, labels in train_loader:
            emb = emb.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)  # (B, 3)

            pred = model(emb)  # (B, 3)
            loss = masked_mse_loss(pred, labels)

            if torch.isnan(loss):
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate (RC-averaged)
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for can, rc, labels in val_loader:
                can = can.to(device, non_blocking=True)
                rc = rc.to(device, non_blocking=True)
                p_can = model(can)  # (B, 3)
                p_rc = model(rc)  # (B, 3)
                val_preds.append(((p_can + p_rc) / 2.0).cpu().numpy())
                val_trues.append(labels.numpy())

        val_preds = np.concatenate(val_preds)  # (N, 3)
        val_trues = np.concatenate(val_trues)  # (N, 3)

        # Compute per-cell-type Pearson, then average for early stopping
        val_pearsons = {}
        for i, cl in enumerate(CELL_LINES):
            mask = ~np.isnan(val_trues[:, i])
            if mask.sum() > 1:
                val_pearsons[cl] = _safe_corr(val_preds[mask, i], val_trues[mask, i], pearsonr)
            else:
                val_pearsons[cl] = 0.0

        mean_val_pearson = float(np.mean(list(val_pearsons.values())))
        train_loss = np.mean(train_losses) if train_losses else float("nan")

        per_cell = "  ".join(f"{cl}={val_pearsons[cl]:.4f}" for cl in CELL_LINES)
        print(
            f"Epoch {epoch + 1:3d}  train_loss={train_loss:.4f}  "
            f"val_pearson_mean={mean_val_pearson:.4f}  [{per_cell}]"
        )

        if mean_val_pearson > best_val_pearson:
            best_val_pearson = mean_val_pearson
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "seed": seed,
                },
                out_dir / "best_model.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stop_patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best and evaluate on test sets
    ckpt = torch.load(out_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    test_metrics = evaluate_test_sets_cached_multitask(
        model,
        cache_dir,
        data_path,
        device,
        chr_split=chr_split,
        include_alt_alleles=include_alt,
    )

    result = {
        "model": cfg["model_name"],
        "multitask": True,
        "seed": seed,
        "embed_dim": embed_dim,
        "hidden_dim": cfg["hidden_dim"],
        "best_val_pearson_mean": best_val_pearson,
        "best_epoch": int(ckpt["epoch"]) + 1,
        "n_train": n_train_seqs,
        "test_metrics": test_metrics,
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_dir / 'result.json'}")

    # Print summary per cell type
    for cl in CELL_LINES:
        cm = test_metrics.get(cl, {})
        id_key = "in_distribution" if "in_distribution" in cm else "in_dist"
        if id_key in cm:
            print(f"  {cl} in_dist Pearson: {cm[id_key]['pearson_r']:.4f}")
        if "ood" in cm:
            print(f"  {cl} OOD Pearson:     {cm['ood']['pearson_r']:.4f}")
        if "snv_abs" in cm:
            print(f"  {cl} SNV abs Pearson: {cm['snv_abs']['pearson_r']:.4f}")

    return result


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    cfg = dict(DEFAULT_CONFIG)
    for arg in sys.argv[1:]:
        if arg.startswith("++"):
            key, _, val = arg[2:].partition("=")
            if key in cfg:
                default_val = cfg[key]
                if default_val is None:
                    cfg[key] = None if val.lower() == "none" else val
                elif isinstance(default_val, bool):
                    cfg[key] = val.lower() in ("true", "1", "yes")
                elif isinstance(default_val, int):
                    cfg[key] = int(val)
                elif isinstance(default_val, float):
                    cfg[key] = float(val)
                else:
                    cfg[key] = val

    print("Config:", json.dumps({k: str(v) for k, v in cfg.items()}, indent=2))
    train(cfg)


if __name__ == "__main__":
    main()
