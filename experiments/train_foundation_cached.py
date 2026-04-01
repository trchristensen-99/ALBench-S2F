#!/usr/bin/env python
"""Train a head on cached foundation model embeddings for K562 MPRA prediction.

Works with any model whose embeddings are pre-cached as numpy arrays:
Enformer (3072D), Borzoi (1536D), Nucleotide Transformer (768D).

Head architecture: LayerNorm → Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear(1)

Usage::

    # NT (3 seeds)
    python experiments/train_foundation_cached.py \
        ++model_name=nt \
        ++cache_dir=outputs/nt_k562_cached/embedding_cache \
        ++embed_dim=768 \
        ++output_dir=outputs/nt_k562_cached

    # Enformer
    python experiments/train_foundation_cached.py \
        ++model_name=enformer \
        ++cache_dir=outputs/enformer_k562_cached/embedding_cache \
        ++embed_dim=3072 \
        ++output_dir=outputs/enformer_k562_cached

    # Borzoi
    python experiments/train_foundation_cached.py \
        ++model_name=borzoi \
        ++cache_dir=outputs/borzoi_k562_cached/embedding_cache \
        ++embed_dim=1536 \
        ++output_dir=outputs/borzoi_k562_cached
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
CELL_LINE_LABEL_COLS = {
    "k562": "K562_log2FC",
    "hepg2": "HepG2_log2FC",
    "sknsh": "SKNSH_log2FC",
}

DEFAULT_CONFIG = {
    "model_name": "nt",
    "cache_dir": "outputs/nt_k562_cached/embedding_cache",
    "embed_dim": 768,
    "output_dir": "outputs/nt_k562_cached",
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
    "cell_line": "k562",
    "chr_split": False,
    "include_alt_alleles": None,  # None = auto (True when chr_split, False otherwise)
    "duplication_cutoff": None,  # If set, duplicate training sequences with label >= cutoff
}


# ── MLP Head ─────────────────────────────────────────────────────────────────
class MLPHead(nn.Module):
    """LayerNorm → Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear(1)."""

    def __init__(self, embed_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Cached dataset ───────────────────────────────────────────────────────────
class CachedEmbeddingDataset(Dataset):
    """Loads pre-computed embeddings (N, D) from numpy files."""

    def __init__(
        self,
        canonical_path: str | Path,
        rc_path: str | Path,
        labels: np.ndarray,
        rc_aug: bool = True,
    ):
        self.canonical = np.load(canonical_path, mmap_mode="r")
        self.rc = np.load(rc_path, mmap_mode="r")
        self.labels = labels.astype(np.float32)
        self.rc_aug = rc_aug
        assert len(self.canonical) == len(self.labels)

    def __len__(self) -> int:
        # If RC aug: doubled length (canonical then RC)
        if self.rc_aug:
            return len(self.labels) * 2
        return len(self.labels)

    def __getitem__(self, idx: int):
        N = len(self.labels)
        if self.rc_aug and idx >= N:
            # RC sample
            real_idx = idx - N
            emb = self.rc[real_idx].astype(np.float32)
        else:
            real_idx = idx if not self.rc_aug else idx
            emb = self.canonical[real_idx].astype(np.float32)
        label = self.labels[real_idx if self.rc_aug else idx]
        return torch.from_numpy(emb), torch.tensor(label, dtype=torch.float32)


class ValCachedDataset(Dataset):
    """Val set: returns both canonical and RC for RC-averaged prediction."""

    def __init__(self, canonical_path: str | Path, rc_path: str | Path, labels: np.ndarray):
        self.canonical = np.load(canonical_path, mmap_mode="r")
        self.rc = np.load(rc_path, mmap_mode="r")
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        can = torch.from_numpy(self.canonical[idx].astype(np.float32))
        rc = torch.from_numpy(self.rc[idx].astype(np.float32))
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return can, rc, label


# ── Helpers ──────────────────────────────────────────────────────────────────
def _safe_corr(pred: np.ndarray, target: np.ndarray, fn) -> float:
    if pred.size < 2 or target.size < 2:
        return 0.0
    if np.std(pred) == 0.0 or np.std(target) == 0.0:
        return 0.0
    return float(fn(pred, target)[0])


# ── Test evaluation using cached embeddings ──────────────────────────────────
def evaluate_test_sets_cached(
    model: nn.Module,
    cache_dir: Path,
    data_path: Path,
    device: torch.device,
    cell_line: str = "k562",
    chr_split: bool = False,
    include_alt_alleles: bool = False,
) -> dict[str, dict[str, float]]:
    """Evaluate on test sets using cached embeddings (RC-averaged).

    For chr_split mode, uses K562Dataset test split (chr7+13) for in-dist
    and filters SNV pairs to chr7+13 chromosomes only.
    """
    import pandas as pd

    fc_col = CELL_LINE_LABEL_COLS[cell_line]
    test_dir = data_path / "test_sets"
    metrics: dict[str, dict[str, float]] = {}

    def _predict_cached(prefix: str) -> np.ndarray:
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
        return np.concatenate(preds)

    if chr_split:
        # For chr_split: in-dist test = chr7+13 from K562Dataset
        # The cache was built from K562Dataset(split="test", chr_split=True)
        # Labels come from the same dataset
        from data.k562 import K562Dataset

        test_ds = K562Dataset(
            data_path=str(data_path),
            split="test",
            label_column=fc_col,
            use_hashfrag=False,
            use_chromosome_fallback=True,
            include_alt_alleles=include_alt_alleles,
        )
        in_true = test_ds.labels.astype(np.float32)
        in_pred = _predict_cached("test_in_dist")
        # Verify lengths match
        if len(in_pred) == len(in_true):
            metrics["in_dist"] = {
                "pearson_r": _safe_corr(in_pred, in_true, pearsonr),
                "spearman_r": _safe_corr(in_pred, in_true, spearmanr),
                "mse": float(np.mean((in_pred - in_true) ** 2)),
            }
        else:
            print(
                f"WARNING: in_dist length mismatch: pred={len(in_pred)} vs true={len(in_true)}. "
                f"Falling back to hashfrag TSV labels.",
                flush=True,
            )
            in_df = pd.read_csv(test_dir / "test_in_distribution_hashfrag.tsv", sep="\t")
            in_true = in_df[fc_col].to_numpy(dtype=np.float32)
            metrics["in_dist"] = {
                "pearson_r": _safe_corr(in_pred, in_true, pearsonr),
                "spearman_r": _safe_corr(in_pred, in_true, spearmanr),
                "mse": float(np.mean((in_pred - in_true) ** 2)),
            }
    else:
        # HashFrag: use TSV labels
        in_df = pd.read_csv(test_dir / "test_in_distribution_hashfrag.tsv", sep="\t")
        in_pred = _predict_cached("test_in_dist")
        in_true = in_df[fc_col].to_numpy(dtype=np.float32)
        metrics["in_distribution"] = {
            "pearson_r": _safe_corr(in_pred, in_true, pearsonr),
            "spearman_r": _safe_corr(in_pred, in_true, spearmanr),
            "mse": float(np.mean((in_pred - in_true) ** 2)),
        }

    # SNV pairs
    snv_df = pd.read_csv(test_dir / "test_snv_pairs_hashfrag.tsv", sep="\t")
    # For chr_split, filter to chr7+13 only
    if chr_split and "IDs_ref" in snv_df.columns:
        test_chrs = {"7", "13", "chr7", "chr13"}
        chroms = snv_df["IDs_ref"].str.split(":", expand=True)[0]
        chr_mask = chroms.isin(test_chrs)
        n_before = len(snv_df)
        snv_df = snv_df[chr_mask].reset_index(drop=True)
        print(f"Chr-split SNV filter: {n_before} -> {len(snv_df)} (chr7+13 only)", flush=True)
    ref_pred = _predict_cached("test_snv_ref")
    alt_pred = _predict_cached("test_snv_alt")
    # For chr_split, we need to filter predictions to match filtered SNV df
    if chr_split and "IDs_ref" in snv_df.columns:
        # The cached predictions are for ALL SNV pairs; we need the filtered subset
        snv_full = pd.read_csv(test_dir / "test_snv_pairs_hashfrag.tsv", sep="\t")
        chroms_full = snv_full["IDs_ref"].str.split(":", expand=True)[0]
        chr_mask_full = chroms_full.isin(test_chrs)
        ref_pred = ref_pred[chr_mask_full.to_numpy()]
        alt_pred = alt_pred[chr_mask_full.to_numpy()]
    # Try cell-line-specific alt column, fall back to generic
    alt_col = f"{fc_col}_alt"
    if alt_col not in snv_df.columns:
        alt_col = "K562_log2FC_alt"  # fallback for K562
    if alt_col in snv_df.columns:
        alt_true = snv_df[alt_col].to_numpy(dtype=np.float32)
        metrics["snv_abs"] = {
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
        metrics["snv_delta"] = {
            "pearson_r": _safe_corr(delta_pred, delta_true, pearsonr),
            "spearman_r": _safe_corr(delta_pred, delta_true, spearmanr),
            "mse": float(np.mean((delta_pred - delta_true) ** 2)),
        }

    # OOD — only evaluate when cell_line has matching labels (K562 only)
    ood_file = test_dir / f"test_ood_designed_{cell_line}.tsv"
    if not ood_file.exists() and cell_line == "k562":
        ood_file = test_dir / "test_ood_designed_k562.tsv"
    if ood_file.exists() and (cache_dir / "test_ood_canonical.npy").exists():
        ood_df = pd.read_csv(ood_file, sep="\t")
        if fc_col in ood_df.columns:
            ood_true = ood_df[fc_col].to_numpy(dtype=np.float32)
        elif cell_line == "k562" and "K562_log2FC" in ood_df.columns:
            ood_true = ood_df["K562_log2FC"].to_numpy(dtype=np.float32)
        else:
            ood_true = None
            print(
                f"Skipping OOD eval for {cell_line}: no matching labels in {ood_file.name}",
                flush=True,
            )
        if ood_true is not None:
            ood_pred = _predict_cached("test_ood")
            metrics["ood"] = {
                "pearson_r": _safe_corr(ood_pred, ood_true, pearsonr),
                "spearman_r": _safe_corr(ood_pred, ood_true, spearmanr),
                "mse": float(np.mean((ood_pred - ood_true) ** 2)),
            }

    return metrics


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

    # Load labels (cell-line-specific)
    cell_line = cfg.get("cell_line", "k562")
    label_col = CELL_LINE_LABEL_COLS.get(cell_line, "K562_log2FC")
    chr_split = cfg.get("chr_split", False)
    include_alt = cfg.get("include_alt_alleles")
    if include_alt is None:
        include_alt = chr_split  # default: True for chr_split to match Malinois paper
    elif isinstance(include_alt, str):
        include_alt = include_alt.lower() in ("true", "1", "yes")
    ds_kwargs: dict = {"data_path": str(data_path), "label_column": label_col}
    if chr_split:
        ds_kwargs["use_hashfrag"] = False
        ds_kwargs["use_chromosome_fallback"] = True
    ds_kwargs["include_alt_alleles"] = include_alt
    dup_cutoff = cfg.get("duplication_cutoff")
    if dup_cutoff is not None:
        dup_cutoff = float(dup_cutoff)
        ds_kwargs["duplication_cutoff"] = dup_cutoff
    train_ds = K562Dataset(split="train", **ds_kwargs)
    val_ds = K562Dataset(split="val", **ds_kwargs)
    train_labels = train_ds.labels
    val_labels = val_ds.labels

    # Create datasets from cached embeddings
    train_dataset = CachedEmbeddingDataset(
        cache_dir / "train_canonical.npy",
        cache_dir / "train_rc.npy",
        train_labels,
        rc_aug=cfg["rc_aug"],
    )
    val_dataset = ValCachedDataset(
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
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Embed dim: {embed_dim}")

    # Model
    model = MLPHead(
        embed_dim=embed_dim,
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Head parameters: {n_params:,}")

    criterion = nn.MSELoss()
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
            labels = labels.to(device, non_blocking=True)

            pred = model(emb)
            loss = criterion(pred, labels)

            if torch.isnan(loss):
                continue  # skip NaN batches (from corrupted cache rows)

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
                p_can = model(can)
                p_rc = model(rc)
                val_preds.append(((p_can + p_rc) / 2.0).cpu().numpy())
                val_trues.append(labels.numpy())

        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        val_pearson = _safe_corr(val_preds, val_trues, pearsonr)
        train_loss = np.mean(train_losses)

        print(
            f"Epoch {epoch + 1:3d}  train_loss={train_loss:.4f}  "
            f"val_pearson={val_pearson:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_pearson > best_val_pearson:
            best_val_pearson = val_pearson
            patience_counter = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "seed": seed},
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

    test_metrics = evaluate_test_sets_cached(
        model,
        cache_dir,
        data_path,
        device,
        cell_line=cell_line,
        chr_split=chr_split,
        include_alt_alleles=include_alt,
    )

    result = {
        "model": cfg["model_name"],
        "seed": seed,
        "embed_dim": embed_dim,
        "hidden_dim": cfg["hidden_dim"],
        "best_val_pearson_r": best_val_pearson,
        "best_epoch": int(ckpt["epoch"]) + 1,
        "n_train": len(train_ds),
        "test_metrics": test_metrics,
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_dir / 'result.json'}")
    print(f"  in_dist Pearson:   {test_metrics['in_distribution']['pearson_r']:.4f}")
    if "ood" in test_metrics:
        print(f"  OOD Pearson:       {test_metrics['ood']['pearson_r']:.4f}")
    if "snv_abs" in test_metrics:
        print(f"  SNV abs Pearson:   {test_metrics['snv_abs']['pearson_r']:.4f}")
    if "snv_delta" in test_metrics:
        print(f"  SNV delta Pearson: {test_metrics['snv_delta']['pearson_r']:.4f}")

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
