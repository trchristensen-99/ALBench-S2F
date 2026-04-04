#!/usr/bin/env python
"""Train Malinois (BassetBranched) on K562 HashFrag data and evaluate on all 4 test sets.

Uses the boda2 BassetBranched architecture adapted for single-output K562 prediction.
Sequences are padded from 200bp to 600bp using MPRA flanking sequences.

Run:
    python experiments/train_malinois_k562.py [++output_dir=...] [++seed=...]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset

from data.k562 import K562Dataset
from data.utils import one_hot_encode
from models.basset_branched import BassetBranched

# ── MPRA flanking sequences (from boda2 constants) ──────────────────────────
MPRA_UPSTREAM = (
    "ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTAC"
    "GTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTAT"
    "TTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAA"
    "ACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCC"
    "TAACTGGCCGCTTGACG"
)
MPRA_DOWNSTREAM = (
    "CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCC"
    "TCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTG"
    "TTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCT"
    "GGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAG"
    "CTGACCCTGAAGTTCATCT"
)

# Pre-compute flank tensors (4, 200) each
_LEFT_FLANK = torch.from_numpy(
    one_hot_encode(MPRA_UPSTREAM[-200:], add_singleton_channel=False)
).float()
_RIGHT_FLANK = torch.from_numpy(
    one_hot_encode(MPRA_DOWNSTREAM[:200], add_singleton_channel=False)
).float()


# ── Config ───────────────────────────────────────────────────────────────────
CELL_LINE_LABEL_COLS = {
    "k562": "K562_log2FC",
    "hepg2": "HepG2_log2FC",
    "sknsh": "SKNSH_log2FC",
}

DEFAULT_CONFIG = {
    "data_path": "data/k562",
    "output_dir": "outputs/malinois_k562",
    "seed": None,
    "epochs": 200,
    "batch_size": 512,
    "lr": 0.00327,
    "weight_decay": 0.000344,
    "early_stop_patience": 15,
    "use_reverse_complement": True,
    "shift_aug": False,
    "max_shift": 15,
    "num_workers": 4,
    "lr_schedule": "cosine",  # "cosine" = CosineAnnealingWarmRestarts, "onecycle" = OneCycleLR
    "rc_mode": "flip",  # "flip" = random 50% RC per batch, "interleave" = double dataset with RC (boda2 style)
    "pretrained_weights": None,
    "cell_line": "k562",
    "chr_split": False,
    "include_alt_alleles": None,  # None = auto (True when chr_split, False otherwise)
    "duplication_cutoff": None,  # If set, duplicate training sequences with label >= cutoff
    "multitask": False,  # If True, train with 3 cell-type outputs (K562, HepG2, SknSh)
    "paper_mode": False,  # If True, use exact boda2 paper settings (overrides arch + training)
    "loss": "mse",  # "mse", "l1kl" (L1KLmixed with beta=5 across cell-type dim)
    "adam_betas": None,  # None = default (0.9, 0.999); "paper" = (0.866, 0.879)
    "adam_amsgrad": False,
    # Architecture overrides (paper_mode sets these automatically)
    "n_linear_layers": 2,
    "linear_dropout_p": 0.3,
    "n_branched_layers": 1,
    "branched_channels": 250,
    "branched_dropout_p": 0.0,
}

# Paper-exact settings (Gosai et al. 2024, Nature)
PAPER_OVERRIDES = {
    "multitask": True,
    "loss": "l1kl",
    "rc_mode": "interleave",
    "duplication_cutoff": 0.5,
    "early_stop_patience": 30,
    "adam_betas": "paper",
    "adam_amsgrad": True,
    "n_linear_layers": 1,
    "linear_dropout_p": 0.116,
    "n_branched_layers": 3,
    "branched_channels": 140,
    "branched_dropout_p": 0.576,
    "include_alt_alleles": True,
}


# ── Basset pretrained weight loading ─────────────────────────────────────────
# Weights from Zenodo (Kelley et al., 2016): Conv2d format, valid convolutions.
# Only conv+BN layers are compatible (linear layers have different input size
# due to valid vs padded convolutions: 2000 vs 2600 flatten features).
_BASSET_CONV_MAP = {
    # conv1
    "0.weight": ("conv1.conv.weight", True),
    "0.bias": ("conv1.conv.bias", False),
    "1.weight": ("conv1.bn_layer.weight", False),
    "1.bias": ("conv1.bn_layer.bias", False),
    "1.running_mean": ("conv1.bn_layer.running_mean", False),
    "1.running_var": ("conv1.bn_layer.running_var", False),
    # conv2
    "4.weight": ("conv2.conv.weight", True),
    "4.bias": ("conv2.conv.bias", False),
    "5.weight": ("conv2.bn_layer.weight", False),
    "5.bias": ("conv2.bn_layer.bias", False),
    "5.running_mean": ("conv2.bn_layer.running_mean", False),
    "5.running_var": ("conv2.bn_layer.running_var", False),
    # conv3
    "8.weight": ("conv3.conv.weight", True),
    "8.bias": ("conv3.conv.bias", False),
    "9.weight": ("conv3.bn_layer.weight", False),
    "9.bias": ("conv3.bn_layer.bias", False),
    "9.running_mean": ("conv3.bn_layer.running_mean", False),
    "9.running_var": ("conv3.bn_layer.running_var", False),
}


def _load_basset_pretrained(model: BassetBranched, weights_path: str) -> int:
    """Load Basset pretrained conv weights (Kelley 2016) into BassetBranched.

    Only loads conv1/conv2/conv3 + their BatchNorm layers. Linear layers are
    skipped because the original Basset uses valid convolutions (flatten=2000)
    while BassetBranched uses padded convolutions (flatten=2600).

    Returns number of parameters loaded.
    """
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    model_sd = model.state_dict()
    n_loaded = 0
    for src_key, (dst_key, is_conv2d) in _BASSET_CONV_MAP.items():
        if src_key not in sd:
            print(f"  WARNING: {src_key} not found in pretrained weights")
            continue
        w = sd[src_key]
        if is_conv2d:
            w = w.squeeze(-1)  # Conv2d (out, in, K, 1) → Conv1d (out, in, K)
        if w.shape != model_sd[dst_key].shape:
            print(
                f"  WARNING: shape mismatch {src_key} {w.shape} vs {dst_key} {model_sd[dst_key].shape}"
            )
            continue
        model_sd[dst_key] = w
        n_loaded += w.numel()
    model.load_state_dict(model_sd)
    return n_loaded


# ── Dataset wrapper ──────────────────────────────────────────────────────────
class K562MalinoisDataset(Dataset):
    """Wraps K562Dataset to produce (4, 600) tensors for Malinois."""

    def __init__(self, k562_ds: K562Dataset):
        self.ds = k562_ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        seq_tensor, label = self.ds[idx]
        # seq_tensor is (5, 200); strip RC channel -> (4, 200)
        seq_4ch = seq_tensor[:4]
        # Pad to 600bp with MPRA flanks
        padded = torch.cat([_LEFT_FLANK, seq_4ch, _RIGHT_FLANK], dim=-1)  # (4, 600)
        return padded, label


class K562MalinoisInterleavedDataset(Dataset):
    """Wraps K562MalinoisDataset to double with RC sequences (boda2 style).

    Index layout: [seq0, RC(seq0), seq1, RC(seq1), ...].
    RC of one-hot (4, L): flip channels [3,2,1,0] then reverse sequence dim.
    """

    def __init__(self, base_ds: K562MalinoisDataset):
        self.base = base_ds

    def __len__(self) -> int:
        return len(self.base) * 2

    def __getitem__(self, idx: int):
        real_idx = idx // 2
        padded, label = self.base[real_idx]
        if idx % 2 == 1:
            # Reverse complement: swap channels [3,2,1,0] and reverse sequence
            padded = padded[[3, 2, 1, 0], :].flip(-1)
        return padded, label


class EncodedMalinoisDataset(Dataset):
    """For test-time prediction from raw sequences."""

    def __init__(self, sequences: list[str]):
        tensors = []
        for seq in sequences:
            seq = _standardize_to_200bp(seq)
            oh = torch.from_numpy(
                one_hot_encode(seq, add_singleton_channel=False)
            ).float()  # (4, 200)
            padded = torch.cat([_LEFT_FLANK, oh, _RIGHT_FLANK], dim=-1)  # (4, 600)
            tensors.append(padded)
        self.x = torch.stack(tensors)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


class K562MalinoisMultitaskDataset(Dataset):
    """Wraps multiple K562Datasets (one per cell type) for multi-task training.

    Returns (4, 600) input tensors with labels of shape (3,) where
    missing labels are NaN.
    """

    CELL_TYPES = ["K562_log2FC", "HepG2_log2FC", "SKNSH_log2FC"]

    def __init__(self, base_ds: K562MalinoisDataset, data_path: str, split: str, **ds_kwargs):
        self.base = base_ds
        # Load extra label columns from the same split
        self._extra_labels: list[np.ndarray | None] = []
        for col in self.CELL_TYPES[1:]:  # HepG2, SknSh
            try:
                extra_ds = K562Dataset(
                    data_path=data_path,
                    split=split,
                    label_column=col,
                    **ds_kwargs,
                )
                self._extra_labels.append(extra_ds.labels)
            except Exception:
                # Column missing — fill with NaN
                self._extra_labels.append(None)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        padded, k562_label = self.base[idx]
        labels = [k562_label]
        for extra in self._extra_labels:
            if extra is not None:
                labels.append(float(extra[idx]))
            else:
                labels.append(float("nan"))
        return padded, torch.tensor(labels, dtype=torch.float32)


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


# ── Prediction / evaluation ──────────────────────────────────────────────────
def _safe_corr(pred: np.ndarray, target: np.ndarray, fn) -> float:
    if pred.size < 2 or target.size < 2:
        return 0.0
    if np.std(pred) == 0.0 or np.std(target) == 0.0:
        return 0.0
    return float(fn(pred, target)[0])


def _predict_sequences(
    model: nn.Module,
    sequences: list[str],
    device: torch.device,
    cfg: dict,
    cell_type_idx: int | None = None,
) -> np.ndarray:
    """Predict on sequences.

    Args:
        cell_type_idx: For multitask models, extract this column from the
            (N, 3) output. If None, squeeze for single-output models.
    """
    if not sequences:
        return np.asarray([], dtype=np.float32)

    ds = EncodedMalinoisDataset(sequences)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device, non_blocking=True)
            out = model(xb)
            if cell_type_idx is None:
                out = out.squeeze(-1)  # (batch,)
            if cfg["use_reverse_complement"]:
                xb_rc = xb.flip(-1)[:, [3, 2, 1, 0], :]
                out_rc = model(xb_rc)
                if cell_type_idx is None:
                    out_rc = out_rc.squeeze(-1)
                out = (out + out_rc) / 2.0
            if cell_type_idx is not None:
                preds.append(out[:, cell_type_idx].cpu().numpy().reshape(-1))
            else:
                preds.append(out.cpu().numpy().reshape(-1))
    return np.concatenate(preds, axis=0)


def _evaluate_chr_split_test(
    model: nn.Module,
    device: torch.device,
    test_ds: "K562MalinoisDataset",
    cfg: dict,
) -> dict[str, dict[str, float]]:
    """Evaluate on chromosome-split test set (in-dist + SNV + OOD)."""
    from torch.utils.data import DataLoader

    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds.append(model(x).cpu().numpy().reshape(-1))
            trues.append(y.numpy().reshape(-1))
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    metrics: dict[str, dict[str, float]] = {
        "in_distribution": {
            "pearson_r": _safe_corr(pred, true, pearsonr),
            "spearman_r": _safe_corr(pred, true, spearmanr),
            "mse": float(np.mean((pred - true) ** 2)),
            "n": len(true),
        },
    }

    # Also evaluate SNV and OOD using the hashfrag eval path
    cell_line = cfg.get("cell_line", "k562")
    test_set_dir = Path(f"data/{cell_line}/test_sets")
    if not test_set_dir.exists():
        test_set_dir = Path("data/k562/test_sets")

    fc_col = CELL_LINE_LABEL_COLS.get(cell_line, "K562_log2FC")

    # SNV (filter to chr7+13 for chr-split)
    snv_path = test_set_dir / "test_snv_pairs_hashfrag.tsv"
    if snv_path.exists():
        snv_df = pd.read_csv(snv_path, sep="\t")
        # Filter to test chromosomes if chr column exists
        if "chr" in snv_df.columns:
            snv_df = snv_df[snv_df["chr"].isin(["chr7", "chr13"])]
        if len(snv_df) > 0:
            ref_pred = _predict_sequences(
                model, snv_df["sequence_ref"].astype(str).tolist(), device, cfg
            )
            alt_pred = _predict_sequences(
                model, snv_df["sequence_alt"].astype(str).tolist(), device, cfg
            )
            alt_col = f"{fc_col}_alt"
            if alt_col not in snv_df.columns:
                alt_col = "K562_log2FC_alt"
            alt_true = snv_df[alt_col].to_numpy(dtype=np.float32)
            metrics["snv_abs"] = {
                "pearson_r": _safe_corr(alt_pred, alt_true, pearsonr),
                "spearman_r": _safe_corr(alt_pred, alt_true, spearmanr),
                "mse": float(np.mean((alt_pred - alt_true) ** 2)),
                "n": len(alt_true),
            }
            delta_pred = alt_pred - ref_pred
            delta_col = f"delta_{fc_col}"
            if delta_col not in snv_df.columns:
                delta_col = "delta_log2FC"
            delta_true = snv_df[delta_col].to_numpy(dtype=np.float32)
            metrics["snv_delta"] = {
                "pearson_r": _safe_corr(delta_pred, delta_true, pearsonr),
                "spearman_r": _safe_corr(delta_pred, delta_true, spearmanr),
                "mse": float(np.mean((delta_pred - delta_true) ** 2)),
                "n": len(delta_true),
            }

    # OOD
    ood_path = test_set_dir / f"test_ood_designed_{cell_line}.tsv"
    if not ood_path.exists():
        ood_path = test_set_dir / "test_ood_designed_k562.tsv"
    if ood_path.exists():
        ood_df = pd.read_csv(ood_path, sep="\t")
        ood_label_col = fc_col if fc_col in ood_df.columns else "K562_log2FC"
        if ood_label_col in ood_df.columns:
            ood_pred = _predict_sequences(
                model, ood_df["sequence"].astype(str).tolist(), device, cfg
            )
            ood_true = ood_df[ood_label_col].to_numpy(dtype=np.float32)
            metrics["ood"] = {
                "pearson_r": _safe_corr(ood_pred, ood_true, pearsonr),
                "spearman_r": _safe_corr(ood_pred, ood_true, spearmanr),
                "mse": float(np.mean((ood_pred - ood_true) ** 2)),
                "n": len(ood_true),
            }

    return metrics


def evaluate_test_sets(
    model: nn.Module,
    device: torch.device,
    test_set_dir: Path,
    cfg: dict,
    cell_type_idx: int | None = None,
) -> dict[str, dict[str, float]]:
    cell_line = cfg.get("cell_line", "k562")
    in_path = test_set_dir / "test_in_distribution_hashfrag.tsv"
    snv_path = test_set_dir / "test_snv_pairs_hashfrag.tsv"
    ood_path = test_set_dir / f"test_ood_designed_{cell_line}.tsv"
    if not ood_path.exists():
        ood_path = test_set_dir / "test_ood_designed_k562.tsv"

    metrics: dict[str, dict[str, float]] = {}

    in_df = pd.read_csv(in_path, sep="\t")
    in_pred = _predict_sequences(
        model, in_df["sequence"].astype(str).tolist(), device, cfg, cell_type_idx=cell_type_idx
    )
    fc_col = CELL_LINE_LABEL_COLS.get(cell_line, "K562_log2FC")
    in_true = in_df[fc_col].to_numpy(dtype=np.float32)
    metrics["in_distribution"] = {
        "pearson_r": _safe_corr(in_pred, in_true, pearsonr),
        "spearman_r": _safe_corr(in_pred, in_true, spearmanr),
        "mse": float(np.mean((in_pred - in_true) ** 2)),
    }

    snv_df = pd.read_csv(snv_path, sep="\t")
    if len(snv_df) > 0:
        ref_pred = _predict_sequences(
            model,
            snv_df["sequence_ref"].astype(str).tolist(),
            device,
            cfg,
            cell_type_idx=cell_type_idx,
        )
        alt_pred = _predict_sequences(
            model,
            snv_df["sequence_alt"].astype(str).tolist(),
            device,
            cfg,
            cell_type_idx=cell_type_idx,
        )
        alt_col = f"{fc_col}_alt"
        if alt_col not in snv_df.columns:
            alt_col = "K562_log2FC_alt"
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
        delta_true = snv_df[delta_col].to_numpy(dtype=np.float32)
        metrics["snv_delta"] = {
            "pearson_r": _safe_corr(delta_pred, delta_true, pearsonr),
            "spearman_r": _safe_corr(delta_pred, delta_true, spearmanr),
            "mse": float(np.mean((delta_pred - delta_true) ** 2)),
        }

    if ood_path.exists():
        ood_df = pd.read_csv(ood_path, sep="\t")
        if fc_col in ood_df.columns:
            ood_label_col = fc_col
        elif "K562_log2FC" in ood_df.columns:
            ood_label_col = "K562_log2FC"
        else:
            ood_label_col = None
        if ood_label_col is not None:
            ood_pred = _predict_sequences(
                model,
                ood_df["sequence"].astype(str).tolist(),
                device,
                cfg,
                cell_type_idx=cell_type_idx,
            )
            ood_true = ood_df[ood_label_col].to_numpy(dtype=np.float32)
            metrics["ood"] = {
                "pearson_r": _safe_corr(ood_pred, ood_true, pearsonr),
                "spearman_r": _safe_corr(ood_pred, ood_true, spearmanr),
                "mse": float(np.mean((ood_pred - ood_true) ** 2)),
            }

    return metrics


# ── Training loop ────────────────────────────────────────────────────────────
def train_malinois(cfg: dict):
    # Apply paper_mode overrides if requested
    paper_mode = cfg.get("paper_mode", False)
    if isinstance(paper_mode, str):
        paper_mode = paper_mode.lower() in ("true", "1", "yes")
    if paper_mode:
        print("=== PAPER MODE: using exact boda2 settings ===")
        for k, v in PAPER_OVERRIDES.items():
            cfg[k] = v

    seed = cfg["seed"]
    if seed is not None:
        seed = int(seed)
    else:
        seed = int.from_bytes(os.urandom(4), byteorder="big") % (2**31)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seed: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path(cfg["data_path"])

    # Data
    cell_line = cfg.get("cell_line", "k562")
    label_col = CELL_LINE_LABEL_COLS.get(cell_line, "K562_log2FC")
    chr_split = cfg.get("chr_split", False)
    include_alt = cfg.get("include_alt_alleles")
    if include_alt is None:
        include_alt = chr_split  # default: True for chr_split to match Malinois paper
    elif isinstance(include_alt, str):
        include_alt = include_alt.lower() in ("true", "1", "yes")
    dup_cutoff = cfg.get("duplication_cutoff")
    if dup_cutoff is not None:
        dup_cutoff = float(dup_cutoff)
    ds_kwargs = dict(
        data_path=str(data_path),
        label_column=label_col,
        use_hashfrag=not chr_split,
        use_chromosome_fallback=chr_split,
        include_alt_alleles=include_alt,
    )
    train_kwargs = {**ds_kwargs}
    if dup_cutoff is not None:
        train_kwargs["duplication_cutoff"] = dup_cutoff
    train_base_ds = K562MalinoisDataset(K562Dataset(split="train", **train_kwargs))
    multitask = cfg.get("multitask", False)
    if isinstance(multitask, str):
        multitask = multitask.lower() in ("true", "1", "yes")
    if multitask:
        mt_kwargs = {k: v for k, v in ds_kwargs.items() if k not in ("label_column", "data_path")}
        train_ds = K562MalinoisMultitaskDataset(
            train_base_ds,
            data_path=str(data_path),
            split="train",
            **mt_kwargs,
        )
        print(f"Multitask mode: training with 3 cell-type outputs")
    else:
        train_ds = train_base_ds
    rc_mode = cfg.get("rc_mode", "flip")
    if rc_mode == "interleave" and not multitask:
        train_ds = K562MalinoisInterleavedDataset(train_ds)
        print(f"RC mode: interleave (dataset doubled to {len(train_ds)})")
    else:
        print(f"RC mode: flip (random 50% RC per batch)")
    val_base_ds = K562MalinoisDataset(K562Dataset(split="val", **ds_kwargs))
    if multitask:
        val_ds = K562MalinoisMultitaskDataset(
            val_base_ds,
            data_path=str(data_path),
            split="val",
            **mt_kwargs,
        )
    else:
        val_ds = val_base_ds

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model: BassetBranched
    n_outputs = 3 if multitask else 1
    model = BassetBranched(
        input_len=600,
        n_outputs=n_outputs,
        n_linear_layers=int(cfg.get("n_linear_layers", 2)),
        linear_channels=1000,
        linear_dropout_p=float(cfg.get("linear_dropout_p", 0.3)),
        n_branched_layers=int(cfg.get("n_branched_layers", 1)),
        branched_channels=int(cfg.get("branched_channels", 250)),
        branched_dropout_p=float(cfg.get("branched_dropout_p", 0.0)),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    if cfg["pretrained_weights"]:
        n_loaded = _load_basset_pretrained(model, cfg["pretrained_weights"])
        print(f"Loaded {n_loaded:,} pretrained conv parameters from {cfg['pretrained_weights']}")

    # Loss function
    loss_type = cfg.get("loss", "mse")
    if loss_type == "l1kl":
        from models.loss_utils import L1KLMixedLoss

        criterion = L1KLMixedLoss(alpha=1.0, beta=5.0, multitask=True)
        print("Loss: L1KLmixed (alpha=1, beta=5, across cell-type dim)")
    elif multitask:
        from models.loss_utils import NaNMaskedMSELoss

        criterion = NaNMaskedMSELoss()
        print("Loss: NaN-masked MSE (multitask)")
    else:
        criterion = nn.MSELoss()

    # Optimizer
    adam_betas_cfg = cfg.get("adam_betas")
    if adam_betas_cfg == "paper":
        adam_betas = (0.866, 0.879)
    elif adam_betas_cfg and adam_betas_cfg != "None":
        adam_betas = tuple(float(x) for x in adam_betas_cfg.split(","))
    else:
        adam_betas = (0.9, 0.999)
    adam_amsgrad = cfg.get("adam_amsgrad", False)
    if isinstance(adam_amsgrad, str):
        adam_amsgrad = adam_amsgrad.lower() in ("true", "1", "yes")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        betas=adam_betas,
        amsgrad=adam_amsgrad,
    )
    if adam_betas != (0.9, 0.999) or adam_amsgrad:
        print(f"Optimizer: Adam(betas={adam_betas}, amsgrad={adam_amsgrad})")
    lr_schedule = cfg.get("lr_schedule", "cosine")
    if lr_schedule == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg["lr"],
            steps_per_epoch=len(train_loader),
            epochs=cfg["epochs"],
        )
    else:
        # Default: CosineAnnealingWarmRestarts (matching boda2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=len(train_loader) * 4, T_mult=1
        )
    print(f"LR schedule: {lr_schedule}")

    # Output
    out_dir = Path(cfg["output_dir"]) / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_pearson = -float("inf")
    patience_counter = 0
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(cfg["epochs"]):
        # Train
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            if multitask:
                yb = yb.to(device, non_blocking=True)  # (batch, 3)
            else:
                yb = yb.to(device, non_blocking=True).unsqueeze(-1)  # (batch, 1)

            # RC augmentation: randomly reverse-complement half the batch
            # Skip when rc_mode="interleave" (RC already in dataset)
            if cfg["use_reverse_complement"] and rc_mode != "interleave":
                mask = torch.rand(xb.shape[0], device=device) > 0.5
                if mask.any():
                    xb[mask] = xb[mask].flip(-1)[:, [3, 2, 1, 0], :]

            # Shift augmentation: randomly shift half the batch by ±max_shift bp
            if cfg.get("shift_aug", False):
                max_shift = int(cfg.get("max_shift", 15))
                shift_mask = torch.rand(xb.shape[0], device=device) > 0.5
                if shift_mask.any():
                    shifts = torch.randint(
                        -max_shift, max_shift + 1, (int(shift_mask.sum()),), device=device
                    )
                    for idx, s in zip(torch.where(shift_mask)[0], shifts):
                        if s != 0:
                            xb[idx] = torch.roll(xb[idx], int(s.item()), dims=-1)

            with torch.amp.autocast("cuda"):
                pred = model(xb)  # (batch, 1)
                loss = criterion(pred, yb)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    pred = model(xb)
                    if not multitask:
                        pred = pred.squeeze(-1)
                    if cfg["use_reverse_complement"]:
                        xb_rc = xb.flip(-1)[:, [3, 2, 1, 0], :]
                        pred_rc = model(xb_rc)
                        if not multitask:
                            pred_rc = pred_rc.squeeze(-1)
                        pred = (pred + pred_rc) / 2.0
                val_preds.append(pred.cpu().numpy())
                val_trues.append(yb.numpy())

        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        if multitask:
            # Use K562 (column 0) for early stopping
            val_pearson = _safe_corr(val_preds[:, 0], val_trues[:, 0], pearsonr)
        else:
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

    # Load best and evaluate
    ckpt = torch.load(out_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    if multitask:
        # Multitask evaluation: per-cell-type metrics
        cell_type_names = ["k562", "hepg2", "sknsh"]
        cell_type_cols = ["K562_log2FC", "HepG2_log2FC", "SKNSH_log2FC"]
        test_metrics: dict = {}
        for ct_idx, (ct_name, ct_col) in enumerate(zip(cell_type_names, cell_type_cols)):
            ct_cfg = {**cfg, "cell_line": ct_name}
            ct_label_col = ct_col
            if chr_split:
                try:
                    ct_ds_kwargs = {k: v for k, v in ds_kwargs.items() if k != "label_column"}
                    ct_ds_kwargs["label_column"] = ct_label_col
                    test_ds = K562MalinoisDataset(K562Dataset(split="test", **ct_ds_kwargs))
                    ct_metrics = _evaluate_chr_split_test(
                        model,
                        device,
                        test_ds,
                        ct_cfg,
                    )
                    # Override predictions to use correct cell_type_idx
                    # Re-evaluate with the right output column
                    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)
                    preds_list, trues_list = [], []
                    model.eval()
                    with torch.no_grad():
                        for x, y in loader:
                            x = x.to(device)
                            out = model(x)[:, ct_idx]
                            preds_list.append(out.cpu().numpy().reshape(-1))
                            trues_list.append(y.numpy().reshape(-1))
                    pred = np.concatenate(preds_list)
                    true = np.concatenate(trues_list)
                    ct_metrics["in_distribution"] = {
                        "pearson_r": _safe_corr(pred, true, pearsonr),
                        "spearman_r": _safe_corr(pred, true, spearmanr),
                        "mse": float(np.mean((pred - true) ** 2)),
                        "n": len(true),
                    }
                except Exception as e:
                    print(f"  Warning: could not evaluate {ct_name}: {e}")
                    ct_metrics = {}
            else:
                test_dir = data_path / "test_sets"
                ct_cfg_predict = {**cfg}
                ct_cfg_predict["cell_line"] = ct_name
                try:
                    ct_metrics = evaluate_test_sets(
                        model,
                        device,
                        test_dir,
                        ct_cfg_predict,
                        cell_type_idx=ct_idx,
                    )
                except Exception as e:
                    print(f"  Warning: could not evaluate {ct_name}: {e}")
                    ct_metrics = {}
            test_metrics[ct_name] = ct_metrics
    else:
        if chr_split:
            # Chr-split: evaluate on K562Dataset test split (chr7+13)
            test_ds = K562MalinoisDataset(K562Dataset(split="test", **ds_kwargs))
            test_metrics = _evaluate_chr_split_test(model, device, test_ds, cfg)
        else:
            test_dir = data_path / "test_sets"
            test_metrics = evaluate_test_sets(model, device, test_dir, cfg)

    result = {
        "model": "malinois" + ("_multitask" if multitask else ""),
        "seed": seed,
        "best_val_pearson_r": best_val_pearson,
        "best_epoch": int(ckpt["epoch"]) + 1,
        "n_train": len(train_ds),
        "test_metrics": test_metrics,
        "multitask": multitask,
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_dir / 'result.json'}")
    if multitask:
        for ct_name in ["k562", "hepg2", "sknsh"]:
            ct_m = test_metrics.get(ct_name, {})
            if "in_distribution" in ct_m:
                print(f"  {ct_name} in_dist Pearson: {ct_m['in_distribution']['pearson_r']:.4f}")
    else:
        print(f"  in_dist Pearson: {test_metrics['in_distribution']['pearson_r']:.4f}")
        if "ood" in test_metrics:
            print(f"  OOD Pearson:     {test_metrics['ood']['pearson_r']:.4f}")
        if "snv_abs" in test_metrics:
            print(f"  SNV abs Pearson: {test_metrics['snv_abs']['pearson_r']:.4f}")
        if "snv_delta" in test_metrics:
            print(f"  SNV delta Pearson: {test_metrics['snv_delta']['pearson_r']:.4f}")

    return result


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    cfg = dict(DEFAULT_CONFIG)
    # Parse simple ++key=value overrides from argv
    for arg in sys.argv[1:]:
        if arg.startswith("++"):
            key, _, val = arg[2:].partition("=")
            if key in cfg:
                # Type-cast to match default
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
    train_malinois(cfg)


if __name__ == "__main__":
    main()
