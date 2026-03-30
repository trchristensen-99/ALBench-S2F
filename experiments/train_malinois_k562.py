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
    "num_workers": 4,
    "pretrained_weights": None,
    "cell_line": "k562",
    "chr_split": False,
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
    model: nn.Module, sequences: list[str], device: torch.device, cfg: dict
) -> np.ndarray:
    if not sequences:
        return np.asarray([], dtype=np.float32)

    ds = EncodedMalinoisDataset(sequences)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device, non_blocking=True)
            out = model(xb).squeeze(-1)  # (batch,)
            if cfg["use_reverse_complement"]:
                xb_rc = xb.flip(-1)[:, [3, 2, 1, 0], :]
                out_rc = model(xb_rc).squeeze(-1)
                out = (out + out_rc) / 2.0
            preds.append(out.cpu().numpy().reshape(-1))
    return np.concatenate(preds, axis=0)


def _evaluate_chr_split_test(
    model: nn.Module,
    device: torch.device,
    test_ds: "K562MalinoisDataset",
    cfg: dict,
) -> dict[str, dict[str, float]]:
    """Evaluate on chromosome-split test set (in-dist only)."""
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
    return {
        "in_distribution": {
            "pearson_r": _safe_corr(pred, true, pearsonr),
            "spearman_r": _safe_corr(pred, true, spearmanr),
            "mse": float(np.mean((pred - true) ** 2)),
            "n": len(true),
        },
    }


def evaluate_test_sets(
    model: nn.Module, device: torch.device, test_set_dir: Path, cfg: dict
) -> dict[str, dict[str, float]]:
    cell_line = cfg.get("cell_line", "k562")
    in_path = test_set_dir / "test_in_distribution_hashfrag.tsv"
    snv_path = test_set_dir / "test_snv_pairs_hashfrag.tsv"
    ood_path = test_set_dir / f"test_ood_designed_{cell_line}.tsv"
    if not ood_path.exists():
        ood_path = test_set_dir / "test_ood_designed_k562.tsv"

    metrics: dict[str, dict[str, float]] = {}

    in_df = pd.read_csv(in_path, sep="\t")
    in_pred = _predict_sequences(model, in_df["sequence"].astype(str).tolist(), device, cfg)
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
                model, ood_df["sequence"].astype(str).tolist(), device, cfg
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
    ds_kwargs = dict(
        data_path=str(data_path),
        label_column=label_col,
        use_hashfrag=not chr_split,
        use_chromosome_fallback=chr_split,
    )
    train_ds = K562MalinoisDataset(K562Dataset(split="train", **ds_kwargs))
    val_ds = K562MalinoisDataset(K562Dataset(split="val", **ds_kwargs))

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

    # Model: BassetBranched with n_outputs=1 for K562 only
    model = BassetBranched(
        input_len=600,
        n_outputs=1,
        n_linear_layers=2,
        linear_channels=1000,
        n_branched_layers=1,
        branched_channels=250,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    if cfg["pretrained_weights"]:
        n_loaded = _load_basset_pretrained(model, cfg["pretrained_weights"])
        print(f"Loaded {n_loaded:,} pretrained conv parameters from {cfg['pretrained_weights']}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=len(train_loader) * 4, T_mult=1
    )

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
            yb = yb.to(device, non_blocking=True).unsqueeze(-1)  # (batch, 1)

            # RC augmentation: randomly reverse-complement half the batch
            if cfg["use_reverse_complement"]:
                mask = torch.rand(xb.shape[0], device=device) > 0.5
                if mask.any():
                    xb[mask] = xb[mask].flip(-1)[:, [3, 2, 1, 0], :]

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
                    pred = model(xb).squeeze(-1)
                    if cfg["use_reverse_complement"]:
                        xb_rc = xb.flip(-1)[:, [3, 2, 1, 0], :]
                        pred_rc = model(xb_rc).squeeze(-1)
                        pred = (pred + pred_rc) / 2.0
                val_preds.append(pred.cpu().numpy())
                val_trues.append(yb.numpy())

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

    # Load best and evaluate
    ckpt = torch.load(out_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    if chr_split:
        # Chr-split: evaluate on K562Dataset test split (chr7+13)
        test_ds = K562MalinoisDataset(K562Dataset(split="test", **ds_kwargs))
        test_metrics = _evaluate_chr_split_test(model, device, test_ds, cfg)
    else:
        test_dir = data_path / "test_sets"
        test_metrics = evaluate_test_sets(model, device, test_dir, cfg)

    result = {
        "model": "malinois",
        "seed": seed,
        "best_val_pearson_r": best_val_pearson,
        "best_epoch": int(ckpt["epoch"]) + 1,
        "n_train": len(train_ds),
        "test_metrics": test_metrics,
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_dir / 'result.json'}")
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
