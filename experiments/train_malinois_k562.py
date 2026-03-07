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
from boda.model.basset import BassetBranched
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset

from data.k562 import K562Dataset
from data.utils import one_hot_encode

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
}


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


def evaluate_test_sets(
    model: nn.Module, device: torch.device, test_set_dir: Path, cfg: dict
) -> dict[str, dict[str, float]]:
    in_path = test_set_dir / "test_in_distribution_hashfrag.tsv"
    snv_path = test_set_dir / "test_snv_pairs_hashfrag.tsv"
    ood_path = test_set_dir / "test_ood_designed_k562.tsv"

    metrics: dict[str, dict[str, float]] = {}

    in_df = pd.read_csv(in_path, sep="\t")
    in_pred = _predict_sequences(model, in_df["sequence"].astype(str).tolist(), device, cfg)
    in_true = in_df["K562_log2FC"].to_numpy(dtype=np.float32)
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

    ood_df = pd.read_csv(ood_path, sep="\t")
    ood_pred = _predict_sequences(model, ood_df["sequence"].astype(str).tolist(), device, cfg)
    ood_true = ood_df["K562_log2FC"].to_numpy(dtype=np.float32)
    metrics["ood"] = {
        "pearson_r": _safe_corr(ood_pred, ood_true, pearsonr),
        "spearman_r": _safe_corr(ood_pred, ood_true, spearmanr),
        "mse": float(np.mean((ood_pred - ood_true) ** 2)),
    }

    return metrics


# ── Training loop ────────────────────────────────────────────────────────────
def train_malinois(cfg: dict):
    seed = cfg["seed"]
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big") % (2**31)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seed: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path(cfg["data_path"])

    # Data
    train_ds = K562MalinoisDataset(K562Dataset(data_path=str(data_path), split="train"))
    val_ds = K562MalinoisDataset(K562Dataset(data_path=str(data_path), split="val"))

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
    print(f"  OOD Pearson:     {test_metrics['ood']['pearson_r']:.4f}")
    print(f"  SNV abs Pearson: {test_metrics['snv_abs']['pearson_r']:.4f}")
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
