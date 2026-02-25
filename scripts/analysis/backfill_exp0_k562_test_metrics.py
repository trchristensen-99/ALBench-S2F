#!/usr/bin/env python3
"""Backfill K562 test metrics into existing exp0 result.json files.

Adds/refreshes:
- in_distribution
- snv_delta
- snv_abs
- ood
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset

from data.utils import one_hot_encode
from models.dream_rnn import create_dream_rnn


def _safe_corr(pred: np.ndarray, target: np.ndarray, fn: Any) -> float:
    if pred.size < 2 or target.size < 2:
        return 0.0
    if np.std(pred) == 0.0 or np.std(target) == 0.0:
        return 0.0
    return float(fn(pred, target)[0])


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


def _encode_k562_sequence(sequence: str) -> np.ndarray:
    sequence = _standardize_to_200bp(sequence)
    one_hot = one_hot_encode(sequence, add_singleton_channel=False)
    rc = np.zeros((1, one_hot.shape[1]), dtype=np.float32)
    return np.concatenate([one_hot, rc], axis=0)


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


def _predict_sequences(
    model: torch.nn.Module,
    sequences: list[str],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if not sequences:
        return np.asarray([], dtype=np.float32)
    ds = EncodedK562Dataset(sequences)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = model.predict(xb, use_reverse_complement=True)
            preds.append(yb.detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds, axis=0)


def _evaluate_test_sets(
    model: torch.nn.Module,
    device: torch.device,
    test_set_dir: Path,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    in_path = test_set_dir / "test_in_distribution_hashfrag.tsv"
    snv_path = test_set_dir / "test_snv_pairs_hashfrag.tsv"
    ood_path = test_set_dir / "test_ood_cre.tsv"
    missing = [str(p) for p in [in_path, snv_path, ood_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing K562 test-set files: {missing}")

    metrics: dict[str, dict[str, float]] = {}

    in_df = pd.read_csv(in_path, sep="\t")
    in_pred = _predict_sequences(model, in_df["sequence"].astype(str).tolist(), device, batch_size)
    in_true = in_df["K562_log2FC"].to_numpy(dtype=np.float32)
    metrics["in_distribution"] = {
        "pearson_r": _safe_corr(in_pred, in_true, pearsonr),
        "spearman_r": _safe_corr(in_pred, in_true, spearmanr),
        "mse": float(np.mean((in_pred - in_true) ** 2)),
    }

    snv_df = pd.read_csv(snv_path, sep="\t")
    if len(snv_df) > 0:
        ref_pred = _predict_sequences(
            model, snv_df["sequence_ref"].astype(str).tolist(), device, batch_size
        )
        alt_pred = _predict_sequences(
            model, snv_df["sequence_alt"].astype(str).tolist(), device, batch_size
        )

        snv_abs_pred = np.concatenate([ref_pred, alt_pred], axis=0)
        ref_true = snv_df["K562_log2FC_ref"].to_numpy(dtype=np.float32)
        alt_true = snv_df["K562_log2FC_alt"].to_numpy(dtype=np.float32)
        snv_abs_true = np.concatenate([ref_true, alt_true], axis=0)
        metrics["snv_abs"] = {
            "pearson_r": _safe_corr(snv_abs_pred, snv_abs_true, pearsonr),
            "spearman_r": _safe_corr(snv_abs_pred, snv_abs_true, spearmanr),
            "mse": float(np.mean((snv_abs_pred - snv_abs_true) ** 2)),
        }

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
    ood_pred = _predict_sequences(
        model, ood_df["sequence"].astype(str).tolist(), device, batch_size
    )
    ood_true = ood_df["K562_log2FC"].to_numpy(dtype=np.float32)
    metrics["ood"] = {
        "pearson_r": _safe_corr(ood_pred, ood_true, pearsonr),
        "spearman_r": _safe_corr(ood_pred, ood_true, spearmanr),
        "mse": float(np.mean((ood_pred - ood_true) ** 2)),
    }

    return metrics


def _load_checkpoint(model: torch.nn.Module, fraction_dir: Path) -> bool:
    best = fraction_dir / "best_model.pt"
    final = fraction_dir / "final_model.pt"
    ckpt_path = best if best.exists() else final
    if not ckpt_path.exists():
        return False
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("data/k562"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    result_paths = sorted(args.output_root.glob("fraction_*/result.json"))
    if not result_paths:
        raise SystemExit(f"No result.json files found under {args.output_root}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    test_set_dir = args.data_path / "test_sets"
    print(f"Device: {device}")
    print(f"Test set dir: {test_set_dir}")

    for result_path in result_paths:
        payload = json.loads(result_path.read_text())
        existing = payload.get("test_metrics", {})
        if (not args.force) and existing.get("snv_abs"):
            print(f"Skipping (already has snv_abs): {result_path}")
            continue

        model = create_dream_rnn(
            input_channels=5,
            sequence_length=200,
            task_mode="k562",
            hidden_dim=320,
            cnn_filters=160,
            dropout_cnn=0.1,
            dropout_lstm=0.1,
        ).to(device)

        if not _load_checkpoint(model, result_path.parent):
            print(f"Skipping (missing checkpoint): {result_path.parent}")
            continue
        model.to(device)

        metrics = _evaluate_test_sets(
            model=model,
            device=device,
            test_set_dir=test_set_dir,
            batch_size=args.batch_size,
        )
        payload["test_metrics"] = metrics
        result_path.write_text(json.dumps(payload, indent=2))
        print(
            f"Updated {result_path}: "
            f"in={metrics['in_distribution']['pearson_r']:.4f}, "
            f"snv_abs={metrics['snv_abs']['pearson_r']:.4f}, "
            f"snv_delta={metrics['snv_delta']['pearson_r']:.4f}, "
            f"ood={metrics['ood']['pearson_r']:.4f}"
        )


if __name__ == "__main__":
    main()
