#!/usr/bin/env python
"""Evaluate the pre-trained Malinois MVP baseline on K562 test sets.

Supports two evaluation modes:

1. **chrom_test** (always run): ALBench chromosome-based test split (chrs 7, 13) from
   K562FullDataset.  Reports MSE, Pearson R, Spearman R.

2. **hashfrag** (--test_tsv_dir): HashFrag ID / SNV_abs / SNV_delta / OOD test sets
   (same TSVs used by eval_ag.py), enabling direct apples-to-apples comparison with
   AlphaGenome adapter heads.

Output JSON structure::

    {
        "mse": ...,
        "pearson_r": ...,
        "spearman_r": ...,
        "dataset_size": ...,
        "hashfrag": {          # only present when --test_tsv_dir is given
            "in_distribution": {"pearson_r": ..., "n": ...},
            "snv_abs":         {"pearson_r": ..., "n": ...},
            "snv_delta":       {"pearson_r": ..., "n": ...},
            "ood":             {"pearson_r": ..., "n": ...}
        }
    }
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boda_dir",
        type=str,
        required=True,
        help="Path to the boda2-main repository (needed for modules & model defs).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pre-trained 'my-model.epoch_5-step_19885.pkl' tar/folder.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/k562",
        help="Path to K562 raw data directory (for chrom_test split).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/malinois_eval/result.json",
        help="Path to save the JSON metrics output.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--test_tsv_dir",
        type=str,
        default=None,
        help=(
            "If provided, also evaluate on HashFrag ID/SNV/OOD TSVs from this directory "
            "(e.g. data/k562/test_sets).  Enables direct comparison with AlphaGenome eval."
        ),
    )
    parser.add_argument(
        "--target_len",
        type=int,
        default=200,
        help="Sequence length for HashFrag evaluation (default: 200 bp).",
    )
    return parser.parse_args()


# ── Encoding helpers ──────────────────────────────────────────────────────────


def _encode_seq(seq_str: str, target_len: int) -> np.ndarray:
    """One-hot encode *seq_str* to a (4, target_len) float32 array (channels first).

    Sequences longer than *target_len* are center-cropped; shorter ones are
    center-padded with N (zero rows).
    """
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    curr_len = len(seq_str)
    if curr_len > target_len:
        start = (curr_len - target_len) // 2
        seq_str = seq_str[start : start + target_len]
    elif curr_len < target_len:
        pad = target_len - curr_len
        left = pad // 2
        seq_str = "N" * left + seq_str + "N" * (pad - left)
    arr = np.zeros((4, target_len), dtype=np.float32)
    for i, c in enumerate(seq_str):
        if c in mapping:
            arr[mapping[c], i] = 1.0
    return arr


# ── Chromosome-based dataset wrapper ─────────────────────────────────────────


class MalinoisDatasetWrapper(Dataset):
    """Wraps the ALBench K562FullDataset to emit (4, L) tensors for boda2."""

    def __init__(self, data_path: str, split: str = "test"):
        from data.k562_full import K562FullDataset

        self.k562_ds = K562FullDataset(data_path, split=split)
        self.length = len(self.k562_ds)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        seq, label = self.k562_ds[idx]
        seq_tensor = seq[:4, :]  # (4, L)
        return torch.tensor(seq_tensor, dtype=torch.float32), torch.tensor(
            label, dtype=torch.float32
        )


# ── HashFrag evaluation helpers ───────────────────────────────────────────────


def _predict_malinois(
    model, seqs_str: list[str], batch_size: int, device, target_len: int
) -> np.ndarray:
    """Run *model* on sequence strings and return K562 track (track 0) predictions."""
    tensors = torch.stack([torch.tensor(_encode_seq(s, target_len)) for s in seqs_str])
    preds: list[np.ndarray] = []
    for i in range(0, len(tensors), batch_size):
        batch = tensors[i : i + batch_size].to(device)
        with torch.no_grad():
            out = model(batch)  # (B, 3) – track 0 = K562_mean
            preds.append(out[:, 0].cpu().numpy())
    return np.concatenate(preds)


def eval_on_hashfrag(
    model,
    test_tsv_dir: str,
    batch_size: int,
    device,
    target_len: int = 200,
) -> dict:
    """Evaluate Malinois on HashFrag ID / SNV_abs / SNV_delta / OOD test sets.

    Args:
        model:        Loaded boda2 MPRA_Basset model (eval mode, on *device*).
        test_tsv_dir: Directory containing the three HashFrag TSV files.
        batch_size:   Inference batch size.
        device:       torch device.
        target_len:   Pad/crop length for input sequences.

    Returns:
        Dict with keys ``in_distribution``, ``snv_abs``, ``snv_delta``, ``ood``,
        each a dict with ``pearson_r`` and ``n``.
    """
    d = Path(test_tsv_dir)
    metrics: dict = {}

    # In-distribution
    in_df = pd.read_csv(d / "test_in_distribution_hashfrag.tsv", sep="\t")
    in_pred = _predict_malinois(
        model, in_df["sequence"].astype(str).tolist(), batch_size, device, target_len
    )
    in_true = in_df["K562_log2FC"].to_numpy(dtype=np.float32)
    metrics["in_distribution"] = {
        "pearson_r": float(pearsonr(in_pred, in_true)[0]),
        "n": int(len(in_pred)),
    }

    # SNV – absolute and delta
    snv_df = pd.read_csv(d / "test_snv_pairs_hashfrag.tsv", sep="\t")
    ref_pred = _predict_malinois(
        model, snv_df["sequence_ref"].astype(str).tolist(), batch_size, device, target_len
    )
    alt_pred = _predict_malinois(
        model, snv_df["sequence_alt"].astype(str).tolist(), batch_size, device, target_len
    )

    snv_abs_pred = np.concatenate([ref_pred, alt_pred])
    snv_abs_true = np.concatenate(
        [
            snv_df["K562_log2FC_ref"].to_numpy(dtype=np.float32),
            snv_df["K562_log2FC_alt"].to_numpy(dtype=np.float32),
        ]
    )
    metrics["snv_abs"] = {
        "pearson_r": float(pearsonr(snv_abs_pred, snv_abs_true)[0]),
        "n": int(len(snv_abs_pred)),
    }

    delta_pred = alt_pred - ref_pred
    delta_true = snv_df["delta_log2FC"].to_numpy(dtype=np.float32)
    metrics["snv_delta"] = {
        "pearson_r": float(pearsonr(delta_pred, delta_true)[0]),
        "n": int(len(delta_pred)),
    }

    # OOD
    ood_df = pd.read_csv(d / "test_ood_cre.tsv", sep="\t")
    ood_pred = _predict_malinois(
        model, ood_df["sequence"].astype(str).tolist(), batch_size, device, target_len
    )
    ood_true = ood_df["K562_log2FC"].to_numpy(dtype=np.float32)
    metrics["ood"] = {
        "pearson_r": float(pearsonr(ood_pred, ood_true)[0]),
        "n": int(len(ood_pred)),
    }

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = get_args()

    sys.path.insert(0, os.path.abspath(args.boda_dir))

    print(f"Loading pre-trained Malinois model from {args.model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from boda.model.deprecated_mpra_basset import MPRA_Basset

    model = MPRA_Basset(basset_weights_path=args.model_path)
    model.to(device)
    model.eval()

    # ── Chromosome-based test split ───────────────────────────────────────────
    print(f"Loading Full K562 ALBench test split from {args.data_path}")
    test_ds = MalinoisDatasetWrapper(args.data_path, split="test")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    all_preds: list = []
    all_targets: list = []

    print(f"Starting chromosome-based evaluation on {len(test_ds)} sequences ...")
    with torch.no_grad():
        for seqs, targets in tqdm(test_loader, desc="chrom_test"):
            seqs = seqs.to(device)
            preds = model(seqs)
            all_preds.extend(preds[:, 0].cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())

    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)

    mse = float(np.mean((y_pred - y_true) ** 2))
    r_val = float(pearsonr(y_true, y_pred)[0])
    spearman_val = float(spearmanr(y_true, y_pred)[0])

    print(f"\nChromosome-based test results:")
    print(f"  MSE:        {mse:.4f}")
    print(f"  Pearson R:  {r_val:.4f}")
    print(f"  Spearman R: {spearman_val:.4f}")

    output: dict = {
        "mse": mse,
        "pearson_r": r_val,
        "spearman_r": spearman_val,
        "dataset_size": len(test_ds),
    }

    # ── HashFrag test sets (optional) ────────────────────────────────────────
    if args.test_tsv_dir:
        print(f"\nRunning HashFrag evaluation from {args.test_tsv_dir} ...")
        hashfrag_metrics = eval_on_hashfrag(
            model, args.test_tsv_dir, args.batch_size, device, args.target_len
        )
        output["hashfrag"] = hashfrag_metrics
        print("HashFrag results:")
        for key, vals in hashfrag_metrics.items():
            print(f"  {key}: Pearson R = {vals['pearson_r']:.4f}  (n={vals['n']})")

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
