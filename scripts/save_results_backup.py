#!/usr/bin/env python
"""Save test predictions and back up lightweight results.

This module provides:
1. save_test_predictions() - saves pred vs true arrays as .npz
2. backup_results() - copies result.json + predictions to a
   permanent backup directory that should NEVER be deleted

The backup directory structure:
    outputs/results_backup_DO_NOT_DELETE/
        {model_name}/
            result.json          (metrics: Pearson R, MSE, etc.)
            test_predictions.npz (pred/true arrays for scatter plots)

Usage as a module:
    from scripts.save_results_backup import save_test_predictions, backup_results

Usage as a standalone script (backs up all existing results):
    python scripts/save_results_backup.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[1]
BACKUP_DIR = REPO / "outputs" / "results_backup_DO_NOT_DELETE"
TEST_DIR = REPO / "data" / "k562" / "test_sets"


def _corr(pred: np.ndarray, true: np.ndarray, fn) -> float:
    mask = np.isfinite(pred) & np.isfinite(true)
    return float(fn(pred[mask], true[mask])[0]) if mask.sum() >= 3 else 0.0


def save_test_predictions(
    predict_fn,
    output_dir: Path,
    test_dir: Path | None = None,
) -> dict:
    """Run predictions on all K562 test sets and save arrays + metrics.

    Args:
        predict_fn: Callable that takes list[str] -> np.ndarray of predictions
        output_dir: Where to save test_predictions.npz and result metrics
        test_dir: Path to K562 test_sets directory (default: data/k562/test_sets)

    Returns:
        Dict of test metrics with Pearson R, Spearman R, MSE, and n per test set.
    """
    if test_dir is None:
        test_dir = TEST_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # In-distribution
    in_df = pd.read_csv(test_dir / "test_in_distribution_hashfrag.tsv", sep="\t")
    in_pred = predict_fn(in_df["sequence"].tolist())
    in_true = in_df["K562_log2FC"].to_numpy(dtype=np.float32)

    # SNV pairs
    snv_df = pd.read_csv(test_dir / "test_snv_pairs_hashfrag.tsv", sep="\t")
    ref_pred = predict_fn(snv_df["sequence_ref"].tolist())
    alt_pred = predict_fn(snv_df["sequence_alt"].tolist())
    alt_true = snv_df["K562_log2FC_alt"].to_numpy(dtype=np.float32)
    delta_pred = alt_pred - ref_pred
    delta_true = snv_df["delta_log2FC"].to_numpy(dtype=np.float32)

    # OOD
    ood_df = pd.read_csv(test_dir / "test_ood_designed_k562.tsv", sep="\t")
    ood_pred = predict_fn(ood_df["sequence"].tolist())
    ood_true = ood_df["K562_log2FC"].to_numpy(dtype=np.float32)

    # Save predictions
    pred_path = output_dir / "test_predictions.npz"
    np.savez_compressed(
        pred_path,
        in_dist_pred=in_pred,
        in_dist_true=in_true,
        snv_alt_pred=alt_pred,
        snv_alt_true=alt_true,
        snv_ref_pred=ref_pred,
        snv_delta_pred=delta_pred,
        snv_delta_true=delta_true,
        ood_pred=ood_pred,
        ood_true=ood_true,
    )

    # Compute metrics
    metrics = {
        "in_distribution": {
            "pearson_r": _corr(in_pred, in_true, pearsonr),
            "spearman_r": _corr(in_pred, in_true, spearmanr),
            "mse": float(np.mean((in_pred - in_true) ** 2)),
            "n": len(in_true),
        },
        "snv_abs": {
            "pearson_r": _corr(alt_pred, alt_true, pearsonr),
            "spearman_r": _corr(alt_pred, alt_true, spearmanr),
            "mse": float(np.mean((alt_pred - alt_true) ** 2)),
            "n": len(alt_true),
        },
        "snv_delta": {
            "pearson_r": _corr(delta_pred, delta_true, pearsonr),
            "spearman_r": _corr(delta_pred, delta_true, spearmanr),
            "mse": float(np.mean((delta_pred - delta_true) ** 2)),
            "n": len(delta_true),
        },
        "ood": {
            "pearson_r": _corr(ood_pred, ood_true, pearsonr),
            "spearman_r": _corr(ood_pred, ood_true, spearmanr),
            "mse": float(np.mean((ood_pred - ood_true) ** 2)),
            "n": len(ood_true),
        },
    }

    print(f"  Predictions saved: {pred_path} ({pred_path.stat().st_size / 1024:.0f} KB)")
    for k, v in metrics.items():
        print(f"  {k}: R={v['pearson_r']:.4f}, MSE={v['mse']:.4f}")

    return metrics


def backup_results(
    model_name: str,
    source_dir: Path,
    extra_metadata: dict | None = None,
) -> None:
    """Copy lightweight results to the permanent backup directory.

    Copies result.json and test_predictions.npz (NOT model weights).
    """
    backup = BACKUP_DIR / model_name
    backup.mkdir(parents=True, exist_ok=True)

    # Copy result.json
    for fname in ["result.json", "test_metrics.json"]:
        src = source_dir / fname
        if src.exists():
            shutil.copy2(src, backup / fname)
            print(f"  Backed up: {fname}")

    # Copy predictions
    pred_src = source_dir / "test_predictions.npz"
    if pred_src.exists():
        shutil.copy2(pred_src, backup / "test_predictions.npz")
        print(f"  Backed up: test_predictions.npz")

    # Save extra metadata
    if extra_metadata:
        meta_path = backup / "metadata.json"
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        meta.update(extra_metadata)
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

    print(f"  Backup: {backup}")


def backup_all_existing():
    """Scan for all existing results and back them up."""
    print("=== Backing up all existing results ===\n")

    # S1 models
    s1_models = [
        ("DREAM-RNN_S1", "dream_rnn_k562_with_preds/seed_42/seed_42/fraction_1.0000"),
        ("Malinois_S1", "malinois_k562_with_preds/seed_42/seed_42"),
        ("Malinois_S1_sweep", "malinois_k562_sweep/lr0.001_wd1e-3/seed_0/seed_1455421226"),
        ("NTv3_S1", "foundation_grid_search/ntv3_post/lr0.0005_wd1e-6_do0.1/seed_42/seed_42"),
        ("Borzoi_S1", "borzoi_k562_3seeds/seed_824292012"),
        ("Enformer_S1", "enformer_k562_3seeds/seed_598125057"),
        ("AG_fold1_S1", "ag_fold_1_k562_s1_full"),
        ("AG_allfolds_S1", "ag_all_folds_k562_s1_full"),
    ]

    # S2 models
    s2_models = [
        ("Enformer_S2", "enformer_k562_stage2_final/elr1e-4_all/run_0"),
        ("AG_fold1_S2", "stage2_k562_fold1/run_0"),
        ("NTv3_S2", "ntv3_k562_stage2_final/run_0"),
    ]

    for name, rel_path in s1_models + s2_models:
        src = REPO / "outputs" / rel_path
        if src.exists() and any(src.glob("result*.json")):
            print(f"{name}:")
            backup_results(name, src, {"source_dir": str(rel_path)})
        else:
            print(f"{name}: source not found ({rel_path})")

    # Also save the reference metrics
    ref_dir = REPO / "outputs" / "reference_metrics"
    if ref_dir.exists():
        for f in ref_dir.glob("*.json"):
            dst = BACKUP_DIR / f.name
            shutil.copy2(f, dst)
            print(f"Backed up reference: {f.name}")

    print(f"\n=== Backup complete: {BACKUP_DIR} ===")


if __name__ == "__main__":
    backup_all_existing()
