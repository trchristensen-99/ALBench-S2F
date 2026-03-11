#!/usr/bin/env python
"""Aggregate per-fold Stage 2 pseudolabel predictions into final NPZ files.

Reads fold_preds/fold_{0..9}.npz (produced by generate_stage2_pseudolabels_single_fold.py)
and produces the same output as generate_oracle_pseudolabels_stage2_k562_ag.py:

    output_dir/
        train_oracle_labels.npz
        val_oracle_labels.npz
        test_in_dist_oracle_labels.npz
        test_snv_oracle_labels.npz
        test_ood_oracle_labels.npz
        summary.json

Usage:
    python scripts/analysis/aggregate_stage2_oracle_pseudolabels.py \
        --preds-dir outputs/oracle_pseudolabels_stage2_k562_ag/fold_preds \
        --output-dir outputs/oracle_pseudolabels_stage2_k562_ag \
        --k562-data-path data/k562
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from data.k562 import K562Dataset


def _safe_corr(y_true, y_pred, fn):
    if y_true.size < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0
    return float(fn(y_true, y_pred)[0])


def _oracle_fold_val_indices(n_total: int, n_folds: int = 10) -> dict[int, np.ndarray]:
    perm = np.random.default_rng(seed=42).permutation(n_total)
    fold_size = n_total // n_folds
    fold_val_idx: dict[int, np.ndarray] = {}
    for fold_id in range(n_folds):
        val_start = fold_id * fold_size
        val_end = val_start + fold_size if fold_id < n_folds - 1 else n_total
        fold_val_idx[fold_id] = perm[val_start:val_end]
    return fold_val_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-dir", required=True, help="Directory with fold_N.npz files")
    parser.add_argument("--output-dir", required=True, help="Output directory for final NPZs")
    parser.add_argument("--k562-data-path", default="data/k562")
    parser.add_argument("--n-folds", type=int, default=10)
    args = parser.parse_args()

    preds_dir = Path(args.preds_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load per-fold predictions ─────────────────────────────────────────────
    fold_data = {}
    for fold_id in range(args.n_folds):
        p = preds_dir / f"fold_{fold_id}.npz"
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
        fold_data[fold_id] = dict(np.load(p))
        print(f"  Loaded fold {fold_id}: {p}")

    n_models = len(fold_data)
    print(f"All {n_models} folds loaded.", flush=True)

    # ── Load true labels ──────────────────────────────────────────────────────
    ds_train = K562Dataset(data_path=args.k562_data_path, split="train")
    ds_val = K562Dataset(data_path=args.k562_data_path, split="val")
    train_labels = ds_train.labels.astype(np.float32)
    val_labels = ds_val.labels.astype(np.float32)
    n_train = len(train_labels)

    test_dir = Path(args.k562_data_path) / "test_sets"
    in_dist_df = pd.read_csv(test_dir / "test_in_distribution_hashfrag.tsv", sep="\t")
    in_dist_labels = in_dist_df["K562_log2FC"].to_numpy(dtype=np.float32)
    snv_df = pd.read_csv(test_dir / "test_snv_pairs_hashfrag.tsv", sep="\t")
    snv_alt_labels = snv_df["K562_log2FC_alt"].to_numpy(dtype=np.float32)
    snv_delta_labels = snv_df["delta_log2FC"].to_numpy(dtype=np.float32)
    ood_df = pd.read_csv(test_dir / "test_ood_designed_k562.tsv", sep="\t")
    ood_labels = ood_df["K562_log2FC"].to_numpy(dtype=np.float32)

    # ── OOF fold indices ──────────────────────────────────────────────────────
    fold_val_idx = _oracle_fold_val_indices(n_train, args.n_folds)
    train_oof = np.full(n_train, np.nan, dtype=np.float32)

    # ── Accumulate ensemble statistics ────────────────────────────────────────
    keys = [
        "train_preds",
        "val_preds",
        "in_dist_preds",
        "snv_ref_preds",
        "snv_alt_preds",
        "ood_preds",
    ]
    sums = {}
    sumsqs = {}
    for k in keys:
        n = len(fold_data[0][k])
        sums[k] = np.zeros(n, dtype=np.float64)
        sumsqs[k] = np.zeros(n, dtype=np.float64)

    for fold_id in range(n_models):
        d = fold_data[fold_id]
        for k in keys:
            preds = d[k].astype(np.float64)
            sums[k] += preds
            sumsqs[k] += preds**2

        # OOF
        if fold_id in fold_val_idx:
            oof_idx = fold_val_idx[fold_id]
            train_oof[oof_idx] = d["train_preds"][oof_idx]

    def _finalize(k):
        mean = (sums[k] / n_models).astype(np.float32)
        var = np.maximum((sumsqs[k] / n_models) - mean.astype(np.float64) ** 2, 0.0)
        std = np.sqrt(var).astype(np.float32)
        return mean, std

    train_mean, train_std = _finalize("train_preds")
    val_mean, val_std = _finalize("val_preds")
    in_dist_mean, in_dist_std = _finalize("in_dist_preds")
    snv_ref_mean, _ = _finalize("snv_ref_preds")
    snv_alt_mean, snv_alt_std = _finalize("snv_alt_preds")
    snv_delta_mean = snv_alt_mean - snv_ref_mean
    ood_mean, ood_std = _finalize("ood_preds")

    # ── Save NPZ files ───────────────────────────────────────────────────────
    np.savez_compressed(
        output_dir / "train_oracle_labels.npz",
        oracle_mean=train_mean,
        oracle_std=train_std,
        oof_oracle=train_oof,
        true_label=train_labels,
    )
    np.savez_compressed(
        output_dir / "val_oracle_labels.npz",
        oracle_mean=val_mean,
        oracle_std=val_std,
        true_label=val_labels,
    )
    np.savez_compressed(
        output_dir / "test_in_dist_oracle_labels.npz",
        oracle_mean=in_dist_mean,
        oracle_std=in_dist_std,
        true_label=in_dist_labels,
    )
    np.savez_compressed(
        output_dir / "test_snv_oracle_labels.npz",
        ref_oracle_mean=snv_ref_mean,
        alt_oracle_mean=snv_alt_mean,
        delta_oracle_mean=snv_delta_mean,
        alt_oracle_std=snv_alt_std,
        true_alt_label=snv_alt_labels,
        true_delta=snv_delta_labels,
    )
    np.savez_compressed(
        output_dir / "test_ood_oracle_labels.npz",
        oracle_mean=ood_mean,
        oracle_std=ood_std,
        true_label=ood_labels,
    )
    print(f"Wrote NPZ files to {output_dir}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    oof_mask = np.isfinite(train_oof)
    summary = {
        "n_oracle_models": n_models,
        "oracle_folds": list(range(n_models)),
        "stage": 2,
        "train_pool": {
            "n": int(n_train),
            "ensemble_pearson_r": _safe_corr(train_labels, train_mean, pearsonr),
            "ensemble_spearman_r": _safe_corr(train_labels, train_mean, spearmanr),
            "oof_covered": int(np.sum(oof_mask)),
            "oof_pearson_r": _safe_corr(train_labels[oof_mask], train_oof[oof_mask], pearsonr),
            "oof_spearman_r": _safe_corr(train_labels[oof_mask], train_oof[oof_mask], spearmanr),
        },
        "val": {
            "n": int(len(val_labels)),
            "ensemble_pearson_r": _safe_corr(val_labels, val_mean, pearsonr),
            "ensemble_spearman_r": _safe_corr(val_labels, val_mean, spearmanr),
        },
        "test_in_distribution": {
            "n": int(len(in_dist_labels)),
            "ensemble_pearson_r": _safe_corr(in_dist_labels, in_dist_mean, pearsonr),
            "ensemble_spearman_r": _safe_corr(in_dist_labels, in_dist_mean, spearmanr),
        },
        "test_snv_alt": {
            "n": int(len(snv_alt_labels)),
            "ensemble_pearson_r": _safe_corr(snv_alt_labels, snv_alt_mean, pearsonr),
            "ensemble_spearman_r": _safe_corr(snv_alt_labels, snv_alt_mean, spearmanr),
        },
        "test_snv_delta": {
            "n": int(len(snv_delta_labels)),
            "ensemble_pearson_r": _safe_corr(snv_delta_labels, snv_delta_mean, pearsonr),
            "ensemble_spearman_r": _safe_corr(snv_delta_labels, snv_delta_mean, spearmanr),
        },
        "test_ood": {
            "n": int(len(ood_labels)),
            "ensemble_pearson_r": _safe_corr(ood_labels, ood_mean, pearsonr),
            "ensemble_spearman_r": _safe_corr(ood_labels, ood_mean, spearmanr),
        },
    }

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote summary to {output_dir / 'summary.json'}", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
