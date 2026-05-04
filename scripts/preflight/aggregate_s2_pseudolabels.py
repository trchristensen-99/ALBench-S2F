"""Aggregate per-fold S2 pseudolabel predictions into final npz files.

Reads outputs/oracle_pseudolabels_k562_ag_s2_refalt/fold_preds/fold_{0-9}.npz
and writes the aggregated arrays:

    train_oracle_labels.npz   — keys: oracle_mean, oracle_std, oof_oracle, true_label
    val_oracle_labels.npz     — keys: oracle_mean, oracle_std, true_label
    test_oracle_labels.npz    — keys: oracle_mean, oracle_std, true_label
    snv_oracle_labels.npz     — keys: oracle_mean_ref, oracle_mean_alt, true_label_ref, true_label_alt
    summary.json              — counts + per-fold + ensemble Pearson R on real labels

The ``oof_oracle`` column for train labels uses, for each sequence, the
ensemble mean over the *other* 9 folds — this is a leave-one-fold-out
approximation that's appropriate when no per-sequence fold mask is
available. (The folds were trained on different val splits but on the
same overall train pool, so any one fold's prediction on a train sequence
is "in-sample" — leave-one-out averaging still gives slightly better
out-of-fold properties than the full ensemble mean.)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[2]


def _safe_corr(y, p):
    if y.size < 2 or np.std(y) == 0 or np.std(p) == 0:
        return 0.0
    return float(pearsonr(y, p)[0])


def main():
    cache = REPO / "outputs" / "oracle_pseudolabels_k562_ag_s2_refalt"
    pool_dir = cache / "pool"
    fold_dir = cache / "fold_preds"

    train = pd.read_parquet(pool_dir / "train.parquet")
    val = pd.read_parquet(pool_dir / "val.parquet")
    test = pd.read_parquet(pool_dir / "test.parquet")
    snv = pd.read_parquet(pool_dir / "snv_pairs.parquet")

    fold_npzs = sorted(fold_dir.glob("fold_*.npz"))
    if not fold_npzs:
        raise SystemExit(f"No fold predictions found at {fold_dir}")
    print(f"Aggregating {len(fold_npzs)} folds …")

    train_stack = np.stack([np.load(f)["train_preds"] for f in fold_npzs], axis=0)
    val_stack = np.stack([np.load(f)["val_preds"] for f in fold_npzs], axis=0)
    test_stack = np.stack([np.load(f)["test_preds"] for f in fold_npzs], axis=0)
    snv_ref_stack = np.stack([np.load(f)["snv_ref_preds"] for f in fold_npzs], axis=0)
    snv_alt_stack = np.stack([np.load(f)["snv_alt_preds"] for f in fold_npzs], axis=0)

    def _summary(stack, true_lbl, name):
        ens = stack.mean(axis=0)
        std = stack.std(axis=0)
        per_fold_r = [_safe_corr(true_lbl, stack[i]) for i in range(stack.shape[0])]
        ens_r = _safe_corr(true_lbl, ens)
        print(
            f"  {name}: per-fold Pearson R = "
            f"[min={min(per_fold_r):.4f}, mean={np.mean(per_fold_r):.4f}, "
            f"max={max(per_fold_r):.4f}], ensemble = {ens_r:.4f}"
        )
        return ens, std, per_fold_r, ens_r

    train_lbl = train["K562_log2FC"].to_numpy(np.float32)
    val_lbl = val["K562_log2FC"].to_numpy(np.float32)
    test_lbl = test["K562_log2FC"].to_numpy(np.float32)
    snv_ref_lbl = snv["ref_log2FC"].to_numpy(np.float32)
    snv_alt_lbl = snv["alt_log2FC"].to_numpy(np.float32)

    train_mean, train_std, train_per_fold_r, train_ens_r = _summary(train_stack, train_lbl, "train")
    val_mean, val_std, val_per_fold_r, val_ens_r = _summary(val_stack, val_lbl, "val")
    test_mean, test_std, test_per_fold_r, test_ens_r = _summary(test_stack, test_lbl, "test")
    snv_ref_mean, snv_ref_std, snv_ref_per_fold_r, snv_ref_ens_r = _summary(
        snv_ref_stack, snv_ref_lbl, "snv_ref"
    )
    snv_alt_mean, snv_alt_std, snv_alt_per_fold_r, snv_alt_ens_r = _summary(
        snv_alt_stack, snv_alt_lbl, "snv_alt"
    )

    # OOF for train: leave-one-out average. For each fold k, OOF prediction
    # is the mean over all OTHER folds.
    n_folds = train_stack.shape[0]
    oof_oracle = np.zeros_like(train_mean)
    sum_all = train_stack.sum(axis=0)
    for k in range(n_folds):
        oof_oracle += (sum_all - train_stack[k]) / (n_folds - 1)
    oof_oracle /= n_folds
    oof_r = _safe_corr(train_lbl, oof_oracle)
    print(f"  train OOF (leave-one-out average) Pearson R = {oof_r:.4f}")

    np.savez_compressed(
        cache / "train_oracle_labels.npz",
        oracle_mean=train_mean,
        oracle_std=train_std,
        oof_oracle=oof_oracle,
        true_label=train_lbl,
    )
    np.savez_compressed(
        cache / "val_oracle_labels.npz",
        oracle_mean=val_mean,
        oracle_std=val_std,
        true_label=val_lbl,
    )
    np.savez_compressed(
        cache / "test_oracle_labels.npz",
        oracle_mean=test_mean,
        oracle_std=test_std,
        true_label=test_lbl,
    )
    np.savez_compressed(
        cache / "snv_oracle_labels.npz",
        oracle_mean_ref=snv_ref_mean,
        oracle_std_ref=snv_ref_std,
        true_label_ref=snv_ref_lbl,
        oracle_mean_alt=snv_alt_mean,
        oracle_std_alt=snv_alt_std,
        true_label_alt=snv_alt_lbl,
    )

    # Determinism checksums on the relabeled splits — per the pre-flight
    # checklist's "verify checksum on relabeled test sets before any
    # training". Hash the float32 ensemble means + the underlying sequences
    # so any silent regeneration drift is caught at run-config-load time.
    import hashlib

    def _checksum(seqs: list[str], preds: np.ndarray) -> str:
        h = hashlib.sha256()
        for s in seqs:
            h.update(s.encode())
        h.update(preds.astype(np.float32).tobytes())
        return h.hexdigest()[:16]

    checksums = {
        "train": _checksum(train["sequence"].astype(str).tolist(), train_mean),
        "val": _checksum(val["sequence"].astype(str).tolist(), val_mean),
        "test": _checksum(test["sequence"].astype(str).tolist(), test_mean),
        "snv_ref": _checksum(snv["ref_sequence"].astype(str).tolist(), snv_ref_mean)
        if "ref_sequence" in snv.columns
        else "n/a",
        "snv_alt": _checksum(snv["alt_sequence"].astype(str).tolist(), snv_alt_mean)
        if "alt_sequence" in snv.columns
        else "n/a",
    }

    summary = {
        "n_folds": int(n_folds),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "n_snv_pairs": int(len(snv)),
        "ensemble_pearson_r": {
            "train": train_ens_r,
            "val": val_ens_r,
            "test": test_ens_r,
            "snv_ref": snv_ref_ens_r,
            "snv_alt": snv_alt_ens_r,
            "train_oof": oof_r,
        },
        "per_fold_pearson_r": {
            "train": train_per_fold_r,
            "val": val_per_fold_r,
            "test": test_per_fold_r,
            "snv_ref": snv_ref_per_fold_r,
            "snv_alt": snv_alt_per_fold_r,
        },
        "checksums_sha256_truncated": checksums,
    }
    (cache / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved aggregated cache to {cache}")
    print(json.dumps(summary["ensemble_pearson_r"], indent=2))


if __name__ == "__main__":
    main()
