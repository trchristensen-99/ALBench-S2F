"""Compute PROPER 10-fold ensemble metrics (averaging predictions, not per-fold metrics).

For each oracle, loads test_predictions.npz from all 10 folds, averages predictions
across folds, then computes Pearson R on the ensemble predictions.

This is the CORRECT measurement of a 10-fold ensemble's accuracy.
Per-fold mean significantly underestimates ensemble performance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")


def compute_ensemble(oracle_dir: Path) -> dict:
    """Load 10 folds' test predictions, average them, compute metrics on ensemble."""
    keys = {
        "in_dist": ("in_dist_pred", "in_dist_true"),
        "snv_delta": ("snv_delta_pred", "snv_delta_true"),
        "snv_alt": ("snv_alt_pred", "snv_alt_true"),
        "ood": ("ood_pred", "ood_true"),
    }
    # Collect per-fold predictions
    fold_preds = {k: [] for k in keys}
    truth = {k: None for k in keys}
    n_folds_found = 0
    for fold in range(10):
        npz_path = oracle_dir / f"fold_{fold}" / "test_predictions.npz"
        if not npz_path.exists():
            continue
        d = np.load(npz_path, allow_pickle=True)
        for k, (pkey, tkey) in keys.items():
            if pkey in d.files and tkey in d.files:
                fold_preds[k].append(d[pkey])
                if truth[k] is None:
                    truth[k] = d[tkey]
        n_folds_found += 1

    if n_folds_found == 0:
        return None

    out = {"n_folds": n_folds_found}
    for k, (_, _) in keys.items():
        if not fold_preds[k] or truth[k] is None:
            continue
        per_fold_pred = np.stack(fold_preds[k])
        per_fold_r = [pearsonr(p, truth[k])[0] for p in per_fold_pred]
        ensemble_pred = np.mean(per_fold_pred, axis=0)
        ens_r = pearsonr(ensemble_pred, truth[k])[0]
        ens_rho = spearmanr(ensemble_pred, truth[k])[0]
        ens_mse = np.mean((ensemble_pred - truth[k]) ** 2)
        out[f"{k}_per_fold_R_mean"] = float(np.mean(per_fold_r))
        out[f"{k}_per_fold_R_std"] = float(np.std(per_fold_r))
        out[f"{k}_ensemble_R"] = float(ens_r)
        out[f"{k}_ensemble_rho"] = float(ens_rho)
        out[f"{k}_ensemble_mse"] = float(ens_mse)
        out[f"{k}_truth_mean"] = float(truth[k].mean())
        out[f"{k}_truth_std"] = float(truth[k].std())
        out[f"{k}_pred_mean"] = float(ensemble_pred.mean())
        out[f"{k}_pred_std"] = float(ensemble_pred.std())
    return out


def main():
    oracles = {
        "baseline": REPO / "outputs/stage2_k562_oracle",
        "c28_10fold": REPO / "outputs/oracle_neg_sweep/debias_oracle_c28_10fold",
        "c63_10fold": REPO / "outputs/oracle_neg_sweep/debias_c63_10fold",
        "c86_10fold": REPO / "outputs/oracle_neg_sweep/debias_c86_10fold",
        "c91_10fold": REPO / "outputs/oracle_neg_sweep/debias_c91_10fold",
    }
    results = {}
    for name, path in oracles.items():
        print(f"\n=== {name} ({path.name}) ===")
        r = compute_ensemble(path)
        if r is None:
            print("  no folds found")
            continue
        results[name] = r
        for k in ["in_dist", "snv_delta", "ood"]:
            if f"{k}_ensemble_R" in r:
                print(
                    f"  {k:>11}: per-fold={r[f'{k}_per_fold_R_mean']:.4f}±"
                    f"{r[f'{k}_per_fold_R_std']:.4f}  ENSEMBLE={r[f'{k}_ensemble_R']:.4f}  "
                    f"MSE={r[f'{k}_ensemble_mse']:.4f}"
                )
        # Prediction distribution on each test set
        print(f"  (pred means: in_dist={r.get('in_dist_pred_mean',0):.3f}, "
              f"snv_alt={r.get('snv_alt_pred_mean',0):.3f}, "
              f"ood={r.get('ood_pred_mean',0):.3f})")
        print(f"  (true means: in_dist={r.get('in_dist_truth_mean',0):.3f}, "
              f"snv_alt={r.get('snv_alt_truth_mean',0):.3f}, "
              f"ood={r.get('ood_truth_mean',0):.3f})")

    # Save
    out_path = REPO / "results/preflight/oracle_ensemble_metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
