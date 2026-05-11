"""Analyze prediction bias by activity bin across all test sets.

For each (oracle, test_set), bin predictions by ground-truth activity quintile
and compute mean residual (predicted - true). Reveals systematic over/under-prediction
patterns — e.g., are high-activity designed sequences under-predicted?
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")


def load_ensemble_preds(oracle_dir: Path, key_pred: str, key_true: str):
    fold_preds = []
    truth = None
    for fold in range(10):
        npz = oracle_dir / f"fold_{fold}" / "test_predictions.npz"
        if not npz.exists():
            continue
        d = np.load(npz, allow_pickle=True)
        if key_pred in d.files and key_true in d.files:
            fold_preds.append(d[key_pred])
            if truth is None:
                truth = d[key_true]
    if not fold_preds:
        return None, None
    return np.mean(np.stack(fold_preds), axis=0), truth


def analyze_bias_by_bin(pred, true, n_bins=5):
    """Bin true labels into quantiles and compute mean residual per bin."""
    quantiles = np.quantile(true, np.linspace(0, 1, n_bins + 1))
    rows = []
    for i in range(n_bins):
        lo, hi = quantiles[i], quantiles[i + 1]
        mask = (true >= lo) & (true <= hi if i == n_bins - 1 else true < hi)
        if mask.sum() < 5:
            continue
        residual = pred[mask] - true[mask]
        rows.append({
            "bin_lo": lo,
            "bin_hi": hi,
            "bin_center": (lo + hi) / 2,
            "n": int(mask.sum()),
            "true_mean": float(true[mask].mean()),
            "pred_mean": float(pred[mask].mean()),
            "residual_mean": float(residual.mean()),
            "residual_std": float(residual.std()),
        })
    return rows


oracles = {
    "baseline": "outputs/stage2_k562_oracle",
    "c28": "outputs/oracle_neg_sweep/debias_oracle_c28_10fold",
    "c63": "outputs/oracle_neg_sweep/debias_c63_10fold",
    "c86": "outputs/oracle_neg_sweep/debias_c86_10fold",
    "c91": "outputs/oracle_neg_sweep/debias_c91_10fold",
}

test_sets = {
    "in_dist (chr 7+13)": ("in_dist_pred", "in_dist_true"),
    "OOD (multi-celltype)": ("ood_pred", "ood_true"),
    "SNV-alt (designed variants)": ("snv_alt_pred", "snv_alt_true"),
}

results = {}
for oname, opath in oracles.items():
    results[oname] = {}
    for tname, (kp, kt) in test_sets.items():
        pred, true = load_ensemble_preds(REPO / opath, kp, kt)
        if pred is None:
            continue
        rows = analyze_bias_by_bin(pred, true, n_bins=5)
        results[oname][tname] = {
            "global_mean_true": float(true.mean()),
            "global_mean_pred": float(pred.mean()),
            "global_residual": float((pred - true).mean()),
            "bins": rows,
        }

# Save
(REPO / "results/preflight/prediction_bias_by_bin.json").write_text(json.dumps(results, indent=2))
print(f"Saved {REPO}/results/preflight/prediction_bias_by_bin.json")

# Plot
fig, axes = plt.subplots(len(test_sets), len(oracles), figsize=(4 * len(oracles), 3 * len(test_sets)),
                          squeeze=False, sharex="col")
oracle_list = list(oracles.keys())
test_list = list(test_sets.keys())
for ti, tname in enumerate(test_list):
    for oi, oname in enumerate(oracle_list):
        ax = axes[ti][oi]
        if tname not in results[oname]:
            ax.set_visible(False)
            continue
        bins = results[oname][tname]["bins"]
        if not bins:
            continue
        centers = [b["bin_center"] for b in bins]
        residuals = [b["residual_mean"] for b in bins]
        stds = [b["residual_std"] for b in bins]
        ns = [b["n"] for b in bins]
        ax.errorbar(centers, residuals, yerr=stds, marker="o", capsize=3, color="steelblue")
        ax.axhline(0, color="red", linestyle="--", alpha=0.5)
        for c, r, n in zip(centers, residuals, ns):
            ax.text(c, r + max(stds) * 0.15, f"n={n}", ha="center", fontsize=6, alpha=0.7)
        if ti == 0:
            ax.set_title(oname, fontsize=10)
        if oi == 0:
            ax.set_ylabel(f"{tname}\nresidual\n(pred - true)", fontsize=8)
        if ti == len(test_list) - 1:
            ax.set_xlabel("true label (binned)", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
fig.suptitle("Prediction bias by activity bin: under = over-predicting low / under-predicting high",
             fontsize=12)
fig.tight_layout()
out = REPO / "results/preflight/figures/meeting/10_prediction_bias_by_bin.png"
fig.savefig(out, dpi=130)
plt.close(fig)
print(f"Saved {out}")

# Print summary table
print("\n=== Global residual (pred - true), averaged across test set ===")
print(f"{'oracle':<10} {'in_dist':>10} {'OOD':>10} {'SNV-alt':>10}")
for o in oracle_list:
    row = []
    for t in test_list:
        if t in results[o]:
            row.append(f"{results[o][t]['global_residual']:+.3f}")
        else:
            row.append("  --  ")
    print(f"{o:<10} {row[0]:>10} {row[1]:>10} {row[2]:>10}")
