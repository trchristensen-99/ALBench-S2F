#!/usr/bin/env python3
"""Generate scatter/hexbin grids for the combined best (S1+S2) comparison.

Uses S2 predictions for Enformer, NTv3, AG fold 1, AG all folds.
Uses S1 predictions for DREAM-RNN, Malinois, Borzoi (no S2 for these).

Run:
    python scripts/analysis/plot_combined_scatterplots.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "exp0_plots"

# Best available: S2 for foundation models with finetuning, S1 for from-scratch
MODELS = [
    ("AG all folds", "stage2_k562_full_train/run_1", "#1B5E20"),  # S2
    ("AG fold 1", "stage2_k562_fold1/run_3", "#66BB6A"),  # S2
    (
        "DREAM-RNN",
        "dream_rnn_k562_with_preds/seed_42/seed_42/fraction_1.0000",
        "#7B2D8E",
    ),  # from-scratch
    ("Enformer", "enformer_k562_stage2_final_v2/run_0", "#3A86C8"),  # S2
    ("Malinois", "malinois_k562_with_preds/seed_42/seed_42", "#B07CC6"),  # from-scratch
    ("Borzoi", "borzoi_k562_3seeds/seed_824292012", "#DAA520"),  # S1 (S2 failed)
    (
        "NTv3",
        "ntv3_post_k562_stage2/sweep_elr1e-4_uf4",
        "#E8602C",
    ),  # S2
]

# Also check backup directory as fallback
BACKUP_DIR = REPO / "outputs" / "results_backup_DO_NOT_DELETE"
BACKUP_MODELS = {
    "AG all folds": "AG_allfolds_S2",
    "AG fold 1": "AG_fold1_S2",
    "DREAM-RNN": "DREAM-RNN_S1",
    "Enformer": "Enformer_S1",  # fallback to S1 if S2 not available
    "Malinois": "Malinois_S1",
    "Borzoi": "Borzoi_S1",
    "NTv3": "NTv3_S1",  # fallback to S1 if S2 not available
}

TEST_SETS = [
    ("in_dist", "Reference\n(in-distribution)"),
    ("snv_abs", "SNV\n(absolute)"),
    ("snv_delta", "SNV effect\n(delta)"),
    ("ood", "Synthetic design\n(OOD)"),
]


def load_predictions(model_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    pred_path = model_dir / "test_predictions.npz"
    if not pred_path.exists():
        return None
    data = np.load(pred_path)
    result = {}
    if "in_dist_pred" in data:
        result["in_dist"] = (data["in_dist_pred"], data["in_dist_true"])
    if "snv_alt_pred" in data:
        result["snv_abs"] = (data["snv_alt_pred"], data["snv_alt_true"])
    if "snv_delta_pred" in data:
        result["snv_delta"] = (data["snv_delta_pred"], data["snv_delta_true"])
    if "ood_pred" in data:
        result["ood"] = (data["ood_pred"], data["ood_true"])
    return result


def _pearson(pred, true):
    mask = np.isfinite(pred) & np.isfinite(true)
    return float(pearsonr(pred[mask], true[mask])[0]) if mask.sum() >= 3 else 0.0


def _mse(pred, true):
    mask = np.isfinite(pred) & np.isfinite(true)
    return float(np.mean((pred[mask] - true[mask]) ** 2)) if mask.sum() >= 1 else 0.0


def _make_grid(all_data, plot_fn, title, filename):
    n_models = len(MODELS)
    n_tests = len(TEST_SETS)
    fig, axes = plt.subplots(
        n_models, n_tests, figsize=(4 * n_tests, 3.5 * n_models), squeeze=False
    )

    for row, (model_name, _, color) in enumerate(MODELS):
        data = all_data.get(model_name)
        for col, (test_key, test_label) in enumerate(TEST_SETS):
            ax = axes[row, col]
            if data is None or test_key not in data:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                ax.set_xlim(-2, 5)
                ax.set_ylim(-2, 5)
            else:
                pred, true = data[test_key]
                r = _pearson(pred, true)
                mse = _mse(pred, true)
                plot_fn(ax, pred, true, color, r, mse)
            if row == 0:
                ax.set_title(test_label, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(model_name, fontsize=11, fontweight="bold")
            if row == n_models - 1:
                ax.set_xlabel("True expression", fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = OUT_DIR / filename
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}.png / .pdf")


def _scatter_panel(ax, pred, true, color, r, mse):
    n = len(pred)
    if n > 5000:
        idx = np.random.default_rng(42).choice(n, size=5000, replace=False)
        pred_plot, true_plot = pred[idx], true[idx]
    else:
        pred_plot, true_plot = pred, true
    ax.scatter(true_plot, pred_plot, s=1, alpha=0.15, color=color, rasterized=True)
    lims = [
        min(np.nanmin(true_plot), np.nanmin(pred_plot)),
        max(np.nanmax(true_plot), np.nanmax(pred_plot)),
    ]
    ax.plot(lims, lims, "k--", lw=0.5, alpha=0.5)
    ax.text(
        0.05,
        0.95,
        f"R = {r:.3f}\nMSE = {mse:.3f}",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


def _hexbin_panel(ax, pred, true, color, r, mse):
    mask = np.isfinite(pred) & np.isfinite(true)
    ax.hexbin(true[mask], pred[mask], gridsize=50, cmap="viridis", mincnt=1, linewidths=0.1)
    lims = [
        min(np.nanmin(true[mask]), np.nanmin(pred[mask])),
        max(np.nanmax(true[mask]), np.nanmax(pred[mask])),
    ]
    ax.plot(lims, lims, "r--", lw=1, alpha=0.7)
    ax.text(
        0.05,
        0.95,
        f"R = {r:.3f}\nMSE = {mse:.3f}",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading predictions (best of S1+S2)...")

    all_data: dict[str, dict] = {}
    for model_name, model_dir_name, color in MODELS:
        model_dir = REPO / "outputs" / model_dir_name
        data = load_predictions(model_dir)
        if data is None and model_name in BACKUP_MODELS:
            backup_path = BACKUP_DIR / BACKUP_MODELS[model_name]
            data = load_predictions(backup_path)
            if data:
                print(f"  {model_name}: loaded from backup ({BACKUP_MODELS[model_name]})")
        if data:
            all_data[model_name] = data
            n_points = sum(len(v[0]) for v in data.values())
            print(f"  {model_name}: {len(data)} test sets, {n_points:,} points")
        else:
            print(f"  {model_name}: NO PREDICTIONS")

    print(f"\nModels with data: {len(all_data)}/{len(MODELS)}")

    print("\nGenerating combined scatter grid...")
    _make_grid(
        all_data,
        _scatter_panel,
        "K562 MPRA — Best Available (S1+S2) Predicted vs True",
        "k562_combined_scatter_grid",
    )

    print("Generating combined hexbin grid...")
    _make_grid(
        all_data,
        _hexbin_panel,
        "K562 MPRA — Best Available (S1+S2) Predicted vs True (Density)",
        "k562_combined_hexbin_grid",
    )

    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
