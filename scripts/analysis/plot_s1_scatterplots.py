#!/usr/bin/env python3
"""Generate scatter plot grids for S1 frozen-encoder model comparison.

Produces two figures:
  1. Raw scatter plot grid (7 models × 4 test sets)
  2. Hexbin density grid (same layout, better for 40K+ points)

Each subplot shows predicted vs true expression with Pearson R annotated.

Run:
    python scripts/analysis/plot_s1_scatterplots.py
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

# Model definitions: (display_name, predictions_dir, color)
# Order: best to worst by in_dist Pearson (AG at top)
MODELS = [
    ("AG all folds", "ag_all_folds_k562_s1_full", "#1B5E20"),
    ("AG fold 1", "ag_fold_1_k562_s1_full", "#66BB6A"),
    (
        "DREAM-RNN",
        "dream_rnn_k562_with_preds/seed_42/seed_42/fraction_1.0000",
        "#7B2D8E",
    ),
    ("Enformer", "enformer_k562_3seeds/seed_598125057", "#3A86C8"),
    ("Malinois", "malinois_k562_with_preds/seed_42", "#B07CC6"),
    ("Borzoi", "borzoi_k562_3seeds/seed_824292012", "#DAA520"),
    (
        "NTv3",
        "foundation_grid_search/ntv3_post/lr0.0005_wd1e-6_do0.1/seed_42/seed_42",
        "#E8602C",
    ),
]

TEST_SETS = [
    ("in_dist", "Reference\n(in-distribution)"),
    ("snv_abs", "SNV\n(absolute)"),
    ("snv_delta", "SNV effect\n(delta)"),
    ("ood", "Synthetic design\n(OOD)"),
]


def load_predictions(model_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """Load test_predictions.npz and return dict of (pred, true) per test set."""
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
    if mask.sum() < 3:
        return 0.0
    return float(pearsonr(pred[mask], true[mask])[0])


def _mse(pred, true):
    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() < 1:
        return 0.0
    return float(np.mean((pred[mask] - true[mask]) ** 2))


def plot_scatter_grid(all_data: dict[str, dict]) -> Path | None:
    """7×4 grid of raw scatter plots."""
    n_models = len(MODELS)
    n_tests = len(TEST_SETS)

    fig, axes = plt.subplots(
        n_models, n_tests, figsize=(4 * n_tests, 3.5 * n_models), squeeze=False
    )

    for row, (model_name, model_dir_name, color) in enumerate(MODELS):
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

                # Subsample for scatter if too many points
                n = len(pred)
                if n > 5000:
                    idx = np.random.default_rng(42).choice(n, size=5000, replace=False)
                    pred_plot, true_plot = pred[idx], true[idx]
                else:
                    pred_plot, true_plot = pred, true

                ax.scatter(true_plot, pred_plot, s=1, alpha=0.15, color=color, rasterized=True)

                # Identity line
                lims = [
                    min(np.nanmin(true_plot), np.nanmin(pred_plot)),
                    max(np.nanmax(true_plot), np.nanmax(pred_plot)),
                ]
                ax.plot(lims, lims, "k--", lw=0.5, alpha=0.5)

                # Annotate
                ax.text(
                    0.05,
                    0.95,
                    f"R = {r:.3f}\nMSE = {mse:.3f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

            # Labels
            if row == 0:
                ax.set_title(test_label, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(model_name, fontsize=11, fontweight="bold")
            if row == n_models - 1:
                ax.set_xlabel("True expression", fontsize=9)
            if col == 0:
                pass  # ylabel is the model name

    fig.suptitle(
        "K562 MPRA — Stage 1 (Frozen Encoder) Predicted vs True",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "k562_s1_scatter_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / "k562_s1_scatter_grid.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")
    return out


def plot_hexbin_grid(all_data: dict[str, dict]) -> Path | None:
    """7×4 grid of hexbin density plots."""
    n_models = len(MODELS)
    n_tests = len(TEST_SETS)

    fig, axes = plt.subplots(
        n_models, n_tests, figsize=(4 * n_tests, 3.5 * n_models), squeeze=False
    )

    for row, (model_name, model_dir_name, color) in enumerate(MODELS):
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

                mask = np.isfinite(pred) & np.isfinite(true)
                ax.hexbin(
                    true[mask],
                    pred[mask],
                    gridsize=50,
                    cmap="viridis",
                    mincnt=1,
                    linewidths=0.1,
                )

                # Identity line
                lims = [
                    min(np.nanmin(true[mask]), np.nanmin(pred[mask])),
                    max(np.nanmax(true[mask]), np.nanmax(pred[mask])),
                ]
                ax.plot(lims, lims, "r--", lw=1, alpha=0.7)

                # Annotate
                ax.text(
                    0.05,
                    0.95,
                    f"R = {r:.3f}\nMSE = {mse:.3f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

            if row == 0:
                ax.set_title(test_label, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(model_name, fontsize=11, fontweight="bold")
            if row == n_models - 1:
                ax.set_xlabel("True expression", fontsize=9)

    fig.suptitle(
        "K562 MPRA — Stage 1 (Frozen Encoder) Predicted vs True (Density)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "k562_s1_hexbin_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / "k562_s1_hexbin_grid.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading predictions...")

    all_data: dict[str, dict] = {}
    for model_name, model_dir_name, color in MODELS:
        model_dir = REPO / "outputs" / model_dir_name
        data = load_predictions(model_dir)
        if data:
            all_data[model_name] = data
            n_tests = len(data)
            n_points = sum(len(v[0]) for v in data.values())
            print(f"  {model_name}: {n_tests} test sets, {n_points:,} total points")
        else:
            print(f"  {model_name}: NO PREDICTIONS (will show 'No data')")

    if not all_data:
        print("No prediction data found. Run save_s1_predictions.py first.")
        return

    print(f"\nModels with data: {len(all_data)}/{len(MODELS)}")
    print("\nGenerating scatter grid...")
    plot_scatter_grid(all_data)

    print("Generating hexbin grid...")
    plot_hexbin_grid(all_data)

    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
