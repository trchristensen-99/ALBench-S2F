#!/usr/bin/env python3
"""Publication-quality figure: systematic random DNA bias across models.

Shows that ALL tested sequence-to-function models predict substantial positive
regulatory activity for random DNA sequences, which should have near-zero
activity in mammalian cells (Camellato et al. 2024).

Generates:
  1. Distribution panel (violin/histogram) for each model
  2. Summary bar chart of mean predicted activity
  3. Annotated with expected baseline (log2FC = 0)

Uses pre-computed predictions from the HPC multi-model comparison.
Can also regenerate predictions locally if the NPZ file is available.

Usage:
    python scripts/analysis/plot_random_dna_bias.py [--npz PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "peter_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Model display configuration
MODEL_CONFIG = {
    "AG_S2_new_856K": {
        "name": "AlphaGenome\nS2 Oracle",
        "short": "AG S2",
        "color": "#1B5E20",
        "type": "Foundation",
    },
    "Malinois_pretrained": {
        "name": "Malinois\n(pretrained)",
        "short": "Malinois",
        "color": "#1565C0",
        "type": "CNN",
    },
    "DREAM-RNN_Oracle": {
        "name": "DREAM-RNN\nOracle",
        "short": "DREAM-RNN",
        "color": "#7B2D8E",
        "type": "RNN",
    },
    "LegNet_real_labels": {
        "name": "LegNet\n(real labels)",
        "short": "LegNet (real)",
        "color": "#E8602C",
        "type": "CNN",
    },
    "LegNet_Oracle": {
        "name": "LegNet\nOracle",
        "short": "LegNet (oracle)",
        "color": "#D4A017",
        "type": "CNN",
    },
}

# Display order
MODEL_ORDER = [
    "AG_S2_new_856K",
    "Malinois_pretrained",
    "DREAM-RNN_Oracle",
    "LegNet_real_labels",
    "LegNet_Oracle",
]


def load_predictions(npz_path: Path | None) -> dict[str, np.ndarray]:
    """Load predictions from NPZ or use hardcoded summary stats."""
    if npz_path and npz_path.exists():
        data = np.load(npz_path)
        return {k: data[k] for k in data.files}

    # Generate from scratch using same RNG seed
    print("NPZ not found, generating predictions from scratch...")
    print("(This requires model checkpoints — use --npz to load pre-computed)")
    sys.exit(1)


def plot_random_bias(preds: dict[str, np.ndarray], out_dir: Path):
    """Create publication-quality random DNA bias figure."""
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.35, wspace=0.3)

    # ─── Panel A: Overlaid histograms ───
    ax_hist = fig.add_subplot(gs[0, 0])
    bins = np.linspace(-3, 7, 80)

    for key in MODEL_ORDER:
        if key not in preds:
            continue
        cfg = MODEL_CONFIG[key]
        p = preds[key]
        ax_hist.hist(
            p,
            bins=bins,
            alpha=0.3,
            color=cfg["color"],
            label=f"{cfg['short']} (μ={np.mean(p):.2f})",
            density=True,
            linewidth=0,
        )
        # Add KDE-like outline
        counts, edges = np.histogram(p, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax_hist.plot(centers, counts, color=cfg["color"], linewidth=1.5, alpha=0.8)

    ax_hist.axvline(0, color="k", ls="--", lw=2, alpha=0.7, label="Expected (log2FC=0)")
    ax_hist.set_xlabel("Predicted K562 log2FC", fontsize=11)
    ax_hist.set_ylabel("Density", fontsize=11)
    ax_hist.set_title("A. Predicted activity distributions", fontsize=12, fontweight="bold")
    ax_hist.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.set_xlim(-3, 7)

    # ─── Panel B: Violin plots ───
    ax_violin = fig.add_subplot(gs[0, 1])
    positions = []
    labels = []
    colors = []
    data_list = []
    for i, key in enumerate(MODEL_ORDER):
        if key not in preds:
            continue
        cfg = MODEL_CONFIG[key]
        positions.append(i)
        labels.append(cfg["name"])
        colors.append(cfg["color"])
        data_list.append(preds[key])

    parts = ax_violin.violinplot(
        data_list,
        positions=positions,
        showmeans=True,
        showmedians=True,
        showextrema=False,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.4)
        pc.set_edgecolor(colors[i])
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(2)
    parts["cmedians"].set_color("gray")
    parts["cmedians"].set_linewidth(1)
    parts["cmedians"].set_linestyle("--")

    ax_violin.axhline(0, color="k", ls="--", lw=2, alpha=0.5, label="Expected baseline")
    ax_violin.set_xticks(positions)
    ax_violin.set_xticklabels(labels, fontsize=8)
    ax_violin.set_ylabel("Predicted K562 log2FC", fontsize=11)
    ax_violin.set_title("B. Per-model distributions", fontsize=12, fontweight="bold")
    ax_violin.spines["top"].set_visible(False)
    ax_violin.spines["right"].set_visible(False)

    # ─── Panel C: Summary bar chart ───
    ax_bar = fig.add_subplot(gs[1, 0])
    means = []
    stds = []
    bar_colors = []
    bar_labels = []
    for key in MODEL_ORDER:
        if key not in preds:
            continue
        cfg = MODEL_CONFIG[key]
        p = preds[key]
        means.append(np.mean(p))
        stds.append(np.std(p) / np.sqrt(len(p)))  # SEM
        bar_colors.append(cfg["color"])
        bar_labels.append(cfg["short"])

    x = np.arange(len(means))
    bars = ax_bar.bar(x, means, color=bar_colors, alpha=0.7, edgecolor="white", linewidth=0.5)
    ax_bar.errorbar(x, means, yerr=stds, fmt="none", ecolor="black", capsize=4, capthick=1.5)
    ax_bar.axhline(0, color="k", ls="--", lw=2, alpha=0.5)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(bar_labels, fontsize=9, rotation=15)
    ax_bar.set_ylabel("Mean predicted log2FC", fontsize=11)
    ax_bar.set_title("C. Mean bias (expected = 0)", fontsize=12, fontweight="bold")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # Add value labels on bars
    for bar, m in zip(bars, means):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{m:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # ─── Panel D: Summary statistics table ───
    ax_table = fig.add_subplot(gs[1, 1])
    ax_table.axis("off")

    table_data = []
    col_labels = ["Model", "Type", "Mean", "Median", "Std", "% > 0"]
    for key in MODEL_ORDER:
        if key not in preds:
            continue
        cfg = MODEL_CONFIG[key]
        p = preds[key]
        table_data.append(
            [
                cfg["short"],
                cfg["type"],
                f"{np.mean(p):.3f}",
                f"{np.median(p):.3f}",
                f"{np.std(p):.3f}",
                f"{100 * np.mean(p > 0):.1f}%",
            ]
        )

    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colColours=["#f0f0f0"] * len(col_labels),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # Color model name cells
    for i, key in enumerate(MODEL_ORDER):
        if key not in preds:
            continue
        cell = table[i + 1, 0]
        cell.set_text_props(fontweight="bold", color=MODEL_CONFIG[key]["color"])

    ax_table.set_title(
        "D. Summary: 10K random 200bp sequences",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )

    fig.suptitle(
        "Systematic Random DNA Bias Across Sequence-to-Function Models\n"
        "All models predict substantial positive activity for random DNA "
        "(expected: log2FC ≈ 0)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    fig.savefig(out_dir / "random_dna_bias_systematic.png", dpi=250, bbox_inches="tight")
    fig.savefig(out_dir / "random_dna_bias_systematic.pdf", dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_dir / 'random_dna_bias_systematic.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        type=Path,
        default=REPO
        / "outputs"
        / "oracle_full_856k"
        / "random_bias_comparison"
        / "multi_model_random_preds.npz",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    preds = load_predictions(args.npz)
    print("Loaded models:", list(preds.keys()))
    for k, v in preds.items():
        print(f"  {k}: n={len(v)}, mean={np.mean(v):.3f}, std={np.std(v):.3f}")

    plot_random_bias(preds, args.out_dir)


if __name__ == "__main__":
    main()
