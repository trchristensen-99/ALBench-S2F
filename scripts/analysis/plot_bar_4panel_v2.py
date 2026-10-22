"""Single-plot 4-panel bar chart matching Alan's `plot_episomal_cell_averaged` style.

Layout (mirrors episomal_bar_cell_avg.png):
- Single axes, 4 bar groups, 6 bars per group
- Legend in top-right, frameon=False, ncol=2
- Group labels at the bottom (multiline, no x-axis title)
- Value labels above each bar: bold, rotated 90°, colored to match the bar
- Y-axis: "Pearson Correlation", 0-1, dashed gridlines (alpha 0.5)
- Top and right spines hidden
- Title: "Episomal MPRA"
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

MODEL_ORDER = [
    "DREAM-RNN",
    "Malinois",
    "Enf. (Probing)",
    "Enf. (Fine-tuned)",
    "AG (Probing)",
    "AG (Fine-tuned)",
]

PANEL_ORDER = [
    "Genomic Reference Sequences",
    "High-Activity Designed Sequences",
    "Genomic Alternate Sequences (SNVs)",
    "SNV Effects (Ref - Alt)",
]
PANEL_LABELS = [
    "Genomic Reference\nSequences",
    "High-Activity\nDesigned Sequences",
    "Genomic Alternate\nSequences (SNVs)",
    "SNV Effects\n(Ref − Alt)",
]

# Alan's STARR-seq palette (DREAM-RNN baseline color present).
MODEL_COLORS = {
    "DREAM-RNN":         "#8B9DAF",
    "Malinois":          "#E8DCCF",
    "Enf. (Probing)":    "#E7CDC2",
    "Enf. (Fine-tuned)": "#A65141",
    "AG (Probing)":      "#80A0C7",
    "AG (Fine-tuned)":   "#394165",
}

TEXT_COLORS = {
    "#E8DCCF": "#8B7D6B",
    "#E7CDC2": "#8B6B61",
    "#8B9DAF": "#5A6A7A",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    print(f"Loaded {len(df)} rows from {args.in_csv}")

    sns.set(font_scale=1.2)
    sns.set_style("white")

    n_models = len(MODEL_ORDER)
    n_groups = len(PANEL_ORDER)
    width = 0.13
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for i, model in enumerate(MODEL_ORDER):
        color = MODEL_COLORS[model]
        means = []
        stds = []
        for panel in PANEL_ORDER:
            sub = df[(df["model"] == model) & (df["panel"] == panel)]
            if sub.empty:
                means.append(0.0); stds.append(0.0)
            else:
                means.append(float(sub["weighted_mean"].iloc[0]))
                stds.append(float(sub["weighted_sd"].iloc[0]))

        offset = (i - n_models / 2) * width + width / 2
        bars = ax.bar(
            x + offset, means, width,
            yerr=[s if s > 0 else 0 for s in stds],
            capsize=3, label=model, color=color,
            edgecolor="black", linewidth=1, alpha=0.9,
            error_kw=dict(ecolor="black", capsize=2, lw=0.8),
        )
        txt_color = TEXT_COLORS.get(color, color)
        for rect, val, err in zip(bars, means, stds):
            if val > 0:
                cx = rect.get_x() + rect.get_width() / 2.0
                ax.annotate(
                    f"{val:.3f}",
                    xy=(cx, val + err),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold", rotation=90,
                    color=txt_color,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(PANEL_LABELS)
    ax.set_ylabel("Pearson Correlation")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Episomal MPRA", fontsize=14)
    ax.yaxis.grid(alpha=0.5, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0),
              frameon=False, fontsize=9, ncol=1)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved → {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
