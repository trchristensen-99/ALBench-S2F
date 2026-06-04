#!/usr/bin/env python3
"""Plot acquisition strategy experiments:
1. Uncertainty acquisition (oracle vs student) vs random baseline
2. Pretrained fine-tuning (catastrophic forgetting)
3. Genomic+pool combined training (retrain vs replay)
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

available = set(f.name for f in fm.fontManager.ttflist)
for font in ["Calibri", "Arial", "Helvetica Neue", "Helvetica"]:
    if font in available:
        matplotlib.rcParams["font.family"] = font
        break

matplotlib.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    }
)

data = json.load(open("/tmp/all_experiments.json"))

STRAT_COLORS = {
    "random": "#888888",
    "evoaug_heavy": "#9467bd",
    "genomic": "#1f77b4",
    "prm_1pct": "#d62728",
    "prm_20pct": "#8c564b",
    "motif_grammar": "#2ca02c",
}


def _get_curve(exp_data, strat, metric):
    """Extract (sizes, means, stds) for a strategy+metric."""
    if strat not in exp_data:
        return [], [], []
    sizes, means, stds = [], [], []
    for n_str in sorted(exp_data[strat].keys(), key=int):
        vals = exp_data[strat][n_str].get(metric, [])
        if vals:
            sizes.append(int(n_str))
            means.append(np.mean(vals))
            stds.append(np.std(vals) if len(vals) > 1 else 0)
    return sizes, means, stds


def plot_with_ci(ax, sizes, means, stds, color, label, ls="-", lw=2, marker="o"):
    if not sizes:
        return
    ax.plot(sizes, means, color=color, ls=ls, lw=lw, marker=marker, markersize=4, label=label)
    if any(s > 0 for s in stds):
        lo = [m - s for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]
        ax.fill_between(sizes, lo, hi, alpha=0.15, color=color)


def style_ax(ax, title, xlabel="N Training Sequences", ylabel="Pearson R"):
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


out_dir = Path(__file__).parent / "acquisition_plots"
out_dir.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# Figure 1: Uncertainty Acquisition vs Random
# 3 panels: ID, OOD, SNV for each strategy that has uncertainty data
# ══════════════════════════════════════════════════════════════════════

metrics = [
    ("in_dist", "A. Genomic Sequences"),
    ("ood", "B. High-Activity Designed"),
    ("snv_delta", "C. SNV Effect"),
]

# Find strategies present in uncertainty data
unc_strats = sorted(
    set(data.get("uncertainty_oracle", {}).keys()) | set(data.get("uncertainty_student", {}).keys())
)

if unc_strats:
    # One row per strategy, 3 columns for metrics
    n_strats = len(unc_strats)
    fig, axes = plt.subplots(n_strats, 3, figsize=(16, 4.5 * n_strats), squeeze=False)

    mode_colors = {"random": "#2196F3", "oracle unc.": "#F44336", "student unc.": "#FF9800"}

    for row, strat in enumerate(unc_strats):
        for col, (mk, title) in enumerate(metrics):
            ax = axes[row, col]
            # Random baseline (definitive)
            s, m, sd = _get_curve(data["definitive"], strat, mk)
            plot_with_ci(ax, s, m, sd, mode_colors["random"], "Random downsample", ls="-", lw=2.5)
            # Oracle uncertainty
            s, m, sd = _get_curve(data.get("uncertainty_oracle", {}), strat, mk)
            plot_with_ci(
                ax, s, m, sd, mode_colors["oracle unc."], "Oracle uncertainty", ls="--", lw=2
            )
            # Student uncertainty
            s, m, sd = _get_curve(data.get("uncertainty_student", {}), strat, mk)
            plot_with_ci(
                ax, s, m, sd, mode_colors["student unc."], "Student uncertainty", ls="-.", lw=2
            )

            style_ax(ax, title if row == 0 else "")
            ax.set_ylim(0, 1)
            if col == 0:
                ax.set_ylabel(f"{strat}\nPearson R", fontweight="bold")
            if row == 0 and col == 0:
                ax.legend(fontsize=10, loc="lower right")

    fig.suptitle(
        "Uncertainty-Based Acquisition vs Random Downsampling\n(shaded = ±1 std across 3 replicates)",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "uncertainty_acquisition.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "uncertainty_acquisition.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: uncertainty_acquisition")

# ══════════════════════════════════════════════════════════════════════
# Figure 2: Pretrained Fine-Tuning vs Retraining from Scratch
# Shows: pretrained fine-tune (pool only), replay (pretrained + 1:1 genomic),
# retrain from scratch (genomic+pool)
# ══════════════════════════════════════════════════════════════════════

common_strats = sorted(
    set(data.get("pretrained", {}).keys()) & set(data.get("definitive", {}).keys())
)
# Filter to the 5 core strategies
common_strats = [s for s in common_strats if s in STRAT_COLORS]

if common_strats:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    mode_styles = [
        ("definitive", "From scratch (pool only)", "#2196F3", "-", 2.5, "o"),
        ("pretrained", "Fine-tune pretrained (pool only)", "#F44336", "--", 2, "s"),
        ("gp_v1_replay", "Replay (pretrained + 1:1 genomic mix)*", "#FF9800", "-.", 2, "^"),
    ]
    for ax, (mk, title) in zip(axes, metrics):
        for strat in common_strats:
            color = STRAT_COLORS[strat]
            for exp_key, label, mode_color, ls, lw, marker in mode_styles:
                s, m, sd = _get_curve(data.get(exp_key, {}), strat, mk)
                # Only label first strategy
                use_label = f"{label}" if strat == common_strats[0] else ""
                plot_with_ci(ax, s, m, sd, mode_color, use_label, ls=ls, lw=lw, marker=marker)

        style_ax(ax, title)
        ax.set_ylim(0, 1)
        if "A." in title:
            ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(
        "Fine-Tuning vs (Re)Training from Scratch\n(*replay/retrain used wd=1e-5; corrected runs in progress)",
        fontsize=13,
        fontweight="bold",
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "finetuning_vs_retraining.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "finetuning_vs_retraining.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: finetuning_vs_retraining")

# ══════════════════════════════════════════════════════════════════════
# Figure 3: Genomic+Pool Combined Retraining vs Pool-Only Scaling
# Only shows RETRAIN mode (from scratch on genomic+pool combined)
# ══════════════════════════════════════════════════════════════════════

gp_strats = sorted(data.get("gp_v1_retrain", {}).keys())
gp_strats = [s for s in gp_strats if s in STRAT_COLORS]

if gp_strats:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (mk, title) in zip(axes, metrics):
        for strat in gp_strats:
            color = STRAT_COLORS[strat]
            # From-scratch pool only (definitive)
            s, m, sd = _get_curve(data["definitive"], strat, mk)
            plot_with_ci(
                ax,
                s,
                m,
                sd,
                color,
                f"{strat} (pool only)" if strat == gp_strats[0] else "",
                ls="-",
                lw=2,
            )
            # Retrain (genomic + pool combined from scratch)
            s, m, sd = _get_curve(data["gp_v1_retrain"], strat, mk)
            plot_with_ci(
                ax,
                s,
                m,
                sd,
                color,
                f"{strat} (retrain: genomic+pool)*" if strat == gp_strats[0] else "",
                ls="--",
                lw=2,
                marker="s",
            )

        style_ax(ax, title)
        ax.set_ylim(0, 1)
        if "A." in title:
            # Manual legend for modes only
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], color="gray", ls="-", lw=2, label="Pool only (from scratch)"),
                Line2D(
                    [0],
                    [0],
                    color="gray",
                    ls="--",
                    lw=2,
                    marker="s",
                    markersize=4,
                    label="Retrain from scratch (genomic+pool)*",
                ),
            ]
            ax.legend(handles=legend_elements, fontsize=10, loc="lower right")

    fig.suptitle(
        "Genomic+Pool Combined Retraining vs Pool-Only Scaling\n"
        "(*retrain used wd=1e-5; corrected runs in progress. Colors = strategies)",
        fontsize=13,
        fontweight="bold",
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "genomic_pool_retrain.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "genomic_pool_retrain.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: genomic_pool_retrain")

# ══════════════════════════════════════════════════════════════════════
# Figure 4: All acquisition modes comparison at a single strategy
# For the strategy with most data, show all modes on one plot
# ══════════════════════════════════════════════════════════════════════

# Pick a strategy that has data across all experiments
for focus_strat in ["evoaug_heavy", "motif_grammar", "prm_1pct", "random"]:
    has_all = all(
        focus_strat in data.get(exp, {})
        for exp in ["definitive", "pretrained", "gp_v1_retrain", "gp_v1_replay"]
    )
    if has_all:
        break

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
mode_styles = [
    ("definitive", "Pool only (from scratch)", "#2196F3", "-", 2.5, "o"),
    ("pretrained", "Pretrained fine-tune", "#F44336", "--", 2, "s"),
    ("gp_v1_retrain", "Retrain (genomic+pool)*", "#4CAF50", "-", 2, "D"),
    ("gp_v1_replay", "Replay (pretrained+genomic)*", "#FF9800", "-.", 1.8, "^"),
]
if focus_strat in data.get("uncertainty_oracle", {}):
    mode_styles.append(("uncertainty_oracle", "Oracle uncertainty acq.", "#9C27B0", ":", 2, "v"))
if focus_strat in data.get("uncertainty_student", {}):
    mode_styles.append(
        ("uncertainty_student", "Student uncertainty acq.", "#795548", ":", 1.5, "P")
    )

for ax, (mk, title) in zip(axes, metrics):
    for exp_key, label, color, ls, lw, marker in mode_styles:
        s, m, sd = _get_curve(data.get(exp_key, {}), focus_strat, mk)
        plot_with_ci(ax, s, m, sd, color, label, ls=ls, lw=lw, marker=marker)

    style_ax(ax, title)
    ax.set_ylim(0, 1)
    if "A." in title:
        ax.legend(fontsize=8, loc="lower right")

fig.suptitle(
    f"All Acquisition Modes Compared — {focus_strat}\n(*retrain/replay used wd=1e-5; v2 with correct wd running)",
    fontsize=13,
    fontweight="bold",
    y=1.04,
)
fig.tight_layout()
fig.savefig(out_dir / "all_modes_comparison.png", dpi=200, bbox_inches="tight")
fig.savefig(out_dir / "all_modes_comparison.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved: all_modes_comparison ({focus_strat})")

print(f"\nAll plots saved to {out_dir}")
