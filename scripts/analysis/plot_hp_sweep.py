#!/usr/bin/env python3
"""Plot HP sweep results: LR × BS grid with panels per LR.

Creates figures with vertically stacked panels (one per LR, decreasing top→bottom),
x-axis = batch size, y-axis = Pearson R. Fixed y-axis range across panels.
Separate figures for: combined avg, genomic (ID), high-activity (OOD), SNV effect.
"""

import json
import os
from pathlib import Path

import matplotlib

# Font setup
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
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    }
)

# Load data — try local cache, fall back to exported JSON
_data_path = "/tmp/hp_sweep_all.json"
if not os.path.exists(_data_path):
    raise FileNotFoundError(f"Run data export first: {_data_path}")
data = json.load(open(_data_path))

# Strategy colors
STRAT_COLORS = {
    "random": "#888888",
    "evoaug_heavy": "#9467bd",
    "genomic": "#1f77b4",
    "prm_1pct": "#d62728",
    "prm_20pct": "#8c564b",
    "motif_grammar": "#2ca02c",
}
STRAT_LABELS = {
    "random": "Random",
    "evoaug_heavy": "EvoAug",
    "genomic": "Genomic",
    "prm_1pct": "PRM 1%",
    "prm_20pct": "PRM 20%",
    "motif_grammar": "Motif Grammar",
}

# Organize: {(strat, n, lr, bs): {metric: [values]}}
organized = {}
for d in data:
    key = (d["reservoir"], d["n_train"], round(d["lr"], 6), d["bs"])
    organized[key] = {
        "in_dist": d["in_dist"],
        "ood": d["ood"],
        "snv_delta": d["snv_delta"],
    }

# LRs and BSs to plot
ALL_BS = [32, 64, 128, 256, 512, 1024, 2048]
# Select LRs that have data at BS=256 (most common)
lr_set = sorted(set(k[2] for k in organized), reverse=True)
# Pick representative LRs: focus on the range that matters
TARGET_LRS_100K = [0.012, 0.008, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0005]
TARGET_LRS_5K = [0.003, 0.002, 0.001, 0.0005]

METRICS_BY_N = {
    5000: {
        "in_dist": ("Genomic Sequences (ID)", 0.45, 0.70),
        "ood": ("High-Activity Designed (OOD)", -0.05, 0.30),
        "snv_delta": ("SNV Effect", 0.10, 0.45),
        "combined": ("Combined (avg of ID + OOD + SNV)", 0.20, 0.45),
    },
    100000: {
        "in_dist": ("Genomic Sequences (ID)", 0.84, 0.92),
        "ood": ("High-Activity Designed (OOD)", 0.25, 0.50),
        "snv_delta": ("SNV Effect", 0.55, 0.75),
        "combined": ("Combined (avg of ID + OOD + SNV)", 0.55, 0.70),
    },
}


def make_figure(metric_key, title, ymin, ymax, target_n, strats):
    """Create stacked panel figure for one metric at one training size."""
    candidate_lrs = TARGET_LRS_100K if target_n >= 100000 else TARGET_LRS_5K
    # Filter to LRs that have at least one data point
    TARGET_LRS = []
    for lr in candidate_lrs:
        has_data = any(
            organized.get((s, target_n, lr, bs), {}).get(
                "in_dist" if metric_key != "combined" else "in_dist", []
            )
            for s in strats
            for bs in ALL_BS
        )
        if has_data:
            TARGET_LRS.append(lr)
    if not TARGET_LRS:
        return None
    n_panels = len(TARGET_LRS)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 2.2 * n_panels), sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax_idx, lr in enumerate(TARGET_LRS):
        ax = axes[ax_idx]

        for strat in strats:
            bs_vals = []
            means = []
            stds = []
            n_reps_list = []
            for bs in ALL_BS:
                key = (strat, target_n, lr, bs)
                v = organized.get(key, {})

                if metric_key == "combined":
                    # Average across all 3 metrics
                    all_combined = []
                    min_reps = 999
                    for mk in ["in_dist", "ood", "snv_delta"]:
                        vals = v.get(mk, [])
                        if vals:
                            all_combined.append(np.mean(vals))
                            min_reps = min(min_reps, len(vals))
                    if len(all_combined) == 3:
                        bs_vals.append(bs)
                        means.append(np.mean(all_combined))
                        stds.append(0)
                        n_reps_list.append(min_reps)
                else:
                    vals = v.get(metric_key, [])
                    if vals:
                        bs_vals.append(bs)
                        means.append(np.mean(vals))
                        stds.append(np.std(vals) if len(vals) > 1 else 0)
                        n_reps_list.append(len(vals))

            if bs_vals:
                color = STRAT_COLORS.get(strat, "#333")
                label = STRAT_LABELS.get(strat, strat)
                # Show reps as small text annotation
                for bv, mv, sv, nv in zip(bs_vals, means, stds, n_reps_list):
                    if nv > 1:
                        ax.annotate(
                            f"n={nv}",
                            (bv, mv),
                            fontsize=6,
                            ha="center",
                            va="bottom",
                            color=color,
                            alpha=0.6,
                            xytext=(0, 4),
                            textcoords="offset points",
                        )
                ax.plot(
                    bs_vals,
                    means,
                    color=color,
                    marker="o",
                    markersize=5,
                    linewidth=1.5,
                    label=label,
                )
                # Show CI: use ±1 std for ≥3 reps, ±range/2 for 2 reps, no CI for 1 rep
                ci_lo = []
                ci_hi = []
                has_ci = False
                for mv, sv, nv in zip(means, stds, n_reps_list):
                    if nv >= 2 and sv > 0:
                        ci_lo.append(mv - sv)
                        ci_hi.append(mv + sv)
                        has_ci = True
                    else:
                        ci_lo.append(mv)
                        ci_hi.append(mv)
                if has_ci:
                    ax.fill_between(bs_vals, ci_lo, ci_hi, alpha=0.15, color=color)

        ax.set_ylabel("Pearson R")
        ax.set_ylim(ymin, ymax)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ALL_BS)
        ax.set_xticklabels([str(b) for b in ALL_BS])
        ax.set_title(f"LR = {lr}", fontsize=11, fontweight="bold", loc="left")
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add horizontal line for default config (lr=0.003/bs=256) avg performance
        default_key_vals = []
        for strat in strats:
            v = organized.get((strat, target_n, 0.003, 256), {})
            if metric_key == "combined":
                combo = []
                for mk in ["in_dist", "ood", "snv_delta"]:
                    if v.get(mk):
                        combo.append(np.mean(v[mk]))
                if len(combo) == 3:
                    default_key_vals.append(np.mean(combo))
            else:
                if v.get(metric_key):
                    default_key_vals.append(np.mean(v[metric_key]))
        if default_key_vals:
            ax.axhline(
                np.mean(default_key_vals),
                color="gray",
                ls="--",
                alpha=0.4,
                lw=1,
                zorder=0,
                label="Default avg (lr=0.003/bs=256)" if ax_idx == 0 else "",
            )

    axes[-1].set_xlabel("Batch Size")
    # Place legend on first panel with data
    handles, labels = [], []
    for ax in axes:
        h, lab = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, lab
            break
    if handles:
        # Deduplicate
        seen = set()
        uh, ul = [], []
        for h, lab in zip(handles, labels):
            if lab not in seen:
                seen.add(lab)
                uh.append(h)
                ul.append(lab)
        axes[0].legend(
            uh, ul, loc="upper right", frameon=True, facecolor="white", edgecolor="gray", ncol=2
        )

    fig.suptitle(f"{title}\nN = {target_n:,}", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# Generate all figures
out_dir = Path(__file__).parent / "hp_sweep_plots"
out_dir.mkdir(exist_ok=True)

strats_3 = ["random", "evoaug_heavy", "genomic"]

for target_n in [5000, 100000]:
    # Determine which strats have data at this N
    strats_with_data = []
    for s in ["random", "evoaug_heavy", "genomic", "prm_1pct", "prm_20pct", "motif_grammar"]:
        if any(k[0] == s and k[1] == target_n for k in organized):
            strats_with_data.append(s)

    metrics = METRICS_BY_N[target_n]
    for metric_key, (title, ymin, ymax) in metrics.items():
        fig = make_figure(metric_key, title, ymin, ymax, target_n, strats_with_data)
        if fig is None:
            continue
        fname = f"hp_sweep_n{target_n}_{metric_key}"
        fig.savefig(out_dir / f"{fname}.png", dpi=200, bbox_inches="tight")
        fig.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")

print(f"\nAll plots saved to {out_dir}")
