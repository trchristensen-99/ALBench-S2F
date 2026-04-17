#!/usr/bin/env python3
"""Generate poster figures for Stowers AI for Biology conference.

Panel 5: Debiasing summary — random DNA pred vs OOD for each approach
Panel 7: Updated strategy scaling curves from exp1_1_final
Panel 8: Log-log power law from exp1_1_final

Usage:
    python scripts/analysis/plot_poster_figures.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "poster_stowers"
OUT.mkdir(parents=True, exist_ok=True)

KEY_STRATEGIES = {
    "random": ("Random", "#888888", "--", 2.5),
    "genomic": ("Genomic", "#1f77b4", "-", 2.0),
    "prm_1pct": ("PRM 1%", "#d62728", "-", 2.0),
    "prm_20pct": ("PRM 20%", "#8c564b", "-", 2.0),
    "motif_grammar": ("Motif Grammar", "#2ca02c", "-", 2.0),
    "evoaug_heavy": ("EvoAug", "#9467bd", "-", 2.0),
}


def load_scaling_data():
    """Load deduplicated scaling results from exp1_1_final."""
    results = defaultdict(lambda: defaultdict(list))
    seen = set()

    for pattern in [
        "outputs/exp1_1_definitive/k562/legnet_ag_s2/*/n*/rep*/result.json",
        "outputs/exp1_1_definitive/k562/legnet_ag_s2/*/n*/hp*/seed*/result.json",
        "outputs/exp1_1_final/k562/legnet_ag_s2/*/n*/rep*/result.json",
        "outputs/exp1_1_final/k562/legnet_ag_s2/*/n*/hp*/seed*/result.json",
    ]:
        for f in Path(REPO).glob(pattern):
            d = json.loads(f.read_text())
            seed = d.get("seed", str(f))
            n = d.get("n_train", 0)
            strat = d.get("reservoir", "")
            if not strat:
                parts = f.parts
                for s in KEY_STRATEGIES:
                    if s in parts:
                        strat = s
                        break
            key = (strat, n, seed)
            if key in seen:
                continue
            seen.add(key)

            tm = d.get("test_metrics", {})
            idr = tm.get("in_dist", {}).get("pearson_r")
            ood = tm.get("ood", {}).get("pearson_r")
            snv_d = tm.get("snv_delta", {}).get("pearson_r")
            if idr is not None:
                results[strat][n].append({"id": idr, "ood": ood, "snv_d": snv_d})

    return results


def load_debias_data():
    """Load debiasing sweep results (test_metrics + bias_eval)."""
    results = []
    sweep_dir = REPO / "outputs" / "debias_sweep"
    if not sweep_dir.exists():
        return results

    for config_dir in sorted(sweep_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        tm_f = config_dir / "test_metrics.json"
        be_f = config_dir / "bias_eval.json"
        if not tm_f.exists():
            continue

        tm = json.loads(tm_f.read_text())
        test = tm.get("test_metrics", {})
        idr = test.get("in_distribution", {}).get("pearson_r", 0)
        oodr = test.get("ood", {}).get("pearson_r", 0)

        rand_dna = None
        if be_f.exists():
            be = json.loads(be_f.read_text())
            rand_dna = be.get("random_dna", {}).get("mean")

        results.append(
            {
                "name": config_dir.name,
                "id_r": idr,
                "ood_r": oodr,
                "rand_dna": rand_dna,
            }
        )
    return results


def plot_debiasing_summary(debias_data):
    """Panel 5: 4 points — baseline + best of each strategy. Legend only, no labels."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Find best per category (highest OOD)
    best = {}
    for d in debias_data:
        if d["rand_dna"] is None:
            continue
        cat = "Combined" if "combo" in d["name"] else "Neg-aug" if "negaug" in d["name"] else "Loss"
        if cat not in best or d["ood_r"] > best[cat]["ood_r"]:
            best[cat] = d

    # Baseline point (no debiasing)
    ax.scatter(
        [0.75],
        [0.745],
        c="#888888",
        marker="X",
        s=120,
        label="Baseline (no debiasing)",
        edgecolors="black",
        linewidths=0.5,
        zorder=4,
    )

    # Best of each category
    points = [
        ("Loss", "Loss regularization", "#3498db", "o", "spectral_l01"),
        ("Neg-aug", "Negative augmentation", "#e74c3c", "s", None),
        ("Combined", "Combined", "#2ecc71", "D", None),
    ]
    for cat, label, color, marker, override in points:
        if override:
            # Use specific config instead of auto-best
            for d in debias_data:
                if d["name"] == override and d["rand_dna"] is not None:
                    best[cat] = d
                    break
        if cat not in best:
            continue
        d = best[cat]
        ax.scatter(
            [d["rand_dna"]],
            [d["ood_r"]],
            c=color,
            marker=marker,
            s=120,
            label=label,
            edgecolors="black",
            linewidths=0.5,
            zorder=4,
        )

    # Reference lines
    ax.axhline(y=0.745, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=0.75, color="gray", linestyle=":", alpha=0.4, linewidth=1)
    ax.axvline(x=0.27, color="#27ae60", linestyle=":", alpha=0.5, linewidth=1.5)

    # Compact annotations inside plot bounds
    ax.text(
        0.72, 0.43, "Baseline\nbias", fontsize=7, color="gray", alpha=0.7, ha="right", rotation=90
    )
    ax.text(0.29, 0.43, "Target", fontsize=7, color="#27ae60", alpha=0.8, rotation=90)

    ax.set_xlim(0.05, 0.85)
    ax.set_ylim(0.40, 0.82)
    ax.set_xlabel("Random DNA Prediction (lower = less biased)", fontsize=11)
    ax.set_ylabel("OOD Pearson R (higher = better)", fontsize=11)
    ax.set_title("Oracle Debiasing Approaches", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", frameon=True, facecolor="white", edgecolor="gray")
    ax.grid(alpha=0.2, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "panel5_debiasing_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "panel5_debiasing_summary.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: panel5_debiasing_summary.png")


def plot_scaling_curves(
    data, metric="id", ylabel="In-Dist Pearson R", title="Strategy Scaling Laws", filename="panel7"
):
    """Panel 7/8: Strategy scaling curves."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for strat, (label, color, ls, lw) in KEY_STRATEGIES.items():
        if strat not in data:
            continue
        sizes = sorted(data[strat].keys())
        means, stds, valid_sizes = [], [], []
        for n in sizes:
            vals = [r[metric] for r in data[strat][n] if r.get(metric) is not None]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals) if len(vals) > 1 else 0)
                valid_sizes.append(n)

        if not valid_sizes:
            continue

        means = np.array(means)
        stds = np.array(stds)
        ci95 = 1.96 * stds
        zorder = 10 if strat == "random" else 1

        ax.plot(
            valid_sizes,
            means,
            color=color,
            label=label,
            linewidth=lw,
            linestyle=ls,
            marker="o",
            markersize=5,
            zorder=zorder,
        )
        if stds.any():
            ax.fill_between(
                valid_sizes, means - ci95, means + ci95, alpha=0.18, color=color, zorder=zorder - 1
            )
            ax.errorbar(
                valid_sizes,
                means,
                yerr=ci95,
                fmt="none",
                ecolor=color,
                elinewidth=1.2,
                capsize=4,
                capthick=1.2,
                alpha=0.7,
                zorder=zorder,
            )

    ax.set_xscale("log")
    ax.set_xlabel("N Training Sequences", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", frameon=True, facecolor="white")
    ax.grid(alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / f"{filename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}.png")


def plot_loglog(data):
    """Panel 8: Log-log scaling law."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for strat, (label, color, ls, lw) in KEY_STRATEGIES.items():
        if strat not in data:
            continue
        sizes = sorted(data[strat].keys())
        means, valid_sizes = [], []
        for n in sizes:
            vals = [r["id"] for r in data[strat][n] if r.get("id") is not None]
            if vals:
                means.append(np.mean(vals))
                valid_sizes.append(n)

        if len(valid_sizes) < 2:
            continue

        loss = np.clip(1.0 - np.array(means), 1e-6, None)
        sizes_arr = np.array(valid_sizes, dtype=float)

        # Fit power law
        fit_mask = loss > 0.01
        if fit_mask.sum() >= 2:
            log_n = np.log10(sizes_arr[fit_mask])
            log_loss = np.log10(loss[fit_mask])
            coeffs = np.polyfit(log_n, log_loss, 1)
            alpha = -coeffs[0]
            fit_label = f"{label} (α={alpha:.2f})"
            # Fit line
            fit_x = np.logspace(np.log10(sizes_arr.min()), np.log10(sizes_arr.max()), 50)
            fit_y = 10 ** np.polyval(coeffs, np.log10(fit_x))
            ax.plot(fit_x, fit_y, color=color, linestyle=":", linewidth=1.0, alpha=0.5)
        else:
            fit_label = label

        zorder = 10 if strat == "random" else 1
        ax.plot(
            sizes_arr,
            loss,
            color=color,
            label=fit_label,
            linewidth=lw,
            linestyle=ls,
            marker="o",
            markersize=5,
            zorder=zorder,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N Training Sequences", fontsize=12)
    ax.set_ylabel("1 − Pearson R (loss)", fontsize=12)
    ax.set_title("Scaling Law (Log-Log)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", frameon=True, facecolor="white")
    ax.grid(alpha=0.3, which="both", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "panel8_loglog.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "panel8_loglog.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: panel8_loglog.png")


def main():
    print("Loading data...")
    scaling_data = load_scaling_data()
    debias_data = load_debias_data()

    print(
        f"  Scaling: {sum(len(v) for s in scaling_data for v in scaling_data[s].values())} results"
    )
    print(f"  Debias: {len(debias_data)} configs")

    # Panel 5: Debiasing
    if debias_data:
        plot_debiasing_summary(debias_data)

    # Panel 7: Strategy scaling (in-dist)
    if scaling_data:
        plot_scaling_curves(
            scaling_data,
            metric="id",
            ylabel="In-Dist Pearson R",
            title="LegNet Scaling by Reservoir Strategy (K562, AG S2 Oracle)",
            filename="panel7_strategy_scaling",
        )

        # Panel 7b: OOD
        plot_scaling_curves(
            scaling_data,
            metric="ood",
            ylabel="OOD Pearson R",
            title="OOD Scaling by Reservoir Strategy",
            filename="panel7b_strategy_ood",
        )

        # Panel 8: Log-log
        plot_loglog(scaling_data)

        # Combined 3-panel: ID + OOD + SNV delta
        plot_scaling_3panel(scaling_data)

    print(f"\nAll figures in: {OUT}")


def plot_scaling_3panel(data):
    """3-panel scaling: ID, OOD, SNV delta side by side."""
    metrics = [
        ("id", "In-Dist Pearson R", "A"),
        ("ood", "OOD Pearson R", "B"),
        ("snv_d", "SNV Effect (Δ) Pearson R", "C"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (metric, ylabel, panel_label) in zip(axes, metrics):
        for strat, (label, color, ls, lw) in KEY_STRATEGIES.items():
            if strat not in data:
                continue
            sizes = sorted(data[strat].keys())
            means, stds, valid_sizes = [], [], []
            for n in sizes:
                vals = [r[metric] for r in data[strat][n] if r.get(metric) is not None]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals) if len(vals) > 1 else 0)
                    valid_sizes.append(n)

            if not valid_sizes:
                continue

            means_arr = np.array(means)
            stds_arr = np.array(stds)
            ci95 = 1.96 * stds_arr
            zorder = 10 if strat == "random" else 1

            ax.plot(
                valid_sizes,
                means_arr,
                color=color,
                label=label,
                linewidth=lw,
                linestyle=ls,
                marker="o",
                markersize=4,
                zorder=zorder,
            )
            if stds_arr.any():
                ax.fill_between(
                    valid_sizes,
                    means_arr - ci95,
                    means_arr + ci95,
                    alpha=0.15,
                    color=color,
                    zorder=zorder - 1,
                )

        ax.set_xscale("log")
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("N Training Sequences", fontsize=11)
        ax.set_ylabel(ylabel if panel_label == "A" else "", fontsize=11)
        ax.set_title(f"{panel_label}. {ylabel}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right", frameon=True, facecolor="white")
        ax.grid(alpha=0.3, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "LegNet Scaling by Reservoir Strategy (K562, AG S2 Oracle)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "panel7_scaling_3panel.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "panel7_scaling_3panel.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: panel7_scaling_3panel.png")


if __name__ == "__main__":
    main()
