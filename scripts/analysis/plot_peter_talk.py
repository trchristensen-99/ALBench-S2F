#!/usr/bin/env python3
"""Generate presentation-quality figures for Peter's NYGC/MSKCC talks.

Panel A: Exp0 scaling curves — model architecture comparison at different data sizes
Panel B: Exp1.1 strategy comparison — LegNet student with AG S2 oracle, key strategies
Panel C: Strategy comparison OOD — same data but OOD metric

Key messages:
- Foundation models (AG) have flat scaling (pretrained knowledge dominates)
- From-scratch models show steep scaling (data-hungry)
- Strategic data selection can beat random at every scale
- Different strategies excel at different scales / metrics
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
OUT = REPO / "results" / "talk_figures"
OUT.mkdir(parents=True, exist_ok=True)


def load_exp0_scaling():
    """Load Exp0 K562 scaling curve data (AG oracle, in_dist Pearson R)."""
    base = REPO / "outputs" / "exp0_oracle_scaling_v4" / "k562"
    models = {
        "AG S1 (Probing)": ("alphagenome_k562_s1", "#2980B9"),
        "LegNet": ("legnet", "#D4A017"),
        "DREAM-CNN": ("dream_cnn", "#9B59B6"),
        "DREAM-RNN": ("dream_rnn", "#8B9DAF"),
    }

    data = {}
    for display_name, (model_name, color) in models.items():
        results_by_n = defaultdict(lambda: defaultdict(list))
        for rj in (base / model_name).rglob("result.json"):
            try:
                d = json.loads(rj.read_text())
            except Exception:
                continue
            n = d.get("n_train", 0)
            hp = json.dumps(d.get("hp_config", {}), sort_keys=True)
            val_r = d.get("val_pearson_r", 0)
            p = d.get("test_metrics", {}).get("in_dist", {}).get("pearson_r")
            if p is not None:
                results_by_n[n][hp].append((val_r, p))

        sizes, means, stds = [], [], []
        for n in sorted(results_by_n):
            hp_map = results_by_n[n]
            best_hp = max(hp_map, key=lambda k: np.mean([v[0] for v in hp_map[k]]))
            vals = [v[1] for v in hp_map[best_hp]]
            sizes.append(n)
            means.append(np.mean(vals))
            stds.append(np.std(vals) if len(vals) > 1 else 0)

        if sizes:
            data[display_name] = {
                "sizes": sizes,
                "means": np.array(means),
                "stds": np.array(stds),
                "color": color,
            }
    return data


def load_strategy_data():
    """Load LegNet + AG S2 strategy scaling data.

    Returns dict: {strategy: {n: [pearson_r values for best HP]}}
    Merges results from exp1_1 (1K-500K) and exp1_1_5m_scaling (5M).
    """
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Load from all scaling directories
    for base_path in [
        REPO / "outputs" / "exp1_1" / "k562" / "legnet_ag_s2",
        REPO / "outputs" / "exp1_1_2m_scaling" / "k562" / "legnet_ag_s2",
        REPO / "outputs" / "exp1_1_5m_scaling" / "k562" / "legnet_ag_s2",
    ]:
        if not base_path.exists():
            continue
        for rj in base_path.rglob("result.json"):
            try:
                d = json.loads(rj.read_text())
            except Exception:
                continue
            strategy = d.get("reservoir", str(rj.relative_to(base_path)).split("/")[0])
            n = d.get("n_train", 0)
            hp = json.dumps(d.get("hp_config", {}), sort_keys=True)
            val_r = d.get("val_pearson_r", 0)
            test = d.get("test_metrics", {})
            raw[strategy][n][hp].append((val_r, test))

    result = {}
    for strategy in raw:
        result[strategy] = {}
        for n, hp_map in raw[strategy].items():
            best_hp = max(hp_map, key=lambda k: np.mean([v[0] for v in hp_map[k]]))
            result[strategy][n] = hp_map[best_hp]
    return result


# Strategy display config
KEY_STRATEGIES = {
    "random": ("Random", "#888888", "--", 2.5),
    "genomic": ("Genomic", "#1f77b4", "-", 2.0),
    "dinuc_shuffle": ("Dinuc. Shuffle", "#ff7f0e", "-", 1.5),
    "prm_5pct": ("Mutagenesis 5%", "#e377c2", "-", 1.5),
    "evoaug_structural": ("EvoAug Structural", "#2ca02c", "-", 1.5),
    "evoaug_heavy": ("EvoAug Heavy", "#d62728", "-", 1.5),
    "recombination_uniform": ("Recombination", "#9467bd", "-", 1.5),
    "motif_grammar": ("Motif Grammar", "#8c564b", "-", 1.5),
    "motif_planted": ("Motif Planted", "#17becf", "-", 1.5),
}


def extract_metric(data, strategy, metric_path):
    """Extract metric values by strategy and training size."""
    if strategy not in data:
        return [], [], []
    sizes = sorted(data[strategy].keys())
    means, stds, valid_sizes = [], [], []
    for n in sizes:
        keys = metric_path.split(".")
        vals = []
        for _, test in data[strategy][n]:
            v = test
            for k in keys:
                v = v.get(k, {}) if isinstance(v, dict) else None
                if v is None:
                    break
            if v is not None:
                vals.append(v)
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals) if len(vals) > 1 else 0)
            valid_sizes.append(n)
    return valid_sizes, np.array(means), np.array(stds)


def plot_strategy_panel(ax, data, metric_path, ylabel, title, strategies=None):
    """Plot strategy scaling curves on a single axes."""
    if strategies is None:
        strategies = KEY_STRATEGIES

    for strat, (label, color, ls, lw) in strategies.items():
        sizes, means, stds = extract_metric(data, strat, metric_path)
        if not sizes:
            continue
        zorder = 10 if strat == "random" else 1
        ax.plot(
            sizes,
            means,
            color=color,
            label=label,
            linewidth=lw,
            linestyle=ls,
            marker="o",
            markersize=4,
            zorder=zorder,
        )
        if stds.any():
            ax.fill_between(
                sizes, means - stds, means + stds, alpha=0.15, color=color, zorder=zorder - 1
            )
            # Add visible error bars where CI is too narrow for fill_between
            ax.errorbar(
                sizes,
                means,
                yerr=stds,
                fmt="none",
                ecolor=color,
                elinewidth=1.0,
                capsize=3,
                capthick=1.0,
                alpha=0.6,
                zorder=zorder,
            )

    ax.set_xscale("log")
    ax.set_xlabel("N training sequences", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", frameon=True, facecolor="white", edgecolor="gray")
    ax.grid(alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_exp0_panel(ax, data):
    """Plot Exp0 scaling curves."""
    for name, d in data.items():
        valid = ~np.isnan(d["means"])
        sizes = np.array(d["sizes"])[valid]
        means = d["means"][valid]
        stds = d["stds"][valid]

        ax.plot(
            sizes,
            means,
            "o-",
            color=d["color"],
            label=name,
            linewidth=2.2,
            markersize=5,
            zorder=5,
        )
        if stds.any():
            ax.fill_between(sizes, means - stds, means + stds, alpha=0.12, color=d["color"])

    ax.set_xscale("log")
    ax.set_xlabel("N training sequences", fontsize=12)
    ax.set_ylabel("In-Dist Pearson R", fontsize=12)
    ax.set_title("A. Model Scaling (K562, AG S2 Oracle Labels)", fontsize=13, fontweight="bold")
    ax.set_ylim(0.35, 1.0)
    ax.legend(fontsize=10, loc="lower right", frameon=True, facecolor="white", edgecolor="gray")
    ax.grid(alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    print("Loading data...")
    exp0_data = load_exp0_scaling()
    strategy_data = load_strategy_data()

    print(f"  Exp0: {len(exp0_data)} models")
    print(f"  Strategy: {len(strategy_data)} strategies")

    # ── 2-panel: Exp0 + Strategy In-Dist ─────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plot_exp0_panel(ax1, exp0_data)
    plot_strategy_panel(
        ax2,
        strategy_data,
        "in_dist.pearson_r",
        "In-Dist Pearson R",
        "B. Strategy Comparison (LegNet, AG S2 Oracle)",
    )
    fig.tight_layout(w_pad=3)
    fig.savefig(OUT / "talk_2panel_exp0_strategy.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "talk_2panel_exp0_strategy.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT / 'talk_2panel_exp0_strategy.png'}")

    # ── 2-panel: In-Dist + OOD Strategy ──────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plot_strategy_panel(
        ax1,
        strategy_data,
        "in_dist.pearson_r",
        "In-Dist Pearson R",
        "A. Strategy Scaling — In-Distribution",
    )
    plot_strategy_panel(
        ax2,
        strategy_data,
        "ood.pearson_r",
        "OOD Pearson R",
        "B. Strategy Scaling — Out-of-Distribution",
    )
    fig.tight_layout(w_pad=3)
    fig.savefig(OUT / "talk_2panel_strategy_id_ood.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "talk_2panel_strategy_id_ood.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT / 'talk_2panel_strategy_id_ood.png'}")

    # ── Single panel: Exp0 alone ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_exp0_panel(ax, exp0_data)
    ax.set_title(
        "K562 MPRA — Data Scaling Behavior\n(AG S2 Oracle Labels)", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(OUT / "talk_exp0_scaling.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "talk_exp0_scaling.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT / 'talk_exp0_scaling.png'}")

    # ── Single panel: Strategy In-Dist ───────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_strategy_panel(
        ax,
        strategy_data,
        "in_dist.pearson_r",
        "In-Dist Pearson R",
        "K562 — Reservoir Strategy Scaling (LegNet Student, AG S2 Oracle)",
    )
    fig.tight_layout()
    fig.savefig(OUT / "talk_strategy_indist.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "talk_strategy_indist.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT / 'talk_strategy_indist.png'}")

    # ── Single panel: Strategy OOD ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_strategy_panel(
        ax,
        strategy_data,
        "ood.pearson_r",
        "OOD Pearson R",
        "K562 — Reservoir Strategy Scaling, OOD (LegNet Student, AG S2 Oracle)",
    )
    fig.tight_layout()
    fig.savefig(OUT / "talk_strategy_ood.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "talk_strategy_ood.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT / 'talk_strategy_ood.png'}")

    # ── Single panel: SNV ref+alt ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_strategy_panel(
        ax,
        strategy_data,
        "snv_abs.pearson_r",
        "SNV (ref+alt) Pearson R",
        "K562 — Variant Effect Prediction (LegNet Student, AG S2 Oracle)",
    )
    fig.tight_layout()
    fig.savefig(OUT / "talk_strategy_snv.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "talk_strategy_snv.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT / 'talk_strategy_snv.png'}")

    print(f"\nAll figures in: {OUT}")


if __name__ == "__main__":
    main()
