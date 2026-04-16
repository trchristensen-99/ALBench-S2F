#!/usr/bin/env python3
"""3-panel scaling comparison: AG vs LegNet, real vs oracle labels.

Panel A: In-distribution Pearson R
Panel B: OOD Pearson R
Panel C: SNV effect (delta) Pearson R

Each panel shows 4 curves:
  - AG S1 (real labels) — foundation model trained on real MPRA
  - AG S1 (oracle labels) — foundation model trained on own predictions (ceiling)
  - LegNet (real labels) — from-scratch model trained on real MPRA
  - LegNet (oracle labels) — from-scratch model trained on AG S2 pseudolabels

This shows the benefit of oracle distillation and the scaling gap between
foundation and from-scratch models.
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


def load_model_data(model_name, max_n=300000):
    """Load scaling data for a specific model."""
    base = REPO / "outputs" / "exp0_oracle_scaling_v4" / "k562" / model_name
    results_by_n = defaultdict(lambda: defaultdict(list))

    for rj in base.rglob("result.json"):
        try:
            d = json.loads(rj.read_text())
        except Exception:
            continue
        n = d.get("n_train", 0)
        if n > max_n:
            continue
        hp = json.dumps(d.get("hp_config", {}), sort_keys=True)
        val_r = d.get("val_pearson_r", 0)
        tm = d.get("test_metrics", {})
        results_by_n[n][hp].append((val_r, tm))

    data = {}
    for n in sorted(results_by_n):
        hp_map = results_by_n[n]
        best_hp = max(hp_map, key=lambda k: np.mean([v[0] for v in hp_map[k]]))
        data[n] = [v[1] for v in hp_map[best_hp]]
    return data


def extract_metric(data, metric_path):
    """Extract a specific metric from test_metrics dicts."""
    sizes, means, stds = [], [], []
    for n in sorted(data):
        vals = []
        keys = metric_path.split(".")
        for tm in data[n]:
            v = tm
            for k in keys:
                v = v.get(k, {}) if isinstance(v, dict) else None
                if v is None:
                    break
            if v is not None:
                vals.append(v)
        if vals:
            sizes.append(n)
            means.append(np.mean(vals))
            stds.append(np.std(vals) if len(vals) > 1 else 0)
    return sizes, np.array(means), np.array(stds)


def main():
    # Define the 4 curves
    curves = {
        "AG (real labels)": {
            "model": "alphagenome_k562_s1_ground_truth",
            "color": "#2980B9",
            "ls": "-",
            "lw": 2.5,
            "marker": "s",
        },
        "AG (oracle labels)": {
            "model": "alphagenome_k562_s1",
            "color": "#2980B9",
            "ls": "--",
            "lw": 1.8,
            "marker": "s",
        },
        "LegNet (real labels)": {
            "model": "legnet_ground_truth",
            "color": "#D4A017",
            "ls": "-",
            "lw": 2.5,
            "marker": "o",
        },
        "LegNet (oracle labels)": {
            "model": "legnet_oracle_ag_s2",
            "color": "#D4A017",
            "ls": "--",
            "lw": 1.8,
            "marker": "o",
        },
    }

    # Load all data
    model_data = {}
    for name, cfg in curves.items():
        model_data[name] = load_model_data(cfg["model"])
        print(f"  {name}: {len(model_data[name])} sizes")

    # 3-panel figure
    metrics = [
        ("in_dist.pearson_r", "In-Distribution Pearson R", "A"),
        ("ood.pearson_r", "OOD (Designed CREs) Pearson R", "B"),
        ("snv_delta.pearson_r", "SNV Effect (Δ) Pearson R", "C"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)

    for ax, (metric_path, ylabel, panel_label) in zip(axes, metrics):
        for name, cfg in curves.items():
            data = model_data[name]
            sizes, means, stds = extract_metric(data, metric_path)
            if not sizes:
                continue
            ci95 = 1.96 * stds

            ax.plot(
                sizes,
                means,
                color=cfg["color"],
                label=name,
                linewidth=cfg["lw"],
                linestyle=cfg["ls"],
                marker=cfg["marker"],
                markersize=5,
                zorder=5,
            )
            if stds.any():
                ax.fill_between(
                    sizes,
                    means - ci95,
                    means + ci95,
                    alpha=0.12,
                    color=cfg["color"],
                )

        ax.set_xscale("log")
        ax.set_xlabel("N Training Sequences", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{panel_label}. {ylabel}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right", frameon=True, facecolor="white")
        ax.grid(alpha=0.3, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Data Scaling: AlphaGenome vs LegNet on Real vs Oracle Labels (K562 MPRA)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "panel2_scaling_3panel.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "panel2_scaling_3panel.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: panel2_scaling_3panel.png")

    # Also make the original Exp0 plot capped at 500K
    from scripts.analysis.plot_peter_talk import load_exp0_scaling, plot_exp0_panel

    exp0_data = load_exp0_scaling()
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_exp0_panel(ax, exp0_data)
    ax.set_xlim(right=500000)
    fig.tight_layout()
    fig.savefig(OUT / "panel2_exp0_scaling_500k.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: panel2_exp0_scaling_500k.png")


if __name__ == "__main__":
    main()
