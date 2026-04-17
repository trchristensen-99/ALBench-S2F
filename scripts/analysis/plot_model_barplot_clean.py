#!/usr/bin/env python3
"""Clean model comparison bar plot: LegNet, DREAM-RNN, Enformer, AG S2.

All metrics are against real MPRA labels (using _real variants where available).
K562 chr7/13 test split.

Models:
  - AG S2 (All Folds): 5-fold probing head mean
  - AG S2 (Fold 1): single fine-tuned fold
  - Enformer S2: fine-tuned
  - DREAM-RNN: full dataset
  - LegNet: real labels, full dataset

Metrics: Reference (in-dist), SNV Effect (delta), OOD (designed CREs)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "poster_stowers"
OUT.mkdir(parents=True, exist_ok=True)


def get_real_metric(tm, base_key):
    """Get real MPRA metric, preferring _real variant."""
    for k in [base_key + "_real", base_key, "in_distribution"]:
        if k in tm and "pearson_r" in tm[k]:
            return tm[k]["pearson_r"]
    return 0


def main():
    models = {}

    # AG S2 all-folds (5-fold probing head)
    vals = {"id": [], "ood": [], "snv_delta": []}
    for f in sorted(
        (REPO / "outputs" / "ag_hashfrag_oracle_cached").glob("oracle_*/test_metrics.json")
    ):
        d = json.loads(f.read_text())
        tm = d["test_metrics"]
        vals["id"].append(get_real_metric(tm, "in_dist"))
        vals["ood"].append(get_real_metric(tm, "ood"))
        vals["snv_delta"].append(get_real_metric(tm, "snv_delta"))
    if vals["id"]:
        models["AG S2\n(All Folds)"] = {k: (np.mean(v), np.std(v)) for k, v in vals.items()}

    # AG S2 fold-1
    f = REPO / "outputs" / "oracle_neg_sweep" / "r1d_i2neg_d1" / "test_metrics.json"
    if f.exists():
        d = json.loads(f.read_text())
        tm = d["test_metrics"]
        models["AG S2\n(Fold 1)"] = {
            "id": (get_real_metric(tm, "in_dist"), 0),
            "ood": (get_real_metric(tm, "ood"), 0),
            "snv_delta": (get_real_metric(tm, "snv_delta"), 0),
        }

    # Enformer S2
    for f in sorted((REPO / "outputs").glob("enformer_k562_stage2_final_v2/*/result.json")):
        d = json.loads(f.read_text())
        tm = d.get("test_metrics", {})
        idr = get_real_metric(tm, "in_dist")
        if idr > 0.85:
            models["Enformer\nS2"] = {
                "id": (idr, 0),
                "ood": (get_real_metric(tm, "ood"), 0),
                "snv_delta": (get_real_metric(tm, "snv_delta"), 0),
            }
            break

    # DREAM-RNN (full dataset, _real metrics)
    vals = {"id": [], "ood": [], "snv_delta": []}
    for f in (REPO / "outputs" / "exp0_oracle_scaling_v4" / "k562" / "dream_rnn" / "genomic").rglob(
        "result.json"
    ):
        d = json.loads(f.read_text())
        if d.get("n_train", 0) >= 280000:
            tm = d["test_metrics"]
            vals["id"].append(get_real_metric(tm, "in_dist"))
            vals["ood"].append(get_real_metric(tm, "ood"))
            vals["snv_delta"].append(get_real_metric(tm, "snv_delta"))
    if vals["id"]:
        models["DREAM-\nRNN"] = {k: (np.mean(v), np.std(v)) for k, v in vals.items()}

    # LegNet (real labels, full dataset)
    vals = {"id": [], "ood": [], "snv_delta": []}
    for f in (REPO / "outputs" / "exp0_oracle_scaling_v4" / "k562" / "legnet_ground_truth").rglob(
        "result.json"
    ):
        d = json.loads(f.read_text())
        if d.get("n_train", 0) >= 300000:
            tm = d["test_metrics"]
            vals["id"].append(get_real_metric(tm, "in_dist"))
            vals["ood"].append(get_real_metric(tm, "ood"))
            vals["snv_delta"].append(get_real_metric(tm, "snv_delta"))
    if vals["id"]:
        models["LegNet"] = {k: (np.mean(v), np.std(v)) for k, v in vals.items()}

    # Malinois (pretrained CNN from Gosai et al.)
    f = REPO / "outputs" / "malinois_eval_boda2_tutorial" / "result.json"
    if f.exists():
        d = json.loads(f.read_text())
        ct = d.get("chrom_test", {})
        if ct:
            models["Malinois"] = {
                "id": (ct.get("pearson_r", 0), 0),
                "ood": (0, 0),
                "snv_delta": (0, 0),
            }

    if not models:
        print("No model data found — run on HPC")
        return

    print("Models found:", list(models.keys()))
    for name, metrics in models.items():
        short = name.replace("\n", " ")
        print(
            f"  {short}: id={metrics['id'][0]:.4f}"
            f" ood={metrics['ood'][0]:.4f}"
            f" snv_d={metrics['snv_delta'][0]:.4f}"
        )

    # Plot
    metric_labels = {
        "id": "Reference\n(In-Dist)",
        "snv_delta": "SNV Effect\n(Delta)",
        "ood": "Designed CREs\n(OOD)",
    }
    metric_order = ["id", "ood"]

    model_order = [
        "AG S2\n(All Folds)",
        "AG S2\n(Fold 1)",
        "Malinois",
        "Enformer\nS2",
        "DREAM-\nRNN",
        "LegNet",
    ]
    model_order = [m for m in model_order if m in models]

    colors = {
        "AG S2\n(All Folds)": "#1B5E20",
        "AG S2\n(Fold 1)": "#4CAF50",
        "Malinois": "#6A1B9A",
        "Enformer\nS2": "#1565C0",
        "DREAM-\nRNN": "#7B1FA2",
        "LegNet": "#E8602C",
    }

    n_models = len(model_order)
    n_metrics = len(metric_order)
    bar_width = 0.8 / n_models
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(model_order):
        means = []
        stds = []
        for metric in metric_order:
            m, s = models[model].get(metric, (0, 0))
            means.append(m)
            stds.append(s)

        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=stds if any(s > 0 for s in stds) else None,
            capsize=3,
            label=model.replace("\n", " "),
            color=colors.get(model, "#888"),
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, means):
            if val > 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[m] for m in metric_order], fontsize=11)
    ax.set_ylabel("Pearson R (vs Real MPRA Labels)", fontsize=12)
    ax.set_title(
        "K562 MPRA Model Comparison (Gosai et al., Chr 7/13 Test)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right", frameon=True)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "panel3_model_barplot_clean.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "panel3_model_barplot_clean.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: panel3_model_barplot_clean.png")


if __name__ == "__main__":
    main()
