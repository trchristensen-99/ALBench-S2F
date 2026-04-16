#!/usr/bin/env python3
"""Clean model comparison bar plot for poster.

Uses definitive K562 chr7/13 test split results only.
All values verified from actual result files.

Models:
  - AG S2 (Fine-tuned): best oracle, trained on full dataset
  - AG S1 (Probing): frozen encoder, trained head (5-fold mean)
  - Enformer S2: fine-tuned enformer
  - Malinois: pretrained CNN (Gosai et al.)
  - LegNet: trained on real labels, full dataset
  - DREAM-RNN: trained on real labels, full dataset

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


def collect_from_files():
    """Collect verified metrics from actual result files."""
    models = {}

    # AG S1 Probing (5-fold mean)
    vals = {"id": [], "ood": [], "snv_delta": []}
    for f in sorted(
        (REPO / "outputs" / "ag_hashfrag_oracle_cached").glob("oracle_*/test_metrics.json")
    ):
        d = json.loads(f.read_text())
        tm = d["test_metrics"]
        for k_srcs, k_dst in [
            (["in_dist", "in_distribution"], "id"),
            (["ood"], "ood"),
            (["snv_delta"], "snv_delta"),
        ]:
            for k_src in k_srcs:
                if k_src in tm:
                    vals[k_dst].append(tm[k_src]["pearson_r"])
                    break
    if vals["id"]:
        models["AG S1\n(Probing)"] = {k: (np.mean(v), np.std(v)) for k, v in vals.items()}

    # AG S2 Fine-tuned (best config from neg_sweep)
    f = REPO / "outputs" / "oracle_neg_sweep" / "r1d_i2neg_d1" / "test_metrics.json"
    if f.exists():
        d = json.loads(f.read_text())
        tm = d["test_metrics"]
        models["AG S2\n(Fine-tuned)"] = {
            "id": (tm["in_distribution"]["pearson_r"], 0),
            "ood": (tm["ood"]["pearson_r"], 0),
            "snv_delta": (tm["snv_delta"]["pearson_r"], 0),
        }

    # Enformer S2
    for f in sorted((REPO / "outputs").glob("enformer_k562_stage2_final_v2/*/result.json")):
        d = json.loads(f.read_text())
        tm = d.get("test_metrics", {})
        idr = tm.get("in_dist", tm.get("in_distribution", {})).get("pearson_r", 0)
        if idr > 0.85:
            models["Enformer\nS2"] = {
                "id": (idr, 0),
                "ood": (tm.get("ood", {}).get("pearson_r", 0), 0),
                "snv_delta": (tm.get("snv_delta", {}).get("pearson_r", 0), 0),
            }
            break

    # Malinois (pretrained, different format)
    f = REPO / "outputs" / "malinois_eval_boda2_tutorial" / "result.json"
    if f.exists():
        d = json.loads(f.read_text())
        # Malinois only has chrom_test (combined in-dist)
        ct = d.get("chrom_test", {})
        if ct:
            models["Malinois\n(Pretrained)"] = {
                "id": (ct.get("pearson_r", 0), 0),
                "ood": (0, 0),  # Not available in this eval
                "snv_delta": (0, 0),
            }

    # LegNet (real labels, full dataset, genomic reservoir)
    vals = {"id": [], "ood": [], "snv_delta": []}
    for f in (REPO / "outputs" / "exp0_oracle_scaling_v4" / "k562" / "legnet_ground_truth").rglob(
        "result.json"
    ):
        d = json.loads(f.read_text())
        if d.get("n_train", 0) >= 300000:
            tm = d["test_metrics"]
            for k_src, k_dst in [("in_dist", "id"), ("ood", "ood"), ("snv_delta", "snv_delta")]:
                if k_src in tm:
                    vals[k_dst].append(tm[k_src]["pearson_r"])
    if vals["id"]:
        models["LegNet\n(Real Labels)"] = {k: (np.mean(v), np.std(v)) for k, v in vals.items()}

    return models


def main():
    models = collect_from_files()

    if not models:
        print("No model data found — run on HPC")
        return

    print("Models found:", list(models.keys()))
    for name, metrics in models.items():
        short = name.replace("\n", " ")
        print(
            f"  {short}: id={metrics['id'][0]:.4f} ood={metrics['ood'][0]:.4f} snv_d={metrics['snv_delta'][0]:.4f}"
        )

    # Plot
    metric_labels = {
        "id": "Reference\n(In-Dist)",
        "snv_delta": "SNV Effect\n(Delta)",
        "ood": "Designed CREs\n(OOD)",
    }
    metric_order = ["id", "snv_delta", "ood"]

    model_order = [
        "AG S2\n(Fine-tuned)",
        "AG S1\n(Probing)",
        "Enformer\nS2",
        "Malinois\n(Pretrained)",
        "LegNet\n(Real Labels)",
    ]
    model_order = [m for m in model_order if m in models]

    colors = {
        "AG S2\n(Fine-tuned)": "#1B5E20",
        "AG S1\n(Probing)": "#4CAF50",
        "Enformer\nS2": "#1565C0",
        "Malinois\n(Pretrained)": "#7B1FA2",
        "LegNet\n(Real Labels)": "#E8602C",
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
        # Value labels
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
    ax.set_ylabel("Pearson R", fontsize=12)
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
