#!/usr/bin/env python3
"""Generate Alan-style bar plots for MPRA benchmark comparison.

Creates bar plots matching the style from alphagenome_FT_MPRA repo:
- Consistent color palette across models
- Probing (S1) and Fine-tuned (S2) shown together for Enformer and AG
- Average across cell types (K562, HepG2, SKNSH) weighted by sample count
- Three test metrics: Reference (in-dist), SNV effect (delta), Synthetic seqs (OOD)
- Error bars showing variance across cell types and seeds
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "alan_style_plots"

# ── Alan's color palette ─────────────────────────────────────────────────────
# Matching alphagenome_FT_MPRA-main/scripts/plot_benchmark_results.py
MODEL_COLORS = {
    "Malinois": "#E8DCCF",  # lightest beige (baseline model color)
    "DREAM-RNN": "#8B9DAF",  # medium blue-gray
    "Enf. (Probing)": "#E7CDC2",  # light salmon/pink-beige
    "Enf. (Fine-tuned)": "#A65141",  # dark terracotta/rust red
    "AG (Probing)": "#80A0C7",  # light steel blue
    "AG (Fine-tuned)": "#394165",  # dark navy blue
}

MODEL_ORDER = [
    "Malinois",
    "DREAM-RNN",
    "Enf. (Probing)",
    "Enf. (Fine-tuned)",
    "AG (Probing)",
    "AG (Fine-tuned)",
]

# ── Result directories ───────────────────────────────────────────────────────
# Maps (model_label, cell) -> list of result dirs to search
RESULT_DIRS = {
    # Malinois (from-scratch baseline)
    ("Malinois", "k562"): ["outputs/malinois_k562_sweep/lr0.001_wd1e-3"],
    ("Malinois", "hepg2"): ["outputs/malinois_hepg2_3seeds"],
    ("Malinois", "sknsh"): ["outputs/malinois_sknsh_3seeds"],
    # DREAM-RNN
    ("DREAM-RNN", "k562"): ["outputs/dream_rnn_k562_3seeds"],
    ("DREAM-RNN", "hepg2"): ["outputs/dream_rnn_hepg2_3seeds"],
    ("DREAM-RNN", "sknsh"): ["outputs/dream_rnn_sknsh_3seeds"],
    # Enformer S1 (Probing)
    ("Enf. (Probing)", "k562"): [
        "outputs/enformer_k562_3seeds_v2",
        "outputs/enformer_k562_3seeds",
    ],
    ("Enf. (Probing)", "hepg2"): ["outputs/enformer_hepg2_cached"],
    ("Enf. (Probing)", "sknsh"): ["outputs/enformer_sknsh_cached"],
    # Enformer S2 (Fine-tuned)
    ("Enf. (Fine-tuned)", "k562"): ["outputs/enformer_k562_stage2_final"],
    ("Enf. (Fine-tuned)", "hepg2"): [
        "outputs/enformer_hepg2_stage2",
        "outputs/enformer_hepg2_s2_sweep",
    ],
    ("Enf. (Fine-tuned)", "sknsh"): [
        "outputs/enformer_sknsh_stage2",
        "outputs/enformer_sknsh_s2_sweep",
    ],
    # AG S1 (Probing) — use all-folds
    ("AG (Probing)", "k562"): [
        "outputs/ag_hashfrag_oracle_cached",
        "outputs/ag_all_folds_k562_s1_full",
    ],
    ("AG (Probing)", "hepg2"): ["outputs/ag_hashfrag_hepg2_cached"],
    ("AG (Probing)", "sknsh"): ["outputs/ag_hashfrag_sknsh_cached"],
    # AG S2 (Fine-tuned)
    ("AG (Fine-tuned)", "k562"): ["outputs/stage2_k562_full_train"],
    ("AG (Fine-tuned)", "hepg2"): ["outputs/ag_all_folds_hepg2_s2_from_s1"],
    ("AG (Fine-tuned)", "sknsh"): ["outputs/ag_all_folds_sknsh_s2_from_s1"],
}

# Approximate sample counts per cell for weighted averaging
CELL_WEIGHTS = {"k562": 40718, "hepg2": 40718, "sknsh": 40718}

METRICS = {
    "Reference": ("in_dist", "pearson_r"),
    "SNV Effect": ("snv_delta", "pearson_r"),
    "Synthetic Seqs": ("ood", "pearson_r"),
}


def _safe_float(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return float(x)


def collect_metrics():
    """Collect metrics for all models × cells × test sets.

    Returns:
        {model_label: {metric_name: {"mean": float, "std": float, "values": list}}}
    """
    results = {}

    for model in MODEL_ORDER:
        results[model] = {}
        for metric_name, (tm_key, field) in METRICS.items():
            # Collect values across cells (weighted)
            all_values = []
            cell_means = []
            cell_weights = []

            for cell in ["k562", "hepg2", "sknsh"]:
                dirs = RESULT_DIRS.get((model, cell), [])
                cell_values = []
                for d in dirs:
                    p = REPO / d
                    if not p.exists():
                        continue
                    # Try both result.json and test_metrics.json
                    for pattern in ["result.json", "test_metrics.json"]:
                        for f in p.rglob(pattern):
                            r = json.loads(f.read_text())
                            tm = r.get("test_metrics", r)
                            # Try multiple key formats
                            for key in [
                                tm_key,
                                "in_distribution" if tm_key == "in_dist" else tm_key,
                            ]:
                                if key in tm and field in tm[key]:
                                    val = _safe_float(tm[key][field])
                                    if val is not None and val > 0.01:
                                        cell_values.append(val)
                                    break
                    if cell_values:
                        break  # Found results in first dir

                if cell_values:
                    cell_mean = np.mean(cell_values)
                    cell_means.append(cell_mean)
                    cell_weights.append(CELL_WEIGHTS[cell])
                    all_values.extend(cell_values)

            if cell_means:
                # Weighted mean across cell types
                weights = np.array(cell_weights, dtype=float)
                weights /= weights.sum()
                weighted_mean = np.average(cell_means, weights=weights)
                # Std across all individual seed×cell values
                std = np.std(all_values) if len(all_values) > 1 else 0.0
                results[model][metric_name] = {
                    "mean": weighted_mean,
                    "std": std,
                    "values": all_values,
                    "n": len(all_values),
                }
            else:
                results[model][metric_name] = {
                    "mean": 0,
                    "std": 0,
                    "values": [],
                    "n": 0,
                }

    return results


def plot_alan_style(results, out_stem="mpra_benchmark"):
    """Create Alan-style grouped bar plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metric_names = list(METRICS.keys())
    x_pos = np.arange(len(metric_names))
    n_models = len(MODEL_ORDER)
    width = 0.12

    for i, model in enumerate(MODEL_ORDER):
        means = []
        stds = []
        for metric_name in metric_names:
            m = results[model].get(metric_name, {"mean": 0, "std": 0})
            means.append(m["mean"])
            stds.append(m["std"])

        offset = (i - n_models / 2) * width + width / 2
        bars = ax.bar(
            x_pos + offset,
            means,
            width,
            yerr=stds,
            capsize=2,
            label=model,
            color=MODEL_COLORS[model],
            alpha=0.9,
            edgecolor="black",
            linewidth=0.8,
            error_kw={"linewidth": 0.8},
        )

        # Value labels on top
        for bar, val in zip(bars, means):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                    color=MODEL_COLORS[model],
                    rotation=90,
                )

    ax.set_ylabel("Pearson Correlation", fontsize=12)
    ax.set_title("MPRA Model Comparison (HashFrag Split)", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim([0.15, 1.0])
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0, 1.02),
        frameon=False,
        fontsize=9,
        ncol=3,
    )

    plt.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        path = OUT / f"{out_stem}{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def main():
    print("Collecting metrics...")
    results = collect_metrics()

    # Print summary
    for model in MODEL_ORDER:
        for metric_name in METRICS:
            m = results[model].get(metric_name, {})
            mean = m.get("mean", 0)
            n = m.get("n", 0)
            print(f"  {model:25s} {metric_name:15s}: {mean:.4f} (n={n})")

    print("\nPlotting...")
    plot_alan_style(results)
    print("Done.")


if __name__ == "__main__":
    main()
