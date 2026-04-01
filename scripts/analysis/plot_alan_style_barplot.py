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

# Chr-split result directories
CHR_SPLIT_DIRS = {
    ("Malinois", "k562"): ["outputs/chr_split/k562/malinois"],
    ("Malinois", "hepg2"): ["outputs/chr_split/hepg2/malinois"],
    ("Malinois", "sknsh"): ["outputs/chr_split/sknsh/malinois"],
    ("DREAM-RNN", "k562"): ["outputs/chr_split/k562/dream_rnn"],
    ("DREAM-RNN", "hepg2"): ["outputs/chr_split/hepg2/dream_rnn"],
    ("DREAM-RNN", "sknsh"): ["outputs/chr_split/sknsh/dream_rnn"],
    ("Enf. (Probing)", "k562"): ["outputs/chr_split/k562/enformer_s1"],
    ("Enf. (Probing)", "hepg2"): ["outputs/chr_split/hepg2/enformer_s1"],
    ("Enf. (Probing)", "sknsh"): ["outputs/chr_split/sknsh/enformer_s1"],
    ("Enf. (Fine-tuned)", "k562"): [],  # No chr-split S2 yet
    ("Enf. (Fine-tuned)", "hepg2"): [],
    ("Enf. (Fine-tuned)", "sknsh"): [],
    ("AG (Probing)", "k562"): ["outputs/chr_split/k562/ag_all_folds_s1"],
    ("AG (Probing)", "hepg2"): ["outputs/chr_split/hepg2/ag_all_folds_s1"],
    ("AG (Probing)", "sknsh"): ["outputs/chr_split/sknsh/ag_all_folds_s1"],
    ("AG (Fine-tuned)", "k562"): ["outputs/chr_split/k562/ag_all_folds_s2"],
    ("AG (Fine-tuned)", "hepg2"): ["outputs/chr_split/hepg2/ag_all_folds_s2"],
    ("AG (Fine-tuned)", "sknsh"): ["outputs/chr_split/sknsh/ag_all_folds_s2"],
}

# HashFrag result directories
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


def collect_metrics(split_type="hashfrag"):
    """Collect metrics for all models × cells × test sets.

    Args:
        split_type: "hashfrag" or "chr_split"

    Returns:
        {model_label: {metric_name: {"mean": float, "std": float, "values": list}}}
    """
    dir_map = CHR_SPLIT_DIRS if split_type == "chr_split" else RESULT_DIRS
    results = {}

    for model in MODEL_ORDER:
        results[model] = {}
        for metric_name, (tm_key, field) in METRICS.items():
            # OOD (Synthetic Seqs) only has K562 labels — skip other cells
            # to avoid label mismatch (OOD sequences were designed/measured in K562 only)
            if tm_key == "ood":
                cells_to_use = ["k562"]
            else:
                cells_to_use = ["k562", "hepg2", "sknsh"]

            # Collect values across cells (weighted)
            all_values = []
            cell_means = []
            cell_weights = []

            for cell in cells_to_use:
                dirs = dir_map.get((model, cell), [])
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
                                    if val is not None:
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


def plot_alan_style(results, out_stem="mpra_benchmark", title="MPRA Model Comparison"):
    """Create Alan-style grouped bar plot matching reference lenti_starr_res.png."""
    # Filter to models that have at least one non-zero metric
    active_models = [
        m for m in MODEL_ORDER if any(results[m].get(mn, {}).get("n", 0) > 0 for mn in METRICS)
    ]
    if not active_models:
        print(f"  No data for {out_stem}, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))

    metric_names = list(METRICS.keys())
    x_pos = np.arange(len(metric_names))
    n_models = len(active_models)
    width = 0.13

    for i, model in enumerate(active_models):
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
            edgecolor="black",
            linewidth=0.5,
            error_kw={"linewidth": 0.8, "capthick": 0.8},
        )

        # Value labels on top — italic, color-matched, like reference
        for bar, val, std_val in zip(bars, means, stds):
            if val > 0.01:
                y_top = val + std_val + 0.008
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_top,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    fontstyle="italic",
                    color=MODEL_COLORS[model],
                )

    ax.set_ylabel("Pearson Correlation", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="semibold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=9,
        ncol=1,
    )

    plt.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        path = OUT / f"{out_stem}{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def main():
    for split_type, stem, title in [
        (
            "chr_split",
            "mpra_benchmark_chr_split",
            "lentiMPRA",
        ),
    ]:
        print(f"\n=== {split_type} ===")
        results = collect_metrics(split_type=split_type)

        for model in MODEL_ORDER:
            for metric_name in METRICS:
                m = results[model].get(metric_name, {})
                mean = m.get("mean", 0)
                std = m.get("std", 0)
                n = m.get("n", 0)
                vals = m.get("values", [])
                if n > 0:
                    print(
                        f"  {model:25s} {metric_name:15s}: "
                        f"{mean:.4f} ± {std:.4f} (n={n}, vals={[round(v, 4) for v in vals]})"
                    )

        print(f"\nPlotting {stem}...")
        plot_alan_style(results, out_stem=stem, title=title)

    print("\nDone.")


if __name__ == "__main__":
    main()
