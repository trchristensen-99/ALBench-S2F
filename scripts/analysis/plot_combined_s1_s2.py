#!/usr/bin/env python3
"""Generate bar plots and scatter grids for the combined S1+S2 comparison.

Uses the best available results (S2 where available, S1 otherwise):
  - DREAM-RNN: from-scratch (no S1/S2 distinction)
  - Malinois: from-scratch
  - NTv3-post: S2 (encoder finetuned)
  - Borzoi: S1 only (S2 didn't work — see memory)
  - Enformer: S2 (encoder finetuned)
  - AG fold 1: S2 (encoder finetuned)
  - AG all folds: S2 (encoder finetuned) — if available, else S1
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "exp0_plots"

_K562_TEST_SIZES = {
    "in_distribution": 40_718,
    "ood": 22_862,
    "snv_abs": 35_226,
    "snv_delta": 35_226,
}

# Key aliases for metric normalization
_KEY_ALIASES = {
    "in_distribution": "in_dist",
    "snv_abs": "snv_abs",
    "snv_delta": "snv_delta",
    "ood": "ood",
}


def _normalize_keys(tm: dict) -> dict:
    normalized = {}
    for k, v in tm.items():
        long_key = {v2: k2 for k2, v2 in _KEY_ALIASES.items()}.get(k, k)
        normalized[long_key] = v
        if k not in normalized:
            normalized[k] = v
    return normalized


def _load_metrics(results_dir: Path, json_name: str = "result.json") -> list[dict]:
    metrics_list = []
    for p in sorted(results_dir.rglob(json_name)):
        with open(p) as f:
            d = json.load(f)
        tm = d.get("test_metrics", d)
        tm = _normalize_keys(tm)
        if "in_distribution" in tm:
            metrics_list.append(tm)
    return metrics_list


def _extract(metrics_list: list[dict], test_key: str, field: str) -> list[float]:
    return [m[test_key][field] for m in metrics_list if test_key in m and field in m[test_key]]


# Models for the combined S1+S2 comparison
MODELS = [
    ("DREAM-RNN", "#7B2D8E"),
    ("Malinois", "#B07CC6"),
    ("NTv3", "#E8602C"),
    ("Borzoi", "#DAA520"),
    ("Enformer", "#3A86C8"),
    ("AG fold 1", "#66BB6A"),
    ("AG all folds", "#1B5E20"),
]


def load_all_metrics() -> dict[str, list[dict]]:
    """Load metrics for each model from the best available source."""
    all_metrics: dict[str, list[dict]] = {}

    # DREAM-RNN: from scaling v2 at f=1.0
    for fb_dir in [
        REPO / "outputs" / "dream_rnn_k562_with_preds",
        REPO / "outputs" / "exp0_k562_scaling_v2",
    ]:
        if fb_dir.exists():
            metrics = []
            for rj in fb_dir.rglob("result.json"):
                rd = json.loads(rj.read_text())
                if abs(rd.get("fraction", 1.0) - 1.0) < 0.01 or "fraction" not in rd:
                    tm = _normalize_keys(rd.get("test_metrics", rd))
                    if "in_distribution" in tm:
                        metrics.append(tm)
            if metrics:
                all_metrics["DREAM-RNN"] = metrics
                break

    # Malinois
    for d in [
        REPO / "outputs" / "malinois_k562_with_preds",
        REPO / "outputs" / "malinois_k562_sweep" / "lr0.001_wd1e-3",
    ]:
        m = _load_metrics(d)
        if m:
            all_metrics["Malinois"] = m
            break

    # NTv3 S2
    m = _load_metrics(REPO / "outputs" / "ntv3_k562_stage2_final")
    if m:
        all_metrics["NTv3"] = m
    else:
        # Fallback to S1
        m = _load_metrics(
            REPO / "outputs" / "foundation_grid_search" / "ntv3_post" / "lr0.0005_wd1e-6_do0.1"
        )
        if m:
            all_metrics["NTv3"] = m

    # Borzoi S1 (S2 didn't work)
    m = _load_metrics(REPO / "outputs" / "borzoi_k562_3seeds")
    if m:
        all_metrics["Borzoi"] = m

    # Enformer S2
    m = _load_metrics(REPO / "outputs" / "enformer_k562_stage2_final" / "elr1e-4_all")
    if m:
        all_metrics["Enformer"] = m
    else:
        m = _load_metrics(REPO / "outputs" / "enformer_k562_3seeds")
        if m:
            all_metrics["Enformer"] = m

    # AG fold 1 S2
    for d in [REPO / "outputs" / "stage2_k562_fold1"]:
        m = _load_metrics(d, "test_metrics.json")
        if m:
            all_metrics["AG fold 1"] = m
            break
    if "AG fold 1" not in all_metrics:
        m = _load_metrics(REPO / "outputs" / "ag_fold_1_k562_s1_full")
        if m:
            all_metrics["AG fold 1"] = m

    # AG all folds S2
    m = _load_metrics(REPO / "outputs" / "stage2_k562_full_train", "test_metrics.json")
    if m:
        all_metrics["AG all folds"] = m
    else:
        m = _load_metrics(REPO / "outputs" / "ag_all_folds_k562_s1_full")
        if m:
            all_metrics["AG all folds"] = m

    return all_metrics


def make_bar_plot(all_metrics: dict, metric: str, ylabel: str, title_suffix: str, filename: str):
    """Generate a bar plot for either Pearson R or MSE."""
    test_keys = [
        ("in_distribution", "Reference"),
        ("snv_abs", "SNV"),
        ("snv_delta", "SNV effect (delta)"),
        ("ood", "Synthetic design"),
    ]

    labels = []
    model_means = {name: [] for name, _ in MODELS}

    for key, display in test_keys:
        n_seqs = _K562_TEST_SIZES.get(key, "?")
        labels.append(f"{display}\n(n={n_seqs:,})")
        for name, _ in MODELS:
            vals = _extract(all_metrics.get(name, []), key, metric)
            model_means[name].append(np.mean(vals) if vals else 0)

    active = [(name, color) for name, color in MODELS if any(v > 0 for v in model_means[name])]
    if not active:
        print(f"  Skipping {filename}: no data")
        return

    x = np.arange(len(labels))
    width = 0.8 / len(active)
    offsets = np.linspace(
        -(len(active) - 1) / 2 * width, (len(active) - 1) / 2 * width, len(active)
    )

    fig, ax = plt.subplots(figsize=(14, 5.5))

    for i, (name, color) in enumerate(active):
        means = model_means[name]
        bars = ax.bar(x + offsets[i], means, width, label=name, color=color, zorder=3)
        for bar, val in zip(bars, means):
            if val > 0:
                fmt = f"{val:.3f}" if metric == "pearson_r" else f"{val:.2f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    fmt,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    rotation=30,
                )

    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    if metric == "pearson_r":
        ax.set_ylim(0, 1.0)
    ax.set_title(f"K562 MPRA — Best Available (S1+S2) {title_suffix}", fontsize=15)
    ax.legend(fontsize=9, loc="upper right", frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{filename}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}.png / .pdf")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading metrics...")

    all_metrics = load_all_metrics()
    for name, _ in MODELS:
        n = len(all_metrics.get(name, []))
        print(f"  {name}: {n} results")

    print("\nGenerating Pearson R bar plot...")
    make_bar_plot(all_metrics, "pearson_r", "Pearson R", "Pearson R", "k562_combined_bar")

    print("Generating MSE bar plot...")
    make_bar_plot(all_metrics, "mse", "MSE", "MSE", "k562_combined_mse_bar")

    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
