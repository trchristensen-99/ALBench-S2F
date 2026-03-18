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
    """Load best S2 metrics from the reference file.

    Uses the verified reference numbers from outputs/reference_metrics/
    to ensure consistency. Falls back to result.json files if reference
    is missing.
    """
    all_metrics: dict[str, list[dict]] = {}

    # Try reference file first (most reliable)
    ref_path = REPO / "outputs" / "reference_metrics" / "s2_best_results.json"
    if not ref_path.exists():
        ref_path = REPO / "outputs" / "results_backup_DO_NOT_DELETE" / "s2_best_results.json"

    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        for name, data in ref.get("models", {}).items():
            tm = {
                "in_distribution": {"pearson_r": data["in_dist"], "n": 40718},
                "snv_abs": {"pearson_r": data["snv_abs"], "n": 35226},
                "snv_delta": {"pearson_r": data["snv_delta"], "n": 35226},
                "ood": {"pearson_r": data["ood"], "n": 22862},
            }
            # Add MSE from reference if available
            for test_key, mse_key in [
                ("in_distribution", "mse_in_dist"),
                ("snv_abs", "mse_snv"),
                ("snv_delta", "mse_snvd"),
                ("ood", "mse_ood"),
            ]:
                if mse_key in data:
                    tm[test_key]["mse"] = data[mse_key]
            all_metrics[name] = [tm]
        print(f"  Loaded S2 reference from {ref_path.name}")

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
