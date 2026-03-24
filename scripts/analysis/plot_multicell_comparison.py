#!/usr/bin/env python3
"""Generate multi-cell-line comparison bar plots.

Compares model performance across K562, HepG2, and SK-N-SH for the
in-distribution (Reference) and SNV test sets. OOD is K562-only since
HepG2/SKNSH lack designed-sequence OOD test sets.

Usage:
    python scripts/analysis/plot_multicell_comparison.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "multicell_plots"

# ── Result directories per model per cell line ──────────────────────────────
# For each (model, cell), list candidate result dirs (first existing wins).
# S2 dirs listed before S1 so S2 results are preferred when available.

_RESULT_DIRS = {
    # Enformer
    ("Enformer", "k562"): [
        "outputs/enformer_k562_stage2",
        "outputs/enformer_k562_3seeds",
    ],
    ("Enformer", "hepg2"): [
        "outputs/enformer_hepg2_stage2",
        "outputs/enformer_hepg2_cached",
    ],
    ("Enformer", "sknsh"): [
        "outputs/enformer_sknsh_stage2",
        "outputs/enformer_sknsh_cached",
    ],
    # NTv3-post
    ("NTv3", "k562"): [
        "outputs/ntv3_post_k562_stage2",
        "outputs/ntv3_post_k562_cached",
    ],
    ("NTv3", "hepg2"): [
        "outputs/ntv3_post_hepg2_stage2",
        "outputs/ntv3_post_hepg2_cached",
    ],
    ("NTv3", "sknsh"): [
        "outputs/ntv3_post_sknsh_stage2",
        "outputs/ntv3_post_sknsh_cached",
    ],
    # Borzoi (K562 only, S1 only)
    ("Borzoi", "k562"): [
        "outputs/borzoi_k562_cached",
    ],
    # DREAM-RNN
    ("DREAM-RNN", "k562"): [
        "outputs/dream_rnn_k562_3seeds",
    ],
    ("DREAM-RNN", "hepg2"): [
        "outputs/dream_rnn_hepg2_3seeds",
    ],
    ("DREAM-RNN", "sknsh"): [
        "outputs/dream_rnn_sknsh_3seeds",
    ],
    # Malinois
    ("Malinois", "k562"): [
        "outputs/malinois_k562_3seeds",
    ],
    ("Malinois", "hepg2"): [
        "outputs/malinois_hepg2_3seeds",
    ],
    ("Malinois", "sknsh"): [
        "outputs/malinois_sknsh_3seeds",
    ],
    # AlphaGenome S1
    ("AG S1", "k562"): [
        "outputs/ag_hashfrag_oracle_cached",
    ],
    ("AG S1", "hepg2"): [
        "outputs/ag_hepg2_s1",
    ],
    ("AG S1", "sknsh"): [
        "outputs/ag_sknsh_s1",
    ],
    # AlphaGenome S2
    ("AG S2", "k562"): [
        "outputs/stage2_k562_full_train",
    ],
    ("AG S2", "hepg2"): [
        "outputs/ag_hepg2_s2",
    ],
    ("AG S2", "sknsh"): [
        "outputs/ag_sknsh_s2",
    ],
}

# Normalize test metric keys
_KEY_MAP = {
    "in_dist": "in_distribution",
    "in_distribution": "in_distribution",
    "snv_abs": "snv_abs",
    "snv_delta": "snv_delta",
    "ood": "ood",
}


def _normalize_keys(tm: dict) -> dict:
    out = {}
    for k, v in tm.items():
        canonical = _KEY_MAP.get(k, k)
        out[canonical] = v
        if k != canonical:
            out[k] = v
    return out


def load_results(base_dir: Path) -> list[dict]:
    """Load test_metrics from result.json or test_metrics.json files."""
    results = []
    seen = set()
    for pattern in ["result.json", "test_metrics.json"]:
        for p in sorted(base_dir.rglob(pattern)):
            try:
                d = json.loads(p.read_text())
                tm = d.get("test_metrics", d)
                tm = _normalize_keys(tm)
                if "in_distribution" in tm:
                    # Deduplicate by in_dist pearson (same seed may appear twice)
                    sig = round(tm["in_distribution"]["pearson_r"], 6)
                    if sig not in seen:
                        seen.add(sig)
                        results.append(tm)
            except Exception:
                continue
    return results


def collect_all_metrics() -> dict[tuple[str, str], list[dict]]:
    """Collect metrics for all (model, cell) combinations."""
    all_metrics: dict[tuple[str, str], list[dict]] = {}

    for (model, cell), dirs in _RESULT_DIRS.items():
        for d in dirs:
            p = REPO / d
            if p.exists():
                results = load_results(p)
                if results:
                    all_metrics[(model, cell)] = results
                    break

    return all_metrics


# ── Plotting ────────────────────────────────────────────────────────────────

CELL_LINES = ["k562", "hepg2", "sknsh"]
CELL_DISPLAY = {"k562": "K562", "hepg2": "HepG2", "sknsh": "SK-N-SH"}
CELL_COLORS = {"k562": "#3A86C8", "hepg2": "#E8602C", "sknsh": "#66BB6A"}

MODELS_ORDER = [
    ("DREAM-RNN", "#7B2D8E"),
    ("Malinois", "#B07CC6"),
    ("NTv3", "#E8602C"),
    ("Borzoi", "#DAA520"),
    ("Enformer", "#3A86C8"),
    ("AG S1", "#66BB6A"),
    ("AG S2", "#1B5E20"),
]


def plot_cross_cell_bars(
    all_metrics: dict,
    test_key: str = "in_distribution",
    metric_field: str = "pearson_r",
    title: str = "In-Distribution Pearson R",
    filename: str = "multicell_in_dist_pearson",
):
    """Bar plot: models on x-axis, grouped bars by cell line."""
    # Only include models that have at least one cell line result
    active_models = []
    for name, color in MODELS_ORDER:
        has_data = any(
            (name, cell) in all_metrics
            and any(
                test_key in m and metric_field in m.get(test_key, {})
                for m in all_metrics[(name, cell)]
            )
            for cell in CELL_LINES
        )
        if has_data:
            active_models.append((name, color))

    if not active_models:
        print(f"  Skipping {filename}: no data for {test_key}")
        return

    x = np.arange(len(active_models))
    n_cells = len(CELL_LINES)
    width = 0.8 / n_cells
    offsets = np.linspace(-(n_cells - 1) / 2 * width, (n_cells - 1) / 2 * width, n_cells)

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for ci, cell in enumerate(CELL_LINES):
        means = []
        stds = []
        for name, _ in active_models:
            key = (name, cell)
            if key in all_metrics:
                vals = [
                    m[test_key][metric_field]
                    for m in all_metrics[key]
                    if test_key in m and metric_field in m.get(test_key, {})
                ]
            else:
                vals = []
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if len(vals) > 1 else 0)

        bars = ax.bar(
            x + offsets[ci],
            means,
            width,
            yerr=stds if any(s > 0 for s in stds) else None,
            label=CELL_DISPLAY[cell],
            color=CELL_COLORS[cell],
            alpha=0.85,
            zorder=3,
            capsize=2,
        )

        for bar, val in zip(bars, means):
            if val > 0:
                fmt = f"{val:.3f}" if metric_field == "pearson_r" else f"{val:.2f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    fmt,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                    rotation=45,
                )

    ax.set_ylabel("Pearson R" if metric_field == "pearson_r" else "MSE", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in active_models], fontsize=11)
    if metric_field == "pearson_r":
        ax.set_ylim(0, 1.05)
    ax.set_title(f"Multi-Cell MPRA — {title}", fontsize=14)
    ax.legend(fontsize=11, loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{filename}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}.png / .pdf")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Collecting multi-cell metrics...")

    all_metrics = collect_all_metrics()

    # Print summary
    for (model, cell), results in sorted(all_metrics.items()):
        vals = [
            m["in_distribution"]["pearson_r"]
            for m in results
            if "in_distribution" in m and "pearson_r" in m.get("in_distribution", {})
        ]
        mean_r = np.mean(vals) if vals else float("nan")
        print(f"  {model:12s} {cell:6s}: {len(results)} seeds, in_dist={mean_r:.4f}")

    # Generate plots
    print("\nGenerating cross-cell bar plots...")

    plot_cross_cell_bars(
        all_metrics,
        test_key="in_distribution",
        metric_field="pearson_r",
        title="In-Distribution Pearson R",
        filename="multicell_in_dist_pearson",
    )

    plot_cross_cell_bars(
        all_metrics,
        test_key="snv_abs",
        metric_field="pearson_r",
        title="SNV Pearson R",
        filename="multicell_snv_pearson",
    )

    plot_cross_cell_bars(
        all_metrics,
        test_key="snv_delta",
        metric_field="pearson_r",
        title="SNV Effect Size (Delta) Pearson R",
        filename="multicell_snv_delta_pearson",
    )

    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
