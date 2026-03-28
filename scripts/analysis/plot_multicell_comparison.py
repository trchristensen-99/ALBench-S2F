#!/usr/bin/env python3
"""Generate multi-cell-line comparison bar plots.

Produces three types of plots:

Type 1 — Cross-cell comparison:
    Models on x-axis, grouped bars by cell line (K562/HepG2/SK-N-SH).
    One plot per metric. Both S1-only and combined S1+S2 variants.

Type 2 — Per-cell "original style":
    For EACH cell line, x-axis = 4 test conditions (Reference, SNV,
    SNV delta, OOD), grouped bars for each model. Both S1-only and
    combined S1+S2 variants.

Type 3 — S1 cross-cell:
    Same as Type 1 but restricted to S1 (frozen encoder) models.

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

# ── Key normalization ───────────────────────────────────────────────────────

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


# ── Result loading ──────────────────────────────────────────────────────────


def load_results(base_dir: Path) -> list[dict]:
    """Load test_metrics from result.json or test_metrics.json files.

    Handles both formats:
    - result.json: test_metrics nested under top-level dict
    - test_metrics.json: test_metrics may be at top level or nested
    """
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


# ── Result directory registry ───────────────────────────────────────────────
# For each (model, cell), list candidate dirs (first existing with data wins).
# S2 dirs before S1 so S2 is preferred when available.

_COMBINED_RESULT_DIRS = {
    # DREAM-RNN (from-scratch, all 3 cells)
    ("DREAM-RNN", "k562"): [
        "outputs/dream_rnn_k562_with_preds",
        "outputs/dream_rnn_k562_3seeds",
    ],
    ("DREAM-RNN", "hepg2"): [
        "outputs/dream_rnn_hepg2_3seeds",
    ],
    ("DREAM-RNN", "sknsh"): [
        "outputs/dream_rnn_sknsh_3seeds",
    ],
    # DREAM-CNN (from-scratch, all 3 cells)
    ("DREAM-CNN", "k562"): [
        "outputs/dream_cnn_k562_real",
    ],
    ("DREAM-CNN", "hepg2"): [
        "outputs/dream_cnn_hepg2_real",
    ],
    ("DREAM-CNN", "sknsh"): [
        "outputs/dream_cnn_sknsh_real",
    ],
    # Malinois (from-scratch, all 3 cells)
    ("Malinois", "k562"): [
        "outputs/malinois_k562_with_preds",
        "outputs/malinois_k562_3seeds",
    ],
    ("Malinois", "hepg2"): [
        "outputs/malinois_hepg2_3seeds",
    ],
    ("Malinois", "sknsh"): [
        "outputs/malinois_sknsh_3seeds",
    ],
    # NTv3 (S2 where available, S1 fallback)
    ("NTv3", "k562"): [
        "outputs/ntv3_post_k562_stage2",
        "outputs/ntv3_k562_stage2_final",
        "outputs/ntv3_post_k562_3seeds",
    ],
    ("NTv3", "hepg2"): [
        "outputs/ntv3_post_hepg2_stage2",
        "outputs/ntv3_post_hepg2_cached",
    ],
    ("NTv3", "sknsh"): [
        "outputs/ntv3_post_sknsh_stage2",
        "outputs/ntv3_post_sknsh_cached",
    ],
    # Borzoi (S1 only — S2 doesn't work for MPRA)
    ("Borzoi", "k562"): [
        "outputs/borzoi_k562_3seeds",
    ],
    ("Borzoi", "hepg2"): [
        "outputs/borzoi_hepg2_cached",
    ],
    ("Borzoi", "sknsh"): [
        "outputs/borzoi_sknsh_cached",
    ],
    # Enformer (S2 K562, S1 HepG2/SKNSH since S1 > S2 for those cells)
    ("Enformer", "k562"): [
        "outputs/enformer_k562_stage2_final",
        "outputs/enformer_k562_stage2_final_v2",
        "outputs/enformer_k562_3seeds",
    ],
    ("Enformer", "hepg2"): [
        "outputs/enformer_hepg2_cached",  # S1 is better (0.858 vs 0.850)
    ],
    ("Enformer", "sknsh"): [
        "outputs/enformer_sknsh_cached",  # S1 is better (0.857 vs 0.853)
    ],
    # AG fold 1 (S2 preferred, S1 fallback, all 3 cells)
    ("AG fold 1", "k562"): [
        "outputs/stage2_k562_fold1",
        "outputs/ag_fold_1_k562_s1_full",
    ],
    ("AG fold 1", "hepg2"): [
        "outputs/ag_fold_1_hepg2_s2",
        "outputs/ag_fold_1_hepg2_s1",
    ],
    ("AG fold 1", "sknsh"): [
        "outputs/ag_fold_1_sknsh_s2",
        "outputs/ag_fold_1_sknsh_s1",
    ],
    # AG all folds S1 (all 3 cells)
    ("AG all folds", "k562"): [
        "outputs/ag_all_folds_k562_s1_full",
        "outputs/ag_hashfrag_oracle_cached",
    ],
    ("AG all folds", "hepg2"): [
        "outputs/ag_hashfrag_hepg2_cached",
    ],
    ("AG all folds", "sknsh"): [
        "outputs/ag_hashfrag_sknsh_cached",
    ],
    # AG all folds S2 (K562 for now, HepG2/SKNSH when available)
    ("AG S2", "k562"): [
        "outputs/stage2_k562_full_train",
    ],
    ("AG S2", "hepg2"): [
        "outputs/ag_hepg2_stage2",
        "outputs/ag_s2_hepg2",
    ],
    ("AG S2", "sknsh"): [
        "outputs/ag_sknsh_stage2",
        "outputs/ag_s2_sknsh",
    ],
}

# S1-only directories (frozen encoder + head, plus from-scratch baselines)
_S1_RESULT_DIRS = {
    # From-scratch baselines (same results regardless of S1/S2 distinction)
    ("DREAM-RNN", "k562"): [
        "outputs/dream_rnn_k562_with_preds",
        "outputs/dream_rnn_k562_3seeds",
    ],
    ("DREAM-RNN", "hepg2"): [
        "outputs/dream_rnn_hepg2_3seeds",
    ],
    ("DREAM-RNN", "sknsh"): [
        "outputs/dream_rnn_sknsh_3seeds",
    ],
    ("DREAM-CNN", "k562"): [
        "outputs/dream_cnn_k562_real",
    ],
    ("DREAM-CNN", "hepg2"): [
        "outputs/dream_cnn_hepg2_real",
    ],
    ("DREAM-CNN", "sknsh"): [
        "outputs/dream_cnn_sknsh_real",
    ],
    ("Malinois", "k562"): [
        "outputs/malinois_k562_with_preds",
        "outputs/malinois_k562_3seeds",
    ],
    ("Malinois", "hepg2"): ["outputs/malinois_hepg2_3seeds"],
    ("Malinois", "sknsh"): ["outputs/malinois_sknsh_3seeds"],
    # Foundation models S1 only
    ("NTv3 S1", "k562"): [
        "outputs/ntv3_post_k562_3seeds",
        "outputs/ntv3_k562_3seeds",
    ],
    ("NTv3 S1", "hepg2"): [
        "outputs/ntv3_post_hepg2_cached",
    ],
    ("NTv3 S1", "sknsh"): [
        "outputs/ntv3_post_sknsh_cached",
    ],
    ("Borzoi S1", "k562"): [
        "outputs/borzoi_k562_3seeds",
    ],
    ("Borzoi S1", "hepg2"): [
        "outputs/borzoi_hepg2_cached",
    ],
    ("Borzoi S1", "sknsh"): [
        "outputs/borzoi_sknsh_cached",
    ],
    ("Enformer S1", "k562"): [
        "outputs/enformer_k562_3seeds",
    ],
    ("Enformer S1", "hepg2"): [
        "outputs/enformer_hepg2_cached",
    ],
    ("Enformer S1", "sknsh"): [
        "outputs/enformer_sknsh_cached",
    ],
    ("AG fold 1 S1", "k562"): [
        "outputs/ag_fold_1_k562_s1_full",
    ],
    ("AG fold 1 S1", "hepg2"): [
        "outputs/ag_fold_1_hepg2_s1",
    ],
    ("AG fold 1 S1", "sknsh"): [
        "outputs/ag_fold_1_sknsh_s1",
    ],
    ("AG S1", "k562"): [
        "outputs/ag_all_folds_k562_s1_full",
        "outputs/ag_hashfrag_oracle_cached",
    ],
    ("AG S1", "hepg2"): [
        "outputs/ag_hashfrag_hepg2_cached",
    ],
    ("AG S1", "sknsh"): [
        "outputs/ag_hashfrag_sknsh_cached",
    ],
}


def collect_metrics(result_dirs: dict) -> dict[tuple[str, str], list[dict]]:
    """Collect metrics for all (model, cell) combinations from given dirs."""
    all_metrics: dict[tuple[str, str], list[dict]] = {}

    for (model, cell), dirs in result_dirs.items():
        for d in dirs:
            p = REPO / d
            if p.exists():
                results = load_results(p)
                if results:
                    all_metrics[(model, cell)] = results
                    break

    return all_metrics


# ── Constants ───────────────────────────────────────────────────────────────

CELL_LINES = ["k562", "hepg2", "sknsh"]
CELL_DISPLAY = {"k562": "K562", "hepg2": "HepG2", "sknsh": "SK-N-SH"}
CELL_COLORS = {"k562": "#3A86C8", "hepg2": "#E8602C", "sknsh": "#66BB6A"}

# Model orderings and colors
MODELS_COMBINED = [
    ("DREAM-RNN", "#7B2D8E"),
    ("DREAM-CNN", "#9B59B6"),
    ("Malinois", "#B07CC6"),
    ("NTv3", "#E8602C"),
    ("Borzoi", "#DAA520"),
    ("Enformer", "#3A86C8"),
    ("AG fold 1", "#A5D6A7"),
    ("AG all folds", "#66BB6A"),
    ("AG S2", "#1B5E20"),
]

S1_MODELS = [
    ("DREAM-RNN", "#7B2D8E"),
    ("DREAM-CNN", "#9B59B6"),
    ("Malinois", "#B07CC6"),
    ("NTv3 S1", "#E8602C"),
    ("Borzoi S1", "#DAA520"),
    ("Enformer S1", "#3A86C8"),
    ("AG fold 1 S1", "#A5D6A7"),
    ("AG S1", "#66BB6A"),
]

TEST_CONDITIONS = [
    ("in_distribution", "Reference (n=40,718)"),
    ("snv_abs", "SNV (n=35,226)"),
    ("snv_delta", "SNV delta (n=35,226)"),
    ("ood", "Synthetic design (n=22,862)"),
]


# ── Type 1/3: Cross-cell bar plots ─────────────────────────────────────────


def plot_cross_cell_bars(
    all_metrics: dict,
    test_key: str = "in_distribution",
    metric_field: str = "pearson_r",
    title: str = "In-Distribution Pearson R",
    filename: str = "multicell_in_dist_pearson",
    models_order: list | None = None,
):
    """Bar plot: models on x-axis, grouped bars by cell line."""
    if models_order is None:
        models_order = MODELS_COMBINED
    # Only include models that have at least one cell line result
    active_models = []
    for name, color in models_order:
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
            # yerr=stds if any(s > 0 for s in stds) else None,
            label=CELL_DISPLAY[cell],
            color=CELL_COLORS[cell],
            alpha=0.85,
            zorder=3,
            # capsize=2,
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
    ax.set_title(f"Multi-Cell MPRA -- {title}", fontsize=14)
    ax.legend(fontsize=11, loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{filename}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}.png / .pdf")


# ── Type 2: Per-cell "original style" bar plots ────────────────────────────


def plot_per_cell_bars(
    all_metrics: dict,
    cell: str,
    models_order: list,
    title_tag: str,
    filename: str,
    metric_field: str = "pearson_r",
):
    """Bar plot for a single cell line: test conditions on x-axis, models as bars.

    Skips test conditions that have no data for any model.

    Parameters
    ----------
    metric_field : str
        Which metric to plot (``"pearson_r"`` or ``"mse"``).
    """
    cell_display = CELL_DISPLAY[cell]
    is_mse = metric_field == "mse"
    metric_label = "MSE" if is_mse else "Pearson R"

    # Determine which test conditions have data
    active_conditions = []
    for test_key, label in TEST_CONDITIONS:
        has_data = False
        for name, _ in models_order:
            key = (name, cell)
            if key in all_metrics:
                vals = [
                    m[test_key][metric_field]
                    for m in all_metrics[key]
                    if test_key in m and metric_field in m.get(test_key, {})
                ]
                if vals:
                    has_data = True
                    break
        if has_data:
            active_conditions.append((test_key, label))

    if not active_conditions:
        print(f"  Skipping {filename}: no data for {cell_display}")
        return

    # Determine which models have data for this cell
    active_models = []
    for name, color in models_order:
        key = (name, cell)
        if key in all_metrics:
            has_any = any(
                "in_distribution" in m and metric_field in m.get("in_distribution", {})
                for m in all_metrics[key]
            )
            if has_any:
                active_models.append((name, color))

    if not active_models:
        print(f"  Skipping {filename}: no models with data for {cell_display}")
        return

    x = np.arange(len(active_conditions))
    n_models = len(active_models)
    width = 0.8 / n_models
    offsets = np.linspace(-(n_models - 1) / 2 * width, (n_models - 1) / 2 * width, n_models)

    fig, ax = plt.subplots(figsize=(14, 5.5))

    for i, (name, color) in enumerate(active_models):
        key = (name, cell)
        means = []
        stds = []
        for test_key, _ in active_conditions:
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
            x + offsets[i],
            means,
            width,
            # yerr=stds if any(s > 0 for s in stds) else None,
            label=name,
            color=color,
            zorder=3,
            # capsize=2,
        )

        for bar, val in zip(bars, means):
            if val > 0:
                fmt = f"{val:.4f}" if is_mse else f"{val:.3f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    fmt,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                    rotation=30,
                )

    ax.set_ylabel(metric_label, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in active_conditions], fontsize=11)
    if not is_mse:
        ax.set_ylim(0, 1.0)
    ax.set_title(f"{cell_display} MPRA -- {title_tag} -- {metric_label}", fontsize=14)
    ax.legend(
        fontsize=9,
        loc="upper right",
        frameon=False,
        ncol=2 if n_models > 4 else 1,
    )
    ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{filename}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}.png / .pdf")


# ── Helpers ─────────────────────────────────────────────────────────────────


def _print_summary(label: str, metrics: dict):
    print(f"\n  {label}:")
    for (model, cell), results in sorted(metrics.items()):
        vals = [
            m["in_distribution"]["pearson_r"]
            for m in results
            if "in_distribution" in m and "pearson_r" in m.get("in_distribution", {})
        ]
        mean_r = np.mean(vals) if vals else float("nan")
        print(f"    {model:18s} {cell:6s}: {len(results):2d} seeds, in_dist={mean_r:.4f}")


def _gen_cross_cell_plots(metrics: dict, models_order: list, prefix: str, title_tag: str):
    """Generate cross-cell bar plots (Type 1 / Type 3)."""
    test_configs = [
        ("in_distribution", "pearson_r", "In-Distribution Pearson R"),
        ("snv_abs", "pearson_r", "SNV Pearson R"),
        ("snv_delta", "pearson_r", "SNV Effect Size (Delta) Pearson R"),
        ("in_distribution", "mse", "In-Distribution MSE"),
    ]
    for test_key, metric, title in test_configs:
        suffix = test_key
        if metric == "mse":
            suffix += "_mse"
        plot_cross_cell_bars(
            metrics,
            test_key=test_key,
            metric_field=metric,
            title=f"{title_tag} -- {title}",
            filename=f"{prefix}_{suffix}",
            models_order=models_order,
        )


def _gen_per_cell_plots(metrics: dict, models_order: list, prefix: str, title_tag: str):
    """Generate per-cell bar plots (Type 2) for both Pearson R and MSE."""
    for cell in CELL_LINES:
        # Pearson R
        plot_per_cell_bars(
            metrics,
            cell=cell,
            models_order=models_order,
            title_tag=title_tag,
            filename=f"{prefix}_{cell}",
            metric_field="pearson_r",
        )
        # MSE
        plot_per_cell_bars(
            metrics,
            cell=cell,
            models_order=models_order,
            title_tag=title_tag,
            filename=f"{prefix}_{cell}_mse",
            metric_field="mse",
        )


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Collecting multi-cell metrics...")

    # S1-only metrics
    s1_metrics = collect_metrics(_S1_RESULT_DIRS)
    _print_summary("S1 (frozen encoder + head)", s1_metrics)

    # Combined S1+S2 metrics (best available)
    combined_metrics = collect_metrics(_COMBINED_RESULT_DIRS)
    _print_summary("Combined S1+S2 (best available)", combined_metrics)

    # ── Type 1: Cross-cell combined S1+S2 ──────────────────────────────
    print("\n--- Type 1: Cross-cell combined S1+S2 bar plots ---")
    _gen_cross_cell_plots(combined_metrics, MODELS_COMBINED, "multicell", "Best Available (S1+S2)")

    # ── Type 2: Per-cell "original style" ──────────────────────────────
    print("\n--- Type 2: Per-cell combined S1+S2 bar plots ---")
    _gen_per_cell_plots(
        combined_metrics,
        MODELS_COMBINED,
        "percell_combined",
        "Best Available (S1+S2)",
    )

    print("\n--- Type 2: Per-cell S1-only bar plots ---")
    _gen_per_cell_plots(
        s1_metrics,
        S1_MODELS,
        "percell_s1",
        "S1 (Frozen Encoder)",
    )

    # ── Type 3: S1 cross-cell ──────────────────────────────────────────
    print("\n--- Type 3: S1 cross-cell bar plots ---")
    _gen_cross_cell_plots(s1_metrics, S1_MODELS, "s1_multicell", "S1 (Frozen Encoder)")

    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
