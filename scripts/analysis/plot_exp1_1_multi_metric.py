#!/usr/bin/env python3
"""Multi-metric cross-config analysis for Experiment 1.1.

Improvements over plot_exp1_1_cross_config.py:
- Uses BEST AVAILABLE n_train per config (not fixed n=50k)
- Shows multiple metrics (OOD, in_dist, SNV_abs, SNV_delta)
- Separate plots per task (K562, yeast)
- Includes configs with sparse large-N data by falling back to smaller N

Run:
    python scripts/analysis/plot_exp1_1_multi_metric.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "exp1_1_analysis"

CATEGORIES = {
    "Baseline": ["random", "genomic", "gc_matched", "dinuc_shuffle"],
    "PRM": ["prm_1pct", "prm_5pct", "prm_10pct", "prm_20pct", "prm_50pct", "prm_uniform_1_10"],
    "Recombination": ["recombination_uniform", "recombination_2pt"],
    "Augmentation": ["evoaug_structural", "evoaug_heavy"],
    "Motif": ["motif_planted", "motif_grammar", "motif_grammar_tight"],
    "ISE (Oracle)": ["ise_maximize", "ise_diverse_targets", "ise_target_high"],
    "ISE (Dream10%)": [
        "ise_maximize_dream10",
        "ise_diverse_targets_dream10",
        "ise_target_high_dream10",
    ],
    "ISE (AG10%)": [
        "ise_maximize_ag10",
        "ise_diverse_targets_ag10",
        "ise_target_high_ag10",
    ],
    "Variant": ["snv"],
}

SHORT_NAMES = {
    "alphagenome_k562_s1_ag": "AG oracle\nAG student",
    "alphagenome_k562_s1_dream_rnn": "DREAM oracle\nAG student",
    "dream_rnn_dream_rnn": "DREAM oracle\nDREAM student",
    "dream_rnn_ag": "AG oracle\nDREAM student",
    "alphagenome_yeast_s1_ag": "AG oracle\nAG student",
    "alphagenome_yeast_s1_dream_rnn": "DREAM oracle\nAG student",
    "alphagenome_yeast_s2_ag": "AG oracle\nAG-S2 student",
}

METRICS = ["ood", "in_dist", "snv_abs", "snv_delta"]
METRIC_LABELS = {
    "ood": "OOD",
    "in_dist": "In-Distribution",
    "snv_abs": "SNV Absolute",
    "snv_delta": "SNV Delta",
}


def load_all_results():
    all_data = {}
    for task in ["k562", "yeast"]:
        task_dir = REPO / "outputs" / "exp1_1" / task
        if not task_dir.exists():
            continue
        for config_dir in sorted(task_dir.iterdir()):
            if not config_dir.is_dir() or config_dir.name == "figures":
                continue
            key = f"{task}/{config_dir.name}"
            by_res_n = defaultdict(list)
            for rj in config_dir.rglob("result.json"):
                try:
                    r = json.loads(rj.read_text())
                    res = r.get("reservoir", rj.relative_to(config_dir).parts[0])
                    n = r["n_train"]
                    hp = json.dumps(r.get("hp_config", {}), sort_keys=True)
                    val_r = r.get("val_pearson_r", 0)
                    metrics = {}
                    for ts in METRICS:
                        v = r["test_metrics"].get(ts, {}).get("pearson_r")
                        if v is not None:
                            metrics[ts] = v
                    by_res_n[(res, n)].append({"hp": hp, "val_r": val_r, **metrics})
                except Exception:
                    pass
            if by_res_n:
                all_data[key] = by_res_n
    return all_data


def get_best_hp_metric(data, target_n, metric):
    """Get best-HP score per reservoir for a given metric at target_n."""
    scores = {}
    for (res, n), results in data.items():
        if n != target_n:
            continue
        vals = [r[metric] for r in results if metric in r]
        if len(vals) < 2:
            continue
        by_hp = defaultdict(list)
        for r in results:
            by_hp[r["hp"]].append(r)
        best_hp_results = max(
            by_hp.values(), key=lambda runs: np.mean([r.get("val_r", 0) for r in runs])
        )
        best_vals = [r[metric] for r in best_hp_results if metric in r]
        if best_vals:
            scores[res] = np.mean(best_vals)
    return scores


def find_best_n(data, metric, min_reservoirs=5):
    """Find the largest n_train with at least min_reservoirs having scores."""
    sizes = sorted(set(n for (_, n) in data.keys()), reverse=True)
    for n in sizes:
        scores = get_best_hp_metric(data, n, metric)
        if len(scores) >= min_reservoirs:
            return n
    return sizes[-1] if sizes else None


def plot_multi_metric_category_heatmap(task_data, task_name, out_dir):
    """Category heatmap showing multiple metrics side by side."""
    configs = sorted(task_data.keys())
    cats = list(CATEGORIES.keys())

    # Use 2 key metrics: OOD and in_dist
    key_metrics = ["ood", "in_dist", "snv_abs", "snv_delta"]

    for metric in key_metrics:
        fig, ax = plt.subplots(figsize=(max(6, len(configs) * 1.8), 5))
        matrix = np.full((len(cats), len(configs)), np.nan)
        n_used = {}

        for j, config_key in enumerate(configs):
            data = task_data[config_key]
            best_n = find_best_n(data, metric, min_reservoirs=3)
            if best_n is None:
                continue
            n_used[config_key] = best_n

            for i, (cat, reservoirs) in enumerate(CATEGORIES.items()):
                scores = get_best_hp_metric(data, best_n, metric)
                cat_vals = [scores[r] for r in reservoirs if r in scores]
                if cat_vals:
                    matrix[i, j] = np.mean(cat_vals)

        valid = matrix[~np.isnan(matrix)]
        if len(valid) == 0:
            plt.close()
            continue

        im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if not np.isnan(val):
                    col_vals = matrix[:, j][~np.isnan(matrix[:, j])]
                    mid = np.mean(col_vals) if len(col_vals) > 0 else 0
                    color = "white" if val > mid else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color)
                else:
                    ax.text(j, i, "--", ha="center", va="center", fontsize=8, color="#999")

        labels = []
        for c in configs:
            cfg = c.split("/")[1]
            n = n_used.get(c)
            short = SHORT_NAMES.get(cfg, cfg)
            label = f"{short}\n(n={n // 1000}k)" if n else short
            labels.append(label)

        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels(cats, fontsize=9)
        ax.set_title(
            f"{task_name} — {METRIC_LABELS[metric]} by Category\n(best available N per config)",
            fontsize=12,
            fontweight="bold",
        )
        fig.colorbar(im, ax=ax, label=METRIC_LABELS[metric], shrink=0.8)
        fig.tight_layout()

        out = out_dir / f"category_{metric}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.name}")


def plot_multi_metric_ranking_correlation(task_data, task_name, out_dir):
    """Ranking correlation matrix for multiple metrics."""
    configs = sorted(task_data.keys())

    for metric in ["ood", "in_dist", "snv_abs", "snv_delta"]:
        rankings = {}
        n_used = {}
        for config_key in configs:
            data = task_data[config_key]
            best_n = find_best_n(data, metric, min_reservoirs=5)
            if best_n is None:
                continue
            scores = get_best_hp_metric(data, best_n, metric)
            if len(scores) >= 5:
                ranked = sorted(scores.items(), key=lambda x: -x[1])
                rankings[config_key] = {res: rank for rank, (res, _) in enumerate(ranked)}
                n_used[config_key] = best_n

        config_list = sorted(rankings.keys())
        n = len(config_list)
        if n < 2:
            continue

        corr_matrix = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                r1 = rankings[config_list[i]]
                r2 = rankings[config_list[j]]
                common = sorted(set(r1.keys()) & set(r2.keys()))
                if len(common) >= 5:
                    corr_matrix[i, j] = spearmanr([r1[r] for r in common], [r2[r] for r in common])[
                        0
                    ]

        fig, ax = plt.subplots(figsize=(max(5, n * 1.5), max(4, n * 1.2)))
        im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)

        for i in range(n):
            for j in range(n):
                val = corr_matrix[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

        labels = []
        for c in config_list:
            cfg = c.split("/")[1]
            n_val = n_used.get(c)
            short = SHORT_NAMES.get(cfg, cfg)
            labels.append(f"{short}\n(n={n_val // 1000}k)" if n_val else short)

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(
            f"{task_name} — Ranking Correlation ({METRIC_LABELS[metric]})\n(Spearman rho, best available N)",
            fontsize=11,
            fontweight="bold",
        )
        fig.colorbar(im, ax=ax, label="Spearman rho", shrink=0.8)
        fig.tight_layout()

        out = out_dir / f"ranking_corr_{metric}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.name}")


def main():
    print("Loading results...")
    all_data = load_all_results()

    for task, task_name in [("k562", "K562"), ("yeast", "Yeast")]:
        task_data = {k: v for k, v in all_data.items() if k.startswith(task)}
        if not task_data:
            continue

        out_dir = OUT_DIR / task
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {task_name} ({len(task_data)} configs) ===")
        plot_multi_metric_category_heatmap(task_data, task_name, out_dir)
        plot_multi_metric_ranking_correlation(task_data, task_name, out_dir)

    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
