#!/usr/bin/env python3
"""Cross-config analysis plots for Experiment 1.1.

Generates:
1. Category performance heatmap (category × config)
2. Reservoir ranking correlation matrix (config × config)
3. Rank correlation as a function of training size
4. Data efficiency comparison across configs

Run:
    python scripts/analysis/plot_exp1_1_cross_config.py
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
    "PRM": [
        "prm_1pct",
        "prm_5pct",
        "prm_10pct",
        "prm_20pct",
        "prm_50pct",
        "prm_uniform_1_10",
    ],
    "Recombination": ["recombination_uniform", "recombination_2pt"],
    "Augmentation": ["evoaug_structural", "evoaug_heavy"],
    "Motif": ["motif_planted", "motif_grammar", "motif_grammar_tight"],
    "ISE": ["ise_maximize", "ise_diverse_targets", "ise_target_high"],
    "Variant": ["snv"],
}

SHORT_NAMES = {
    "alphagenome_k562_s1_ag": "K562\nAG→AG",
    "alphagenome_k562_s1_dream_rnn": "K562\nDREAM→AG",
    "dream_rnn_dream_rnn": "DREAM→DREAM",
    "dream_rnn_ag": "AG→DREAM",
    "alphagenome_yeast_s1_ag": "Yeast\nAG→AG",
    "alphagenome_yeast_s1_dream_rnn": "Yeast\nDREAM→AG",
    "alphagenome_yeast_s2_ag": "Yeast\nAG→AG(S2)",
}


def load_all_results():
    """Load all exp1_1 results grouped by config."""
    all_data = {}
    for task in ["k562", "yeast"]:
        task_dir = REPO / "outputs" / "exp1_1" / task
        if not task_dir.exists():
            continue
        for config_dir in sorted(task_dir.iterdir()):
            if not config_dir.is_dir() or config_dir.name == "figures":
                continue
            config = config_dir.name
            key = f"{task}/{config}"

            by_res_n = defaultdict(list)
            for rj in config_dir.rglob("result.json"):
                try:
                    r = json.loads(rj.read_text())
                    res = r.get("reservoir", rj.relative_to(config_dir).parts[0])
                    n = r["n_train"]
                    hp = json.dumps(r.get("hp_config", {}), sort_keys=True)
                    val_r = r.get("val_pearson_r", 0)
                    metrics = {}
                    for ts in ["in_dist", "ood", "snv_abs", "snv_delta"]:
                        v = r["test_metrics"].get(ts, {}).get("pearson_r")
                        if v is not None:
                            metrics[ts] = v
                    by_res_n[(res, n)].append({"hp": hp, "val_r": val_r, **metrics})
                except Exception:
                    pass
            if by_res_n:
                all_data[key] = by_res_n
    return all_data


def get_best_hp_ood(data, target_n):
    """Get best-HP OOD score per reservoir at a given n_train."""
    scores = {}
    for (res, n), results in data.items():
        if n != target_n:
            continue
        ood_vals = [r["ood"] for r in results if "ood" in r]
        if len(ood_vals) < 2:
            continue
        by_hp = defaultdict(list)
        for r in results:
            by_hp[r["hp"]].append(r)
        best_hp_results = max(
            by_hp.values(), key=lambda runs: np.mean([r.get("val_r", 0) for r in runs])
        )
        best_ood = [r["ood"] for r in best_hp_results if "ood" in r]
        if best_ood:
            scores[res] = np.mean(best_ood)
    return scores


def plot_category_heatmap(all_data, target_n=50000):
    """Heatmap: category OOD performance × config."""
    configs = sorted(all_data.keys())
    cats = list(CATEGORIES.keys())

    matrix = np.full((len(cats), len(configs)), np.nan)

    for j, config_key in enumerate(configs):
        data = all_data[config_key]
        for i, (cat, reservoirs) in enumerate(CATEGORIES.items()):
            ood_vals = []
            for res in reservoirs:
                scores = get_best_hp_ood(data, target_n)
                if res in scores:
                    ood_vals.append(scores[res])
            if ood_vals:
                matrix[i, j] = np.mean(ood_vals)

    fig, ax = plt.subplots(figsize=(max(8, len(configs) * 1.5), 5))
    valid = matrix[~np.isnan(matrix)]
    if len(valid) == 0:
        print("No data for category heatmap")
        plt.close()
        return

    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                mid = (np.nanmin(matrix[:, j]) + np.nanmax(matrix[:, j])) / 2
                color = "white" if val > mid else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color)
            else:
                ax.text(j, i, "--", ha="center", va="center", fontsize=8, color="#999")

    short_labels = []
    for c in configs:
        task, config = c.split("/")
        label = f"{task}\n{SHORT_NAMES.get(config, config)}"
        short_labels.append(label)

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(short_labels, fontsize=8, rotation=0)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=10)
    ax.set_title(
        f"OOD Pearson R by Reservoir Category (n={target_n:,})", fontsize=13, fontweight="bold"
    )
    fig.colorbar(im, ax=ax, label="OOD Pearson R", shrink=0.8)

    fig.tight_layout()
    out = OUT_DIR / f"category_heatmap_n{target_n}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")


def plot_ranking_correlation_matrix(all_data, target_n=50000):
    """Pairwise Spearman rank correlation of reservoir rankings across configs."""
    configs = sorted(all_data.keys())
    rankings = {}

    for config_key in configs:
        scores = get_best_hp_ood(all_data[config_key], target_n)
        if len(scores) >= 5:
            ranked = sorted(scores.items(), key=lambda x: -x[1])
            rankings[config_key] = {res: rank for rank, (res, _) in enumerate(ranked)}

    config_list = sorted(rankings.keys())
    n = len(config_list)
    corr_matrix = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(n):
            r1 = rankings[config_list[i]]
            r2 = rankings[config_list[j]]
            common = sorted(set(r1.keys()) & set(r2.keys()))
            if len(common) >= 5:
                ranks1 = [r1[r] for r in common]
                ranks2 = [r2[r] for r in common]
                corr_matrix[i, j] = spearmanr(ranks1, ranks2)[0]

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.0)))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)

    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    short_labels = []
    for c in config_list:
        task, config = c.split("/")
        short_labels.append(f"{task}\n{SHORT_NAMES.get(config, config)}")

    ax.set_xticks(range(n))
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_title(
        f"Reservoir Ranking Correlation Across Configs\n(Spearman rho, OOD at n={target_n:,})",
        fontsize=12,
        fontweight="bold",
    )
    fig.colorbar(im, ax=ax, label="Spearman rho", shrink=0.8)

    fig.tight_layout()
    out = OUT_DIR / f"ranking_correlation_n{target_n}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")


def plot_ranking_correlation_by_size(all_data):
    """How does ranking correlation change with training size?"""
    configs = sorted(all_data.keys())
    sizes = [1000, 5000, 10000, 20000, 50000]

    # For each pair of configs, compute correlation at each size
    pairs = []
    for i in range(len(configs)):
        for j in range(i + 1, len(configs)):
            pairs.append((configs[i], configs[j]))

    fig, ax = plt.subplots(figsize=(10, 6))

    for c1, c2 in pairs:
        corrs = []
        valid_sizes = []
        for n in sizes:
            s1 = get_best_hp_ood(all_data[c1], n)
            s2 = get_best_hp_ood(all_data[c2], n)
            common = sorted(set(s1.keys()) & set(s2.keys()))
            if len(common) >= 5:
                r1 = [s1[r] for r in common]
                r2 = [s2[r] for r in common]
                rho = spearmanr(r1, r2)[0]
                corrs.append(rho)
                valid_sizes.append(n)

        if valid_sizes:
            t1, cfg1 = c1.split("/")
            t2, cfg2 = c2.split("/")
            label = f"{SHORT_NAMES.get(cfg1, cfg1)} vs {SHORT_NAMES.get(cfg2, cfg2)}"
            if t1 != t2:
                label = f"{t1}:{SHORT_NAMES.get(cfg1, cfg1)} vs {t2}:{SHORT_NAMES.get(cfg2, cfg2)}"
            ax.plot(valid_sizes, corrs, marker="o", label=label.replace("\n", " "), linewidth=1.5)

    ax.axhline(0, color="black", linestyle="--", alpha=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("Spearman Rank Correlation (OOD)", fontsize=12)
    ax.set_title(
        "Reservoir Ranking Stability Across Training Sizes", fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 1)

    fig.tight_layout()
    out = OUT_DIR / "ranking_correlation_by_size.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")


def plot_performance_correlation(all_data, target_n=50000):
    """Scatter: reservoir OOD performance in config A vs config B."""
    configs = sorted(all_data.keys())

    # Select interesting pairs (same task, different oracle/student)
    pairs = []
    for i in range(len(configs)):
        for j in range(i + 1, len(configs)):
            t1 = configs[i].split("/")[0]
            t2 = configs[j].split("/")[0]
            if t1 == t2:
                pairs.append((configs[i], configs[j]))

    if not pairs:
        return

    ncols = min(len(pairs), 3)
    nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    for idx, (c1, c2) in enumerate(pairs):
        ax = axes[idx // ncols, idx % ncols]
        s1 = get_best_hp_ood(all_data[c1], target_n)
        s2 = get_best_hp_ood(all_data[c2], target_n)
        common = sorted(set(s1.keys()) & set(s2.keys()))

        if len(common) < 3:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
            continue

        x = [s1[r] for r in common]
        y = [s2[r] for r in common]
        rho = spearmanr(x, y)[0]
        r_pearson = np.corrcoef(x, y)[0, 1]

        # Color by category
        for cat, reservoirs in CATEGORIES.items():
            cx = [s1[r] for r in common if r in reservoirs]
            cy = [s2[r] for r in common if r in reservoirs]
            if cx:
                ax.scatter(cx, cy, s=40, label=cat, alpha=0.8, zorder=3)

        # Annotate reservoir names
        for r in common:
            ax.annotate(
                r[:8],
                (s1[r], s2[r]),
                fontsize=5,
                alpha=0.6,
                xytext=(2, 2),
                textcoords="offset points",
            )

        # Identity line
        lims = [min(min(x), min(y)), max(max(x), max(y))]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)

        cfg1 = SHORT_NAMES.get(c1.split("/")[1], c1.split("/")[1]).replace("\n", " ")
        cfg2 = SHORT_NAMES.get(c2.split("/")[1], c2.split("/")[1]).replace("\n", " ")
        ax.set_xlabel(f"{cfg1} OOD", fontsize=9)
        ax.set_ylabel(f"{cfg2} OOD", fontsize=9)
        ax.set_title(f"{c1.split('/')[0]}: rho={rho:.2f}, R={r_pearson:.2f}", fontsize=10)
        ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.2)

    for j in range(len(pairs), nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.suptitle(
        f"Reservoir OOD Performance Correlation Between Configs (n={target_n:,})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / f"performance_correlation_n{target_n}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading all Exp 1.1 results...")
    all_data = load_all_results()
    print(f"Loaded {len(all_data)} configs: {sorted(all_data.keys())}")

    for n in [1000, 10000, 50000]:
        print(f"\n--- n={n:,} ---")
        plot_category_heatmap(all_data, target_n=n)
        plot_ranking_correlation_matrix(all_data, target_n=n)
        plot_performance_correlation(all_data, target_n=n)

    plot_ranking_correlation_by_size(all_data)

    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
