"""Generate figures for PI meeting (REFRESHED with latest c91/c86 10-fold data).

Outputs:
  results/preflight/figures/meeting/
    01_debias_pareto.png         — test_id vs random_mean (bias), all v8-v17 configs + 10-fold winners
    02_debias_decision.png       — c91/c86/c63/c28 10-fold side-by-side with empirical CIs
    03_hp_best_per_d_empirical.png — best test_mse per D with 2nd-low/2nd-high empirical CI
    04_coverage_summary.png      — HP coverage per (arch, D)
    05_hp_grid_legnet.png        — LR×BS heatmap grid (copy)
    06_oracle_comparison_10fold.png — 10-fold ensemble comparison
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
OUT = REPO / "results/preflight/figures/meeting"
OUT.mkdir(parents=True, exist_ok=True)


def empirical_ci(values):
    v = sorted(values)
    n = len(v)
    if n == 1:
        return v[0], v[0], v[0]
    if n == 2:
        return v[0], np.mean(v), v[1]
    if n == 3:
        return (v[0] + v[1]) / 2, v[1], (v[1] + v[2]) / 2
    return v[1], np.median(v), v[-2]


def collect_debias_results():
    rows = []
    base = REPO / "outputs/oracle_neg_sweep"
    for sd in base.iterdir():
        if not sd.is_dir():
            continue
        for cell in sd.iterdir():
            if not cell.is_dir():
                continue
            fold_dirs = list(cell.glob("fold_*"))
            if not fold_dirs:
                fold_dirs = [cell]
            for fd in fold_dirs:
                tm = fd / "test_metrics.json"
                be = fd / "bias_eval.json"
                if not tm.exists():
                    continue
                try:
                    t = json.loads(tm.read_text())
                    inner = t.get("test_metrics", {})
                    row = {
                        "sweep": sd.name,
                        "label": cell.name,
                        "fold": fd.name if fd.name.startswith("fold_") else "fold_0",
                        "test_id": inner.get("in_distribution", {}).get("pearson_r", np.nan),
                        "ood": inner.get("ood", {}).get("pearson_r", np.nan),
                        "snv_d": inner.get("snv_delta", {}).get("pearson_r", np.nan),
                    }
                    if be.exists():
                        b = json.loads(be.read_text())
                        row["random_mean"] = b.get("random_dna", {}).get("mean", np.nan)
                    rows.append(row)
                except Exception:
                    pass
    return pd.DataFrame(rows)


def fig01_pareto(df, out):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    df_ = df.dropna(subset=["random_mean", "test_id"]).copy()
    df_["base"] = df_["label"].str.replace(r"_fold_\d+", "", regex=True)
    agg = df_.groupby(["sweep", "base"]).agg(
        test_id=("test_id", "mean"),
        random_mean=("random_mean", "mean"),
        n_folds=("fold", "count"),
    ).reset_index()
    cmap = plt.get_cmap("tab20")
    sweeps = sorted(agg.sweep.unique())
    for i, sw in enumerate(sweeps):
        sub = agg[agg.sweep == sw]
        s = 200 if sub.n_folds.iloc[0] >= 9 else 35  # 10-fold = bigger marker
        ax.scatter(sub.random_mean, sub.test_id, color=cmap(i % 20), s=s,
                   label=sw.replace("debias_sweep_", "").replace("debias_", ""),
                   alpha=0.7, edgecolors="white", linewidths=0.6)
    # Annotate notable winners
    notable = ["c91_fold", "c86_fold", "c63_fold", "c28_fold_", "c170_03_10fold",
               "c91", "c170_03", "c12_grid_f10_lam010"]
    for _, r in agg.iterrows():
        if any(n in r.base for n in notable):
            label = r.base.replace("c91_fold_0", "c91 (single)").replace("_10fold", " (10-fold)")
            ax.annotate(label[:25], (r.random_mean, r.test_id),
                        fontsize=7.5, alpha=0.95, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0.929, color="gray", linestyle="--", alpha=0.5, label="baseline test_id=0.929")
    ax.axvline(0.83, color="gray", linestyle=":", alpha=0.5, label="baseline bias=0.83")
    ax.set_xlabel("Random-DNA prediction mean (bias; lower = less biased)", fontsize=11)
    ax.set_ylabel("In-distribution test Pearson R (accuracy)", fontsize=11)
    ax.set_title("Debias Pareto: bias × accuracy across all sweeps (big = 10-fold)", fontsize=12)
    ax.legend(loc="lower left", fontsize=6.5, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")


def fig02_decision_10fold(df, out):
    """10-fold ensemble comparison: c91 vs c86 vs c63 vs c28 vs baseline."""
    candidates = [
        ("baseline", None, "gray"),
        ("c28_10fold", "debias_oracle_c28_10fold", "steelblue"),
        ("c63_10fold", "debias_c63_10fold", "lightsteelblue"),
        ("c86_10fold", "debias_c86_10fold", "lightcoral"),
        ("c91_10fold", "debias_c91_10fold", "tomato"),
    ]
    # baseline metrics (manually known)
    baseline_means = {"test_id": 0.929, "ood": 0.748, "snv_d": 0.390, "random_mean": 0.83}
    metrics = ["test_id", "ood", "snv_d", "random_mean"]
    metric_labels = ["test_id Pearson R", "OOD Pearson R", "SNV delta Pearson R",
                     "Random-DNA mean\n(bias; lower=better)"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for mi, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[mi]
        labels = []
        mids = []; los = []; his = []; colors = []
        for name, sweep_name, color in candidates:
            labels.append(name)
            colors.append(color)
            if sweep_name is None:
                mids.append(baseline_means[metric])
                los.append(baseline_means[metric])
                his.append(baseline_means[metric])
            else:
                sub = df[df.sweep == sweep_name][metric].dropna().tolist()
                if not sub:
                    mids.append(np.nan); los.append(np.nan); his.append(np.nan)
                else:
                    lo, md, hi = empirical_ci(sub)
                    mids.append(md); los.append(lo); his.append(hi)
        ax.bar(range(len(labels)), mids, color=colors,
               yerr=[[m - l for m, l in zip(mids, los)],
                     [h - m for m, h in zip(mids, his)]],
               capsize=4, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(mlabel, fontsize=10)
        for x, m in enumerate(mids):
            if not np.isnan(m):
                yoff = (max(his) - min(los)) * 0.04
                ax.text(x, m + yoff, f"{m:.3f}", ha="center", fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("10-fold ensemble comparison — empirical CI from 2nd-low to 2nd-high",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")


def fig03_hp_best_per_d(out):
    df = pd.read_parquet(REPO / "results/preflight/all_hp_results.parquet")
    df = df.dropna(subset=["test_mse", "d_train"])
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"legnet": "#1f77b4", "dream_rnn": "#ff7f0e", "dream_attn": "#2ca02c"}
    for arch in sorted(df.arch.unique()):
        a = df[df.arch == arch]
        ds = sorted(a.d_train.unique())
        lows, mids, highs = [], [], []
        for d in ds:
            cell = a[a.d_train == d]
            best_per_seed = cell.groupby("seed")["test_mse"].min().tolist() if "seed" in cell.columns else cell.test_mse.tolist()
            lo, md, hi = empirical_ci(best_per_seed)
            lows.append(lo); mids.append(md); highs.append(hi)
        ax.plot(ds, mids, "o-", label=arch, color=colors.get(arch, "k"), markersize=8)
        ax.fill_between(ds, lows, highs, alpha=0.2, color=colors.get(arch, "k"))
    ax.set_xscale("log")
    ax.set_xlabel("D (training samples, log scale)", fontsize=11)
    ax.set_ylabel("Test MSE (best HP per seed)", fontsize=11)
    ax.set_title("Best HP per D × architecture (empirical CI: 2nd-low to 2nd-high)", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")


def fig04_coverage(out):
    df = pd.read_parquet(REPO / "results/preflight/all_hp_results.parquet")
    counts = df.groupby(["arch", "d_train"]).size().reset_index(name="n")
    archs = sorted(counts.arch.unique())
    fig, axes = plt.subplots(1, len(archs), figsize=(5 * len(archs), 4), squeeze=False)
    for i, arch in enumerate(archs):
        ax = axes[0][i]
        sub = counts[counts.arch == arch].sort_values("d_train")
        ax.bar([str(int(d)) for d in sub.d_train], sub.n, color="steelblue")
        for x, y in zip(range(len(sub)), sub.n):
            ax.text(x, y + 1, str(y), ha="center", fontsize=8)
        ax.set_title(f"{arch} ({sum(sub.n):,} total)")
        ax.set_xlabel("D")
        ax.set_ylabel("# configs tested")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("HP coverage map per (arch × D)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")


def fig07_total_runs_summary(df, out):
    """Bar chart showing scope of work: total runs per category."""
    work = {
        "HP optimization": int(pd.read_parquet(REPO / "results/preflight/all_hp_results.parquet").shape[0]),
        "Debias single-fold": len(df[df.fold == "fold_0"]),
        "Debias 10-fold runs": len(df[df.fold != "fold_0"]),
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    cats = list(work.keys())
    vals = list(work.values())
    bars = ax.bar(cats, vals, color=["#4c72b0", "#dd8452", "#55a868"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 5, f"{v:,}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("# Training runs")
    ax.set_title(f"Total work since last PI meeting: {sum(vals):,} training runs", fontsize=12)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    df_debias = collect_debias_results()
    print(f"  {len(df_debias)} debias rows")
    fig01_pareto(df_debias, OUT / "01_debias_pareto.png")
    fig02_decision_10fold(df_debias, OUT / "02_debias_decision.png")
    fig03_hp_best_per_d(OUT / "03_hp_best_per_d_empirical.png")
    fig04_coverage(OUT / "04_coverage_summary.png")
    fig07_total_runs_summary(df_debias, OUT / "07_total_runs.png")

    # Copy LegNet heatmap grid
    src = REPO / "results/preflight/figures/hp_heatmap_grid_legnet.png"
    if src.exists():
        import shutil
        shutil.copy(src, OUT / "05_hp_grid_legnet.png")
        print(f"Copied {src.name}")

    print(f"\nAll meeting figures saved to {OUT}/")


if __name__ == "__main__":
    main()
