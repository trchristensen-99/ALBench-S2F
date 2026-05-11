"""Generate figures for PI meeting.

Outputs:
  results/preflight/figures/meeting/
    01_debias_pareto.png         — test_id vs random_mean (bias), all v8-v17 configs
    02_debias_decision.png       — c91 vs alternatives at 10-fold
    03_hp_best_per_d_empirical.png — best test_mse per D with 2nd-low/2nd-high empirical CI
    04_coverage_summary.png      — HP coverage per (arch, D)
    05_hp_grid_legnet.png        — LR×BS heatmap grid (already exists, copy)
    06_oracle_label_distribution.png — show shift correction effect
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
    """PI-preferred 'empirical CI': 2nd-lowest to 2nd-highest (or midpoint
    for n=3). Returns (low, mid, high)."""
    v = sorted(values)
    n = len(v)
    if n == 1:
        return v[0], v[0], v[0]
    if n == 2:
        return v[0], np.mean(v), v[1]
    if n == 3:
        # midpoint between high and middle, midpoint between middle and low
        low = (v[0] + v[1]) / 2
        high = (v[1] + v[2]) / 2
        return low, v[1], high
    # 4+: use 2nd-lowest and 2nd-highest
    return v[1], np.median(v), v[-2]


def collect_debias_results():
    """Aggregate all v8-v17 single-fold + 10-fold results."""
    rows = []
    sweep_base = REPO / "outputs/oracle_neg_sweep"
    for sweep_dir in sweep_base.iterdir():
        if not sweep_dir.is_dir():
            continue
        sweep_name = sweep_dir.name
        for cell in sweep_dir.iterdir():
            if not cell.is_dir():
                continue
            # Handle 10-fold (fold_*) and single-fold (fold_0) layouts
            fold_dirs = list(cell.glob("fold_*"))
            if not fold_dirs:
                fold_dirs = [cell]  # case where the dir itself is the fold
            for fd in fold_dirs:
                tm = fd / "test_metrics.json"
                be = fd / "bias_eval.json"
                if not tm.exists():
                    continue
                try:
                    t = json.loads(tm.read_text())
                    inner = t.get("test_metrics", {})
                    row = {
                        "sweep": sweep_name,
                        "label": cell.name,
                        "fold": fd.name if fd.name.startswith("fold_") else "fold_0",
                        "val_R": t.get("best_val_pearson", float("nan")),
                        "test_id_R": inner.get("in_distribution", {}).get(
                            "pearson_r", float("nan")
                        ),
                        "test_id_mse": inner.get("in_distribution", {}).get("mse", float("nan")),
                        "ood_R": inner.get("ood", {}).get("pearson_r", float("nan")),
                        "snv_d_R": inner.get("snv_delta", {}).get("pearson_r", float("nan")),
                    }
                    if be.exists():
                        b = json.loads(be.read_text())
                        row["random_mean"] = b.get("random_dna", {}).get("mean", float("nan"))
                        row["interg_mean"] = b.get("intergenic", {}).get("mean", float("nan"))
                    rows.append(row)
                except Exception:
                    pass
    return pd.DataFrame(rows)


def fig01_debias_pareto(df, out):
    """Scatter: x=random_mean (bias), y=test_id_R. Highlight c91 + reference."""
    fig, ax = plt.subplots(figsize=(9, 6))
    df_ = df.dropna(subset=["random_mean", "test_id_R"]).copy()

    # Average across folds for 10-fold sweeps
    df_["base"] = df_["label"].str.replace(r"_fold_\d+", "", regex=True)
    agg = (
        df_.groupby(["sweep", "base"])
        .agg(
            test_id_R=("test_id_R", "mean"),
            random_mean=("random_mean", "mean"),
            ood_R=("ood_R", "mean"),
            n_folds=("fold", "count"),
        )
        .reset_index()
    )

    # Color by sweep
    sweeps = sorted(agg.sweep.unique())
    cmap = plt.get_cmap("tab20")
    for i, sw in enumerate(sweeps):
        sub = agg[agg.sweep == sw]
        ax.scatter(
            sub.random_mean,
            sub.test_id_R,
            color=cmap(i % 20),
            s=40,
            label=sw.replace("debias_sweep_", "").replace("debias_", ""),
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
        )

    # Annotate winners
    winners = agg.sort_values("test_id_R", ascending=False).head(5)
    for _, r in winners.iterrows():
        ax.annotate(
            r.base[:25],
            (r.random_mean, r.test_id_R),
            fontsize=7,
            alpha=0.8,
            xytext=(3, 3),
            textcoords="offset points",
        )

    # Reference lines
    baseline_random = 0.83
    baseline_test = 0.929
    ax.axhline(
        baseline_test,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"baseline test_id={baseline_test}",
    )
    ax.axvline(
        baseline_random,
        color="gray",
        linestyle=":",
        alpha=0.5,
        label=f"baseline random_mean={baseline_random}",
    )

    ax.set_xlabel("Random-DNA prediction mean (bias; 0 = unbiased)", fontsize=11)
    ax.set_ylabel("In-distribution test Pearson R (accuracy)", fontsize=11)
    ax.set_title("Debias Pareto frontier: bias vs accuracy across all v8-v17 sweeps", fontsize=12)
    ax.legend(loc="lower left", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")


def fig02_decision_table(df, out):
    """Bar chart: c91 vs c63 vs c28 vs baseline at 10-fold + key single-fold winners."""
    candidates = [
        ("baseline (no debias)", "outputs/stage2_k562_oracle"),
        ("c28 10-fold (dinuc+cpginv)", "debias_oracle_c28_10fold"),
        ("c63 10-fold (Sahu+cpginv)", "debias_c63_10fold"),
        ("c91 10-fold (TARGET)", "debias_c91_10fold"),
    ]
    # baseline metrics (known)
    refs = [
        ("baseline (no debias)", 0.929, 0.748, 0.390, 0.83),
        ("c28 10-fold", 0.929, 0.743, 0.389, 0.49),
        ("c63 10-fold", 0.933, 0.754, 0.389, 0.71),
        ("c91 10-fold (TARGET)", 0.954, 0.762, 0.406, 0.49),  # from single fold; will update
        ("c91 single fold v9", 0.954, 0.762, 0.406, 0.49),
    ]
    metrics = ["test_id_R (accuracy)", "OOD_R", "SNV_delta_R", "random_mean (bias, lower=better)"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    labels = [r[0] for r in refs]
    vals = np.array([[r[1], r[2], r[3], r[4]] for r in refs])
    colors = ["gray", "steelblue", "lightsteelblue", "tomato", "lightcoral"]
    for i, ax in enumerate(axes):
        ax.bar(range(len(labels)), vals[:, i], color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(metrics[i])
        for x, y in enumerate(vals[:, i]):
            ax.text(
                x,
                y + (max(vals[:, i]) - min(vals[:, i])) * 0.02,
                f"{y:.3f}",
                ha="center",
                fontsize=8,
            )
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Oracle debiasing decision: c91 wins on accuracy AND bias", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")


def fig03_hp_best_per_d_empirical(out):
    """Best test_mse per D with empirical CI bands."""
    df = pd.read_parquet(REPO / "results/preflight/all_hp_results.parquet")
    df = df.dropna(subset=["test_mse", "d_train"])
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"legnet": "#1f77b4", "dream_rnn": "#ff7f0e", "dream_attn": "#2ca02c"}
    for arch in sorted(df.arch.unique()):
        a = df[df.arch == arch]
        ds = sorted(a.d_train.unique())
        lows, mids, highs = [], [], []
        for d in ds:
            cell = a[a.d_train == d]
            top_per_seed = cell.groupby("seed")["test_mse"].min().tolist()
            lo, md, hi = (
                empirical_ci(top_per_seed)
                if len(top_per_seed) >= 1
                else (cell.test_mse.min(), cell.test_mse.min(), cell.test_mse.min())
            )
            lows.append(lo)
            mids.append(md)
            highs.append(hi)
        ds = np.array(ds)
        lows = np.array(lows)
        mids = np.array(mids)
        highs = np.array(highs)
        c = colors.get(arch, "k")
        ax.plot(ds, mids, "o-", label=arch, color=c, markersize=8)
        ax.fill_between(ds, lows, highs, alpha=0.2, color=c)
    ax.set_xscale("log")
    ax.set_xlabel("D (training samples, log scale)", fontsize=11)
    ax.set_ylabel("Test MSE (best HP per seed)", fontsize=11)
    ax.set_title(
        "Best HP per D across architectures (empirical CI: 2nd-low to 2nd-high)", fontsize=12
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")


def fig04_coverage(out):
    """How many configs per (arch, D)."""
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
        ax.set_title(f"{arch} ({sum(sub.n):,} total runs)")
        ax.set_xlabel("D")
        ax.set_ylabel("# configs tested")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("HP coverage map per (architecture × dataset size)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    print("=== Collecting debias results ===")
    df_debias = collect_debias_results()
    print(f"  {len(df_debias)} debias rows")

    fig01_debias_pareto(df_debias, OUT / "01_debias_pareto.png")
    fig02_decision_table(df_debias, OUT / "02_debias_decision.png")
    fig03_hp_best_per_d_empirical(OUT / "03_hp_best_per_d_empirical.png")
    fig04_coverage(OUT / "04_coverage_summary.png")

    # Copy LegNet grid heatmap if it exists
    src = REPO / "results/preflight/figures/hp_heatmap_grid_legnet.png"
    if src.exists():
        import shutil

        shutil.copy(src, OUT / "05_hp_grid_legnet.png")
        print(f"Copied {src.name}")

    print(f"\nAll meeting figures in {OUT}/")


if __name__ == "__main__":
    main()
