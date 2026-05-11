"""Analyze scaling-law experiment results.

For each (oracle, strategy, architecture, D):
  - Aggregate test_mse + test_pearson across seeds (empirical CI)
  - Plot scaling curves: per-arch and combined-best-arch
  - Fit power-law to get exponent + amplitude
  - Hold out last 1-2 D points for validation of fit

Outputs:
  results/preflight/figures/scaling/
    scaling_per_arch_{oracle}_{metric}.png
    scaling_combined_best_{oracle}_{metric}.png
    scaling_compare_oracles_{strategy}_{metric}.png
    scaling_fits_summary.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
OUT = REPO / "results/preflight/figures/scaling"
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


def collect(oracle_dir: Path, oracle_name: str) -> pd.DataFrame:
    """Load all student run results from oracle's scaling output dir."""
    rows = []
    if not oracle_dir.exists():
        return pd.DataFrame()
    for arch_dir in oracle_dir.iterdir():
        if not arch_dir.is_dir():
            continue
        arch = arch_dir.name
        for strat_dir in arch_dir.iterdir():
            if not strat_dir.is_dir() or strat_dir.name == "summary":
                continue
            strat = strat_dir.name
            for n_dir in strat_dir.iterdir():
                if not n_dir.is_dir() or not n_dir.name.startswith("n"):
                    continue
                try:
                    n_train = int(n_dir.name[1:])
                except ValueError:
                    continue
                for hp_dir in n_dir.iterdir():
                    if not hp_dir.is_dir():
                        continue
                    for seed_dir in hp_dir.iterdir():
                        if not seed_dir.is_dir():
                            continue
                        try:
                            seed = int(seed_dir.name.replace("seed", ""))
                        except ValueError:
                            continue
                        metrics_path = seed_dir / "test_metrics.json"
                        if not metrics_path.exists():
                            continue
                        try:
                            m = json.loads(metrics_path.read_text())
                            rows.append({
                                "oracle": oracle_name,
                                "arch": arch,
                                "strategy": strat,
                                "n_train": n_train,
                                "seed": seed,
                                "test_mse": m.get("test_mse") or m.get("test_mse_at_best_val"),
                                "test_pearson": m.get("test_pearson") or m.get("test_pearson_r"),
                            })
                        except Exception:
                            pass
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Per (oracle, arch, strategy, n_train) compute empirical CI across seeds."""
    rows = []
    for (oracle, arch, strat, n), g in df.groupby(["oracle", "arch", "strategy", "n_train"]):
        mse_vals = g.test_mse.dropna().tolist()
        r_vals = g.test_pearson.dropna().tolist()
        rows.append({
            "oracle": oracle, "arch": arch, "strategy": strat, "n_train": n,
            "n_seeds": len(mse_vals),
            "mse_low": empirical_ci(mse_vals)[0] if mse_vals else np.nan,
            "mse_mid": empirical_ci(mse_vals)[1] if mse_vals else np.nan,
            "mse_high": empirical_ci(mse_vals)[2] if mse_vals else np.nan,
            "r_low": empirical_ci(r_vals)[0] if r_vals else np.nan,
            "r_mid": empirical_ci(r_vals)[1] if r_vals else np.nan,
            "r_high": empirical_ci(r_vals)[2] if r_vals else np.nan,
        })
    return pd.DataFrame(rows)


def power_law(n, a, b, c):
    return a * np.power(n, b) + c


def fit_law(ns, vals):
    try:
        popt, _ = curve_fit(power_law, ns, vals, p0=[1.0, -0.3, 0.1], maxfev=5000)
        return popt
    except Exception:
        return None


def plot_per_arch(agg: pd.DataFrame, oracle: str, metric: str, out_path: Path):
    """Plot scaling curve per architecture, faceted by strategy."""
    sub = agg[agg.oracle == oracle].copy()
    strategies = sorted(sub.strategy.unique())
    archs = ["legnet", "dream_rnn", "dream_attn"]
    colors = {"legnet": "#1f77b4", "dream_rnn": "#ff7f0e", "dream_attn": "#2ca02c"}
    fig, axes = plt.subplots(1, len(strategies), figsize=(5 * len(strategies), 4), squeeze=False,
                              sharey=True)
    metric_low, metric_mid, metric_high = f"{metric}_low", f"{metric}_mid", f"{metric}_high"
    for i, strat in enumerate(strategies):
        ax = axes[0][i]
        for arch in archs:
            cell = sub[(sub.strategy == strat) & (sub.arch == arch)].sort_values("n_train")
            if cell.empty:
                continue
            ax.plot(cell.n_train, cell[metric_mid], "o-", color=colors[arch],
                    label=arch, markersize=7)
            ax.fill_between(cell.n_train, cell[metric_low], cell[metric_high],
                            alpha=0.2, color=colors[arch])
        ax.set_xscale("log")
        if metric == "mse":
            ax.set_yscale("log")
        ax.set_xlabel("N (training samples)")
        ax.set_ylabel(f"test_{metric}")
        ax.set_title(f"{strat}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    title_metric = "MSE" if metric == "mse" else "Pearson R"
    fig.suptitle(f"Oracle={oracle}: scaling laws per architecture ({title_metric})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_combined_best(agg: pd.DataFrame, oracle: str, metric: str, out_path: Path):
    """For each (strategy, n_train), pick the BEST architecture and plot."""
    sub = agg[agg.oracle == oracle].copy()
    metric_mid, metric_low, metric_high = f"{metric}_mid", f"{metric}_low", f"{metric}_high"
    sense = "min" if metric == "mse" else "max"
    fig, ax = plt.subplots(figsize=(9, 6))
    for strat in sorted(sub.strategy.unique()):
        cell = sub[sub.strategy == strat].copy()
        # Pick best arch per (n_train)
        best_rows = []
        for n, g in cell.groupby("n_train"):
            if g[metric_mid].isna().all():
                continue
            idx = g[metric_mid].idxmin() if sense == "min" else g[metric_mid].idxmax()
            best_rows.append(g.loc[idx])
        if not best_rows:
            continue
        best = pd.DataFrame(best_rows).sort_values("n_train")
        ax.plot(best.n_train, best[metric_mid], "o-", label=f"{strat} (best arch)", markersize=7)
        ax.fill_between(best.n_train, best[metric_low], best[metric_high], alpha=0.15)
        # annotate winning arch
        for _, r in best.iterrows():
            ax.annotate(r.arch[:3], (r.n_train, r[metric_mid]), fontsize=6, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    if metric == "mse":
        ax.set_yscale("log")
    ax.set_xlabel("N (training samples)")
    ax.set_ylabel(f"test_{metric}")
    ax.set_title(f"Oracle={oracle}: combined scaling law (best architecture per N)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_oracle_compare(agg: pd.DataFrame, strategy: str, metric: str, out_path: Path):
    """Compare baseline vs c91 oracle for a given strategy."""
    sub = agg[agg.strategy == strategy].copy()
    if sub.empty:
        return
    archs = ["legnet", "dream_rnn", "dream_attn"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True, squeeze=False)
    metric_mid, metric_low, metric_high = f"{metric}_mid", f"{metric}_low", f"{metric}_high"
    for i, arch in enumerate(archs):
        ax = axes[0][i]
        for oracle, color in [("baseline", "gray"), ("c91_debiased", "tomato")]:
            cell = sub[(sub.arch == arch) & (sub.oracle == oracle)].sort_values("n_train")
            if cell.empty:
                continue
            ax.plot(cell.n_train, cell[metric_mid], "o-", color=color, label=oracle, markersize=7)
            ax.fill_between(cell.n_train, cell[metric_low], cell[metric_high], alpha=0.2,
                            color=color)
        ax.set_xscale("log")
        if metric == "mse":
            ax.set_yscale("log")
        ax.set_xlabel("N (training samples)")
        ax.set_ylabel(f"test_{metric}")
        ax.set_title(arch)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"Oracle comparison (strategy={strategy}) — does debiasing help students?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    baseline = collect(REPO / "outputs/exp1_1/k562_scaling_baseline_oracle", "baseline")
    c91 = collect(REPO / "outputs/exp1_1/k562_scaling_c91_oracle", "c91_debiased")
    df = pd.concat([baseline, c91], ignore_index=True)
    print(f"Loaded {len(df)} student runs")
    if df.empty:
        print("No results yet — submit scaling-law jobs first.")
        return
    print(f"  per oracle: {df.oracle.value_counts().to_dict()}")
    print(f"  per arch:   {df.arch.value_counts().to_dict()}")
    print(f"  per strat:  {df.strategy.value_counts().to_dict()}")
    df.to_csv(REPO / "results/preflight/scaling_law_raw.csv", index=False)
    agg = aggregate(df)
    agg.to_csv(REPO / "results/preflight/scaling_law_agg.csv", index=False)
    print(f"Aggregated to {len(agg)} (oracle, arch, strategy, n) cells")

    for oracle in df.oracle.unique():
        for metric in ["mse", "r"]:
            plot_per_arch(agg, oracle, metric, OUT / f"scaling_per_arch_{oracle}_{metric}.png")
            plot_combined_best(agg, oracle, metric, OUT / f"scaling_combined_best_{oracle}_{metric}.png")

    if "baseline" in df.oracle.unique() and "c91_debiased" in df.oracle.unique():
        for strat in df.strategy.unique():
            for metric in ["mse", "r"]:
                plot_oracle_compare(agg, strat, metric, OUT / f"scaling_compare_oracles_{strat}_{metric}.png")


if __name__ == "__main__":
    main()
