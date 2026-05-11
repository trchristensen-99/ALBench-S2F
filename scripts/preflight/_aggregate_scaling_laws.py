"""Aggregate scaling-law results for baseline + c91 oracles.

For each (oracle, arch, strategy, n_train) compute mean ± 95% CI across seeds
of test Pearson R and MSE. Plot:
  Panel A: scaling curves per arch (one per oracle)
  Panel B: combined-best-arch envelope per oracle
  Panel C: baseline vs c91 oracle comparison
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


def collect(oracle_root: Path, oracle_name: str) -> pd.DataFrame:
    rows = []
    if not oracle_root.exists():
        return pd.DataFrame()
    for arch_dir in oracle_root.iterdir():
        if not arch_dir.is_dir() or arch_dir.name == "summary":
            continue
        for strat_dir in arch_dir.iterdir():
            if not strat_dir.is_dir():
                continue
            for n_dir in strat_dir.iterdir():
                if not n_dir.name.startswith("n"):
                    continue
                try:
                    n_train = int(n_dir.name[1:])
                except ValueError:
                    continue
                for hp_dir in n_dir.iterdir():
                    if not hp_dir.is_dir():
                        continue
                    for seed_dir in hp_dir.iterdir():
                        r = seed_dir / "result.json"
                        if not r.exists():
                            continue
                        try:
                            d = json.loads(r.read_text())
                        except Exception:
                            continue
                        tm = d.get("test_metrics", {})
                        in_dist = tm.get("in_dist", {})
                        rows.append({
                            "oracle": oracle_name,
                            "arch": arch_dir.name,
                            "strategy": strat_dir.name,
                            "n_train": n_train,
                            "hp": hp_dir.name,
                            "seed": d.get("seed"),
                            "val_R": d.get("val_pearson_r"),
                            "test_R": in_dist.get("pearson_r"),
                            "test_mse": in_dist.get("mse"),
                            "test_spearman": in_dist.get("spearman_r"),
                        })
    return pd.DataFrame(rows)


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Per (oracle, arch, strategy, n_train), pick BEST HP via val_R, then
    aggregate across seeds with mean ± 95% CI."""
    rows = []
    for (ora, arch, strat, n), g in df.groupby(["oracle", "arch", "strategy", "n_train"]):
        # Find best HP by mean val_R
        hp_groups = g.groupby("hp")
        best_hp_val = -np.inf
        best_hp_name = None
        for hp_name, hg in hp_groups:
            mean_val = hg["val_R"].mean()
            if mean_val > best_hp_val:
                best_hp_val = mean_val
                best_hp_name = hp_name
        sub = g[g["hp"] == best_hp_name]
        for metric_name, key in [("test_R", "test_R"), ("test_mse", "test_mse")]:
            vals = sub[key].dropna().to_numpy()
            n_seeds = len(vals)
            if n_seeds == 0:
                continue
            mu = float(vals.mean())
            if n_seeds > 1:
                sem = float(vals.std(ddof=1)) / np.sqrt(n_seeds)
                lo = mu - 1.96 * sem
                hi = mu + 1.96 * sem
            else:
                lo = hi = mu
            rows.append({
                "oracle": ora,
                "arch": arch,
                "strategy": strat,
                "n_train": n,
                "best_hp": best_hp_name,
                "metric": metric_name,
                "mean": mu,
                "low": lo,
                "high": hi,
                "n_seeds": n_seeds,
            })
    return pd.DataFrame(rows)


def plot_per_arch(agg: pd.DataFrame, oracle: str, metric: str, out: Path):
    sub = agg[(agg.oracle == oracle) & (agg.metric == metric)].copy()
    if sub.empty:
        print(f"  no data for {oracle} / {metric}")
        return
    strategies = sorted(sub.strategy.unique())
    archs = sorted(sub.arch.unique())
    colors = {"legnet": "#1f77b4", "dream_rnn": "#ff7f0e",
              "dream_cnn": "#9467bd", "dream_attn": "#2ca02c"}
    fig, axes = plt.subplots(1, len(strategies), figsize=(5 * len(strategies), 4),
                              squeeze=False, sharey=True)
    for i, strat in enumerate(strategies):
        ax = axes[0][i]
        for arch in archs:
            cell = sub[(sub.strategy == strat) & (sub.arch == arch)].sort_values("n_train")
            if cell.empty:
                continue
            ax.plot(cell.n_train, cell["mean"], "o-",
                    color=colors.get(arch, "k"), label=arch, markersize=7)
            ax.fill_between(cell.n_train, cell["low"], cell["high"],
                            alpha=0.2, color=colors.get(arch, "k"))
        ax.set_xscale("log")
        if metric == "test_mse":
            ax.set_yscale("log")
        ax.set_xlabel("N (training samples)")
        if i == 0:
            ax.set_ylabel(f"test {metric}")
        ax.set_title(strat)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    metric_pretty = "Pearson R" if metric == "test_R" else "MSE"
    fig.suptitle(f"Scaling law — {oracle} oracle, {metric_pretty} (mean ± 95% CI)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")


def plot_combined_best(agg: pd.DataFrame, oracle: str, metric: str, out: Path):
    sub = agg[(agg.oracle == oracle) & (agg.metric == metric)].copy()
    if sub.empty:
        return
    sense = "max" if metric == "test_R" else "min"
    fig, ax = plt.subplots(figsize=(9, 6))
    for strat in sorted(sub.strategy.unique()):
        cell = sub[sub.strategy == strat]
        best_rows = []
        for n, g in cell.groupby("n_train"):
            if g["mean"].isna().all():
                continue
            idx = g["mean"].idxmax() if sense == "max" else g["mean"].idxmin()
            best_rows.append(g.loc[idx])
        if not best_rows:
            continue
        best = pd.DataFrame(best_rows).sort_values("n_train")
        ax.plot(best.n_train, best["mean"], "o-", label=f"{strat} (best arch)",
                markersize=7)
        ax.fill_between(best.n_train, best["low"], best["high"], alpha=0.15)
        for _, r in best.iterrows():
            ax.annotate(r.arch[:3], (r.n_train, r["mean"]),
                        fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    if metric == "test_mse":
        ax.set_yscale("log")
    metric_pretty = "Pearson R" if metric == "test_R" else "MSE"
    ax.set_xlabel("N (training samples)")
    ax.set_ylabel(f"test {metric_pretty}")
    ax.set_title(f"Combined best-arch scaling — {oracle} oracle ({metric_pretty})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    print("=== Collecting baseline scaling results ===")
    baseline = collect(REPO / "outputs/exp1_1/k562_scaling_baseline_oracle", "baseline")
    print(f"  {len(baseline)} baseline runs")
    print("=== Collecting c91 scaling results ===")
    c91 = collect(REPO / "outputs/exp1_1/k562_scaling_c91_oracle", "c91")
    print(f"  {len(c91)} c91 runs")
    df = pd.concat([baseline, c91], ignore_index=True)
    if df.empty:
        print("No data yet.")
        return
    df.to_csv(REPO / "results/preflight/scaling_law_raw_v2.csv", index=False)
    print(f"  Raw counts by (oracle, arch):")
    print(df.groupby(["oracle", "arch"]).size())

    agg = aggregate_seeds(df)
    agg.to_csv(REPO / "results/preflight/scaling_law_agg_v2.csv", index=False)

    for oracle in df.oracle.unique():
        for metric in ["test_R", "test_mse"]:
            plot_per_arch(agg, oracle, metric,
                          OUT / f"13_scaling_per_arch_{oracle}_{metric}.png")
            plot_combined_best(agg, oracle, metric,
                                OUT / f"14_scaling_combined_{oracle}_{metric}.png")


if __name__ == "__main__":
    main()
