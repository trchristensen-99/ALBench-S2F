"""Scaling-law plots from focused_train summaries.

Aggregates ElasticNet ensemble metrics across reservoir-sampling seeds per cell,
plots Pearson + log-log MSE with mean ± std error bands.

Outputs:
  outputs/focused_plots/main_3panel{,_mse}.{png,pdf}     — Genomic / SNV Δ / OOD
  outputs/focused_plots/all_panels{,_mse}.{png,pdf}      — full 17-panel grid
  outputs/focused_plots/data.csv                          — long-form table
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DS = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
DS_VISIBLE_MIN = 1_000
FIT_DS_MIN = 3_000

RESERVOIRS = [
    ("genomic", "Genomic", "#1f77b4", "o"),
    ("random", "Random", "#7f7f7f", "s"),
    ("prm_1pct", "PRM 1%", "#2ca02c", "^"),
    ("prm_10pct", "PRM 10%", "#9467bd", "v"),
    ("evoaug_heavy", "EvoAug", "#ff7f0e", "P"),
    ("motif_shuffled", "Motif-shuffled", "#8c564b", "X"),
    ("motif_planted_v2", "Motif-planted-v2", "#e377c2", "D"),
]

PRIMARY_PANELS = [
    ("genomic", "Genomic Reference (chr 7+13)"),
    ("snv_delta", "SNV Effect (Δ log2FC)"),
    ("ood", "High-Activity Designed (Gosai)"),
]

ALL_PANELS = [
    ("genomic", "Genomic Reference"),
    ("snv_delta", "SNV Effect (Δ log2FC)"),
    ("ood", "High-Activity Designed"),
    ("snv_ref", "SNV Ref"),
    ("snv_alt", "SNV Alt"),
    ("sub_low", "Substitution (low)"),
    ("sub_med", "Substitution (med)"),
    ("sub_high", "Substitution (high)"),
    ("ins_low", "Insertion (low)"),
    ("ins_med", "Insertion (med)"),
    ("ins_high", "Insertion (high)"),
    ("del_low", "Deletion (low)"),
    ("del_med", "Deletion (med)"),
    ("del_high", "Deletion (high)"),
    ("translocation", "Translocation"),
    ("inversion", "Inversion"),
    ("dinuc_shuffle", "Dinuc Shuffle"),
    ("random_32k", "Random 32k"),
]

ROOT = Path("outputs/focused_train")
OUT = Path("outputs/focused_plots")


def _ingest_summary(j: dict, reservoir: str, D: int, raw: dict):
    """Add per_set + snv_delta entries from a summary.json to the raw accumulator."""
    for panel, m in j.get("per_set", {}).items():
        pr, mse = m.get("pearson"), m.get("mse")
        if pr is None or (isinstance(pr, float) and np.isnan(pr)):
            continue
        raw[(reservoir, D, panel)]["pearson"].append(pr)
        raw[(reservoir, D, panel)]["mse"].append(mse)
    sd = j.get("snv_delta", {}).get("oracle", {})
    if sd:
        raw[(reservoir, D, "snv_delta")]["pearson"].append(sd["pearson"])
        raw[(reservoir, D, "snv_delta")]["mse"].append(sd["mse"])


def load_aggregated() -> dict:
    """Return {(reservoir, D, panel): {'pearson_mean','pearson_std','mse_mean','mse_std','n_seeds'}}.

    Sources:
      1. outputs/focused_train/k562_{res}_d{D}_seed{S}/summary.json   (3 configs × 3 seeds per cell)
      2. outputs/random_d1M_verify/seed{S}/summary.json               (partial mixed6 verify reps, random_d1M only)
      3. outputs/full_sweep/k562_random_d1000000_seed42/summary.json  (original mixed6, random_d1M only)
    """
    raw: dict[tuple[str, int, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    # 1. focused_train
    for cell in ROOT.iterdir():
        if not cell.is_dir():
            continue
        sj = cell / "summary.json"
        if not sj.exists():
            continue
        try:
            j = json.loads(sj.read_text())
        except Exception:
            continue
        name = cell.name
        parts = name.split("_")
        try:
            d_idx = next(i for i, p in enumerate(parts) if p.startswith("d") and p[1:].isdigit())
        except StopIteration:
            continue
        D = int(parts[d_idx][1:])
        reservoir = "_".join(parts[1:d_idx])
        _ingest_summary(j, reservoir, D, raw)

    # 2. random_d1M_verify (additional seed reps for random reservoir at D=1M)
    verify_root = Path("outputs/random_d1M_verify")
    if verify_root.exists():
        for seed_dir in verify_root.iterdir():
            if not seed_dir.is_dir():
                continue
            sj = seed_dir / "summary.json"
            if not sj.exists():
                continue
            try:
                j = json.loads(sj.read_text())
            except Exception:
                continue
            _ingest_summary(j, "random", 1_000_000, raw)

    # 3. main_sweep seed=42 for random_d1M as 4th seed
    main_sweep_random = Path("outputs/full_sweep/k562_random_d1000000_seed42/summary.json")
    if main_sweep_random.exists():
        try:
            j = json.loads(main_sweep_random.read_text())
            _ingest_summary(j, "random", 1_000_000, raw)
        except Exception:
            pass

    out = {}
    for k, v in raw.items():
        out[k] = {
            "pearson_mean": float(np.mean(v["pearson"])),
            "pearson_std": float(np.std(v["pearson"], ddof=1)) if len(v["pearson"]) > 1 else 0.0,
            "mse_mean": float(np.mean(v["mse"])),
            "mse_std": float(np.std(v["mse"], ddof=1)) if len(v["mse"]) > 1 else 0.0,
            "n_seeds": len(v["pearson"]),
        }
    return out


def power_law_fit(xs, ys):
    mask = (xs > 0) & (ys > 0) & np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 3:
        return None
    coef = np.polyfit(np.log10(xs[mask]), np.log10(ys[mask]), 1)
    return float(coef[1]), float(coef[0])


def plot_panel(ax, data, panel_key, panel_label, metric="pearson", show_legend=False):
    visible_ds = [d for d in DS if d >= DS_VISIBLE_MIN]
    for key, label, color, marker in RESERVOIRS:
        xs, ys, errs = [], [], []
        for D in visible_ds:
            d = data.get((key, D, panel_key))
            if d is None:
                continue
            xs.append(D)
            ys.append(d[f"{metric}_mean"])
            errs.append(d[f"{metric}_std"])
        if not xs:
            continue
        xs = np.array(xs)
        ys = np.array(ys)
        errs = np.array(errs)
        ax.fill_between(xs, ys - errs, ys + errs, color=color, alpha=0.15, edgecolor="none")
        ax.plot(
            xs,
            ys,
            "-",
            color=color,
            marker=marker,
            label=label,
            markersize=7,
            linewidth=1.8,
            alpha=0.92,
        )
        if metric == "mse":
            fit_mask = xs >= FIT_DS_MIN
            if fit_mask.sum() >= 3:
                fit = power_law_fit(xs[fit_mask], ys[fit_mask])
                if fit:
                    intercept, slope = fit
                    xs_s = np.logspace(np.log10(FIT_DS_MIN), np.log10(1e6), 50)
                    ys_s = 10 ** (intercept + slope * np.log10(xs_s))
                    ax.plot(xs_s, ys_s, "--", color=color, linewidth=1.2, alpha=0.55)
    ax.set_xscale("log")
    ax.set_xlabel("Training-set size D", fontsize=10)
    if metric == "mse":
        ax.set_yscale("log")
        ax.set_ylabel("MSE (oracle)", fontsize=10)
    else:
        ax.set_ylabel("Pearson R", fontsize=10)
    ax.set_title(panel_label, fontsize=11, fontweight="bold")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xticks(visible_ds)
    ax.set_xticklabels(
        [f"{d:,}" if d >= 1000 else str(d) for d in visible_ds], rotation=30, ha="right", fontsize=8
    )
    if show_legend:
        loc = "upper right" if metric == "mse" else "lower right"
        ax.legend(loc=loc, fontsize=8, framealpha=0.92)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_aggregated()
    print(f"loaded {len(data)} (reservoir, D, panel) aggregates")

    # 3-panel primary
    for metric in ["pearson", "mse"]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=(metric == "pearson"))
        for i, (key, label) in enumerate(PRIMARY_PANELS):
            plot_panel(axes[i], data, key, label, metric, show_legend=(i == 2))
        if metric == "pearson":
            axes[0].set_ylim(0, 1.0)
        fig.tight_layout()
        suffix = "" if metric == "pearson" else "_mse"
        fig.savefig(OUT / f"main_3panel{suffix}.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT / f"main_3panel{suffix}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  saved main_3panel{suffix}")

    # All-panels grid (6 columns × 3 rows = 18 slots; we use 18 panels)
    for metric in ["pearson", "mse"]:
        n_panels = len(ALL_PANELS)
        ncols = 6
        nrows = (n_panels + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = axes.flatten()
        for i, (key, label) in enumerate(ALL_PANELS):
            plot_panel(axes[i], data, key, label, metric, show_legend=(i == 0))
        for j in range(n_panels, len(axes)):
            axes[j].axis("off")
        if metric == "pearson":
            for ax in axes[:n_panels]:
                ax.set_ylim(-0.1, 1.0)
        fig.tight_layout()
        suffix = "" if metric == "pearson" else "_mse"
        fig.savefig(OUT / f"all_panels{suffix}.png", dpi=180, bbox_inches="tight")
        fig.savefig(OUT / f"all_panels{suffix}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  saved all_panels{suffix}")

    # CSV
    import csv

    with open(OUT / "data.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "reservoir",
                "D",
                "panel",
                "n_seeds",
                "pearson_mean",
                "pearson_std",
                "mse_mean",
                "mse_std",
            ]
        )
        for (r, D, p), v in sorted(data.items()):
            w.writerow(
                [
                    r,
                    D,
                    p,
                    v["n_seeds"],
                    f"{v['pearson_mean']:.4f}",
                    f"{v['pearson_std']:.4f}",
                    f"{v['mse_mean']:.4f}",
                    f"{v['mse_std']:.4f}",
                ]
            )
    print(f"  saved data.csv ({len(data)} rows)")


if __name__ == "__main__":
    main()
