"""Plot per-model val trajectories (+running best) per strategy, and extract
per-model train_time_sec stats to estimate Step-2 R×A×D wall time."""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "outputs/hp_step1_bakeoff_e100/k562_genomic_d30000"
SEEDS = ["seed42_0", "seed43_1", "seed44_2"]
OUT_PNG = "outputs/analysis/budget_per_model_trajectories.png"


def cell_series(cd):
    rows = []
    for m in sorted(glob.glob(os.path.join(cd, "r*_meta.json"))):
        try:
            d = json.load(open(m))
        except Exception:
            continue
        vp = d.get("val_pearson")
        if vp is None:
            continue
        rows.append((int(d.get("round", -1)), float(vp), float(d.get("train_time_sec", np.nan))))
    rows.sort()
    return rows


def collect():
    strat = {}
    for s in SEEDS:
        for cd in sorted(glob.glob(os.path.join(ROOT, s, "*"))):
            if not os.path.isdir(cd):
                continue
            rows = cell_series(cd)
            if len(rows) < 20:
                continue
            strat.setdefault(os.path.basename(cd), []).append(rows)
    return strat


def plot(strat):
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    names = sorted(strat)
    ncol = 3
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.8 * nrow), sharex=True)
    axes = np.array(axes).reshape(-1)
    for ax, name in zip(axes, names):
        for rows in strat[name]:
            x = [r[0] for r in rows]
            y = [r[1] for r in rows]
            rb = np.maximum.accumulate(y)
            ax.scatter(x, y, s=5, alpha=0.25, color="#888", linewidths=0)
            ax.plot(x, rb, color="#1f77b4", lw=1.3, alpha=0.8)
        ax.axvline(100, color="r", ls="--", lw=0.8, alpha=0.6)
        ax.set_title(name, fontsize=9)
        ax.set_ylim(0.55, 0.83)
        ax.grid(alpha=0.2)
    for ax in axes[len(names) :]:
        ax.axis("off")
    fig.supxlabel("model index (round)  —  red dashed = budget 100")
    fig.supylabel("val Pearson (dots) / running best (blue)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"WROTE {OUT_PNG}")


def timing(strat):
    print("\n=== per-model train_time_sec (D=30k genomic) ===")
    all_t = []
    print(f"  {'strategy':22s} {'n':>5s} {'median_s':>9s} {'mean_s':>8s} {'p90_s':>8s}")
    for name in sorted(strat):
        t = np.array([r[2] for rows in strat[name] for r in rows])
        t = t[np.isfinite(t)]
        if not len(t):
            continue
        all_t.append(t)
        print(
            f"  {name:22s} {len(t):5d} {np.median(t):9.1f} {t.mean():8.1f} {np.percentile(t, 90):8.1f}"
        )
    allt = np.concatenate(all_t)
    print(
        f"\n  ALL MODELS  n={len(allt)}  median={np.median(allt):.1f}s  mean={allt.mean():.1f}s  p90={np.percentile(allt, 90):.1f}s"
    )
    return float(np.median(allt)), float(allt.mean())


if __name__ == "__main__":
    strat = collect()
    plot(strat)
    timing(strat)
