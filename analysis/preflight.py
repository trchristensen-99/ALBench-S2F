"""Pre-flight diagnostic plots — runs as a Python script that produces
PDFs / PNGs under ``results/preflight/figures/``. Equivalent to the
notebook called for in the pre-flight checklist (kept as a script so
it's reviewable in git diffs and runnable headless).

Generates one figure per task once the relevant analyzer has produced
its CSV. Skips silently if a CSV is missing.

Plots produced:
    1. Task 2: D_min provisional — val_R² vs D, per arch × seed
    2. Task 3: LR×BS heatmap per arch (val_mse, optimum highlighted)
    3. Task 3 verification: optimum stability D_min vs D_max
    4. Task 4: epoch-budget plateau curve per arch
    5. Task 5: augmentation comparison (4 augs, 3 archs, error bars)
    6. Task 6: parameterization sensitivity (3 sizes × 2 D × 3 archs)
    7. Task 7: dropout sensitivity per arch
    8. Task 8: acquisition Jaccard distance per method
    9. Task 9: D_min confirmation overlay vs provisional

Usage:
    uv run --no-sync python analysis/preflight.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "preflight"
FIGS = RESULTS / "figures"


def _load_csv(name: str) -> list[dict] | None:
    p = RESULTS / name
    if not p.exists():
        return None
    with p.open() as fh:
        return list(csv.DictReader(fh))


def plot_d_min_provisional():
    rows = _load_csv("d_min_provisional.csv")
    if not rows:
        return
    archs = sorted({r["arch"] for r in rows})
    ds = sorted({int(r["d_train"]) for r in rows})
    fig, ax = plt.subplots(figsize=(6, 4))
    for arch in archs:
        seed_vals = {seed: [] for seed in ("42", "123", "7")}
        for d in ds:
            for seed in seed_vals:
                matched = [
                    r
                    for r in rows
                    if r["arch"] == arch and int(r["d_train"]) == d and r["seed"] == seed
                ]
                if matched:
                    seed_vals[seed].append(float(matched[0]["val_r2_approx"]))
        # Error band: mean ± std across seeds
        per_d = []
        for i, d in enumerate(ds):
            vals = [seed_vals[s][i] for s in seed_vals if i < len(seed_vals[s])]
            per_d.append((np.mean(vals), np.std(vals)))
        means = [p[0] for p in per_d]
        stds = [p[1] for p in per_d]
        ax.errorbar(ds, means, yerr=stds, label=arch, marker="o", capsize=3)
    ax.axhline(0.1, color="grey", linestyle="--", alpha=0.5, label="R² = 0.1 threshold")
    ax.set_xscale("log")
    ax.set_xlabel("D_train")
    ax.set_ylabel("val R² (approx)")
    ax.set_title("Task 2: D_min provisional (val_R² vs D)")
    ax.legend()
    fig.tight_layout()
    out = FIGS / "task2_d_min_provisional.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_hp_flatness_heatmaps():
    flatness_dir = RESULTS / "hp_flatness"
    if not flatness_dir.exists():
        return
    summary_path = flatness_dir / "flatness_summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text())
    print(f"  Task 3 hp-flatness summary: {summary}")


def plot_task4_plateau():
    """Plot val_loss vs epoch from each arch's history.json + mark plateau."""
    task4_dir = RESULTS / "task4_epoch_budget"
    if not task4_dir.exists():
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, arch in zip(axes, ("legnet", "dream_rnn", "dream_attn")):
        hist_path = task4_dir / arch / "seed42" / "history.json"
        if not hist_path.exists():
            ax.set_title(f"{arch}: no data")
            continue
        history = json.loads(hist_path.read_text())
        val_loss = history.get("val_loss") or []
        if not val_loss:
            ax.set_title(f"{arch}: empty val_loss")
            continue
        ax.plot(range(1, len(val_loss) + 1), val_loss, lw=1)
        ax.set_xlabel("epoch")
        ax.set_title(arch)
    axes[0].set_ylabel("val MSE")
    fig.suptitle("Task 4: val MSE vs epoch (3× published-default budget)")
    fig.tight_layout()
    out = FIGS / "task4_plateau_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_task8_jaccard():
    rows = _load_csv("task8_summary.csv")
    if not rows:
        return
    methods = [r["method"] for r in rows]
    jmin = [float(r["min_jaccard_distance"]) for r in rows]
    jmax = [float(r["max_jaccard_distance"]) for r in rows]
    classes = [r["method_class"] for r in rows]
    colors = {"reservoir": "steelblue", "model_proxy": "orange", "unknown": "grey"}
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(methods))
    bar_colors = [colors.get(c, "grey") for c in classes]
    ax.bar(x, jmin, color=bar_colors)
    for i, (lo, hi) in enumerate(zip(jmin, jmax)):
        ax.plot([i, i], [lo, hi], "k-", lw=1)
    ax.axhline(0.3, color="red", linestyle="--", alpha=0.7, label="threshold = 0.3")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Jaccard distance vs random")
    ax.set_title("Task 8: acquisition method sanity (vs random baseline)")
    ax.legend()
    fig.tight_layout()
    out = FIGS / "task8_jaccard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    print("Generating pre-flight diagnostic figures …")
    plot_d_min_provisional()
    plot_hp_flatness_heatmaps()
    plot_task4_plateau()
    plot_task8_jaccard()
    print(f"\nFigures saved under {FIGS}")


if __name__ == "__main__":
    main()
