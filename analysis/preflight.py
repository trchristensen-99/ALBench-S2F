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
    """Per-arch LR×BS heatmaps of best_val_mse with the locked optimum
    starred. Reads task3_lr_bs/<arch>/lr<lr>_bs<bs>/seed<seed>/result.json
    and the locked (LR, BS) from pre_flight_decisions.yaml.
    """
    import yaml
    from collections import defaultdict

    task3_dir = RESULTS / "task3_lr_bs"
    if not task3_dir.exists():
        return
    decisions_path = RESULTS / "pre_flight_decisions.yaml"
    decisions = yaml.safe_load(decisions_path.read_text()) if decisions_path.exists() else {}

    # Walk results
    by_arch: dict[str, dict[tuple[float, int], float]] = defaultdict(dict)
    for f in sorted(task3_dir.rglob("result.json")):
        d = json.loads(f.read_text())
        parts = f.parts
        arch = parts[-4]
        cell = parts[-3]
        try:
            lr_str, bs_str = cell.split("_")
            lr = float(lr_str.lstrip("lr"))
            bs = int(bs_str.lstrip("bs"))
        except ValueError:
            continue
        by_arch[arch][(lr, bs)] = float(d.get("best_val_mse", float("inf")))

    flatness_summary = {}
    flatness_path = RESULTS / "hp_flatness" / "flatness_summary.json"
    if flatness_path.exists():
        flatness_summary = json.loads(flatness_path.read_text())

    archs = sorted(by_arch.keys())
    if not archs:
        return
    fig, axes = plt.subplots(1, len(archs), figsize=(5 * len(archs), 4))
    if len(archs) == 1:
        axes = [axes]
    for ax, arch in zip(axes, archs):
        cells = by_arch[arch]
        lrs = sorted({lr for (lr, _) in cells})
        bss = sorted({bs for (_, bs) in cells})
        M = np.full((len(lrs), len(bss)), np.nan)
        for i, lr in enumerate(lrs):
            for j, bs in enumerate(bss):
                if (lr, bs) in cells:
                    M[i, j] = cells[(lr, bs)]
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis_r")
        # Mark locked optimum
        locked_lr = decisions.get("learning_rate", {}).get(arch, {}).get("value")
        locked_bs = decisions.get("batch_size", {}).get(arch, {}).get("value")
        if locked_lr is not None and locked_bs is not None:
            try:
                i_lr = lrs.index(float(locked_lr))
                j_bs = bss.index(int(locked_bs))
                ax.scatter(
                    [j_bs],
                    [i_lr],
                    marker="*",
                    s=400,
                    c="red",
                    edgecolors="white",
                    linewidths=1.5,
                    zorder=10,
                    label=f"locked: lr={locked_lr:g}, bs={locked_bs}",
                )
                ax.legend(loc="upper right", fontsize=8)
            except (ValueError, IndexError):
                pass
        ax.set_xticks(range(len(bss)))
        ax.set_yticks(range(len(lrs)))
        ax.set_xticklabels([str(b) for b in bss])
        ax.set_yticklabels([f"{lr:.0e}" for lr in lrs])
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Learning rate")
        flat = flatness_summary.get(arch, {}).get("interpretation", "?")
        ax.set_title(f"{arch} — best val MSE @ D=600k\nflatness={flat}")
        plt.colorbar(im, ax=ax, label="val MSE")
    fig.suptitle("Task 3: LR × BS sweep at D=600k (lower = better; ★ = locked optimum)")
    fig.tight_layout()
    out = FIGS / "task3_lr_bs_heatmaps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_hp_flatness_summary():
    """Bar plot of 1st- and 2nd-ring rel-range per arch + threshold lines."""
    flatness_path = RESULTS / "hp_flatness" / "flatness_summary.json"
    if not flatness_path.exists():
        return
    summary = json.loads(flatness_path.read_text())
    archs = sorted(summary.keys())
    if not archs:
        return
    r1 = [summary[a].get("1st_ring_rel_range", 0) for a in archs]
    r2 = [summary[a].get("2nd_ring_rel_range", 0) for a in archs]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(archs))
    w = 0.35
    ax.bar(x - w / 2, r1, w, label="1st-ring rel-range", color="steelblue")
    ax.bar(x + w / 2, r2, w, label="2nd-ring rel-range", color="orange")
    ax.axhline(0.05, color="green", linestyle="--", alpha=0.5, label="0.05 (very flat)")
    ax.axhline(0.15, color="red", linestyle="--", alpha=0.5, label="0.15 (sharp)")
    ax.set_xticks(x)
    ax.set_xticklabels(archs)
    ax.set_ylabel("Relative range (max-min)/optimum")
    ax.set_title("Task 3: HP flatness around locked optimum (lower = flatter = safer to lock)")
    ax.legend()
    ax.set_yscale("log")  # 4.5 vs 0.06 needs log
    fig.tight_layout()
    out = FIGS / "task3_flatness_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_task3_verify():
    """Per-arch: LR vs val_mse at D_min, with the locked LR starred."""
    import yaml
    from collections import defaultdict

    verify_dir = RESULTS / "task3_verify_dmin"
    if not verify_dir.exists():
        return
    decisions_path = RESULTS / "pre_flight_decisions.yaml"
    decisions = yaml.safe_load(decisions_path.read_text()) if decisions_path.exists() else {}

    by_arch: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for f in sorted(verify_dir.rglob("result.json")):
        d = json.loads(f.read_text())
        parts = f.parts
        arch = parts[-4]
        cell = parts[-3]
        try:
            lr_str, _ = cell.split("_")
            lr = float(lr_str.lstrip("lr"))
        except ValueError:
            continue
        by_arch[arch][lr].append(float(d.get("best_val_mse", float("inf"))))

    archs = sorted(by_arch.keys())
    if not archs:
        return
    fig, axes = plt.subplots(1, len(archs), figsize=(5 * len(archs), 4))
    if len(archs) == 1:
        axes = [axes]
    for ax, arch in zip(axes, archs):
        lrs = sorted(by_arch[arch].keys())
        means = [float(np.mean(by_arch[arch][lr])) for lr in lrs]
        mins = [float(np.min(by_arch[arch][lr])) for lr in lrs]
        maxs = [float(np.max(by_arch[arch][lr])) for lr in lrs]
        ax.errorbar(
            lrs,
            means,
            yerr=[np.array(means) - np.array(mins), np.array(maxs) - np.array(means)],
            marker="o",
            capsize=4,
            color="steelblue",
            label="D_min runs (across seeds)",
        )
        locked_lr = decisions.get("learning_rate", {}).get(arch, {}).get("value")
        if locked_lr is not None:
            ax.axvline(
                float(locked_lr),
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"locked LR ({locked_lr:g})",
            )
        ax.set_xscale("log")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("val MSE @ D_min")
        ax.set_title(arch)
        ax.legend(fontsize=8)
    fig.suptitle("Task 3 verify: LR sensitivity at D_min (★ locked LR from D_max sweep)")
    fig.tight_layout()
    out = FIGS / "task3_verify_at_dmin.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


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
    plot_hp_flatness_summary()
    plot_task3_verify()
    plot_task4_plateau()
    plot_task8_jaccard()
    print(f"\nFigures saved under {FIGS}")


if __name__ == "__main__":
    main()
