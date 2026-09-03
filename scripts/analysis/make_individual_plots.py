"""Generate one focused figure per validation test from validation_results.json
(saved to ~/Downloads/hp_strategy_curves/). No HPC dependency — pure local plot."""

import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.expanduser("~/Downloads/hp_strategy_curves")
J = json.load(open(os.path.join(OUT, "validation_results.json")))
NOISE = J.get("noise_floor", 0.005)
K5 = J["K5_menu"]
D = J["D"]


def plot_t1():
    rows = J["T1"]
    labels = [r["cell"] for r in rows]
    full = np.array([r["all18"] for r in rows])
    k5 = np.array([r["k5"] for r in rows])
    gap = full - k5
    xx = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [3, 1.4]})
    ax1.bar(xx - 0.20, full, width=0.40, color="#444", alpha=0.85, label="all 18 strategies")
    ax1.bar(xx + 0.20, k5, width=0.40, color="#2ca02c", alpha=0.85, label="K=5 strategies only")
    ax1.set_xticks(xx)
    ax1.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax1.set_ylabel("ensemble oracle_r (within cell)")
    ax1.set_title(f"T1: K=5 strategy subset vs all-18 — within-cell ensemble  (D={D})", fontsize=11)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(axis="y", alpha=0.25)
    ymin = min(k5.min(), full.min()) - 0.01
    ax1.set_ylim(ymin, 1.0)

    ax2.bar(xx, gap, color="#1f77b4", alpha=0.85)
    ax2.axhline(NOISE, color="red", ls="--", lw=1, label=f"noise floor ±{NOISE}")
    ax2.axhline(-NOISE, color="red", ls="--", lw=1)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xticks(xx)
    ax2.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax2.set_ylabel("gap (all18 − K=5)")
    med = float(np.median(gap))
    ax2.set_title(f"per-cell gap   median = {med:+.4f}", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle(f"K=5 menu: {', '.join(K5)}", fontsize=10, color="#444")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT, "T1_K5_vs_all18.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"WROTE {out}")


def plot_t2():
    rows = J["T2"]
    labels = [r["cell"] for r in rows]
    greedy = np.array([r["greedy_k5"] for r in rows])
    top1 = np.array([r["top1_k5"] for r in rows])
    gap = greedy - top1
    xx = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [3, 1.4]})
    ax1.bar(xx - 0.20, greedy, width=0.40, color="#444", alpha=0.85, label="full pool, K=5 strats")
    ax1.bar(
        xx + 0.20,
        top1,
        width=0.40,
        color="#ff7f0e",
        alpha=0.85,
        label="top-1 per K=5 strat (5 models)",
    )
    ax1.set_xticks(xx)
    ax1.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax1.set_ylabel("ensemble oracle_r (within cell)")
    ax1.set_title(f"T2: top-1 per strategy vs K=5 full pool — within-cell  (D={D})", fontsize=11)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(axis="y", alpha=0.25)
    ymin = min(top1.min(), greedy.min()) - 0.01
    ax1.set_ylim(ymin, 1.0)

    ax2.bar(xx, gap, color="#ff7f0e", alpha=0.85)
    ax2.axhline(NOISE, color="red", ls="--", lw=1, label=f"noise floor ±{NOISE}")
    ax2.axhline(-NOISE, color="red", ls="--", lw=1)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xticks(xx)
    ax2.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax2.set_ylabel("gap (full − top1)")
    med = float(np.median(gap))
    ax2.set_title(f"per-cell gap   median = {med:+.4f}", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle(f"K=5 menu: {', '.join(K5)}", fontsize=10, color="#444")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT, "T2_top1_vs_K5_full.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"WROTE {out}")


def plot_t3():
    rows = J["T3"]
    # Build: per N_pilot, per held-out reservoir family, distribution of gaps on truly-held cells
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))

    # Panel 1: distribution by N_pilot (held-out cells only)
    ax = axes[0]
    groups = {1: [], 2: [], 3: []}
    for r in rows:
        if not r["held_in_pilot"]:
            groups[r["n_pilot"]].append(r["gap"])
    data = [groups[k] for k in (1, 2, 3) if groups[k]]
    labels = [f"{k} pilot Rs\n(n={len(groups[k])})" for k in (1, 2, 3) if groups[k]]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    bp = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True, widths=0.55)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.axhline(NOISE, color="red", ls="--", lw=1, label=f"noise floor +{NOISE}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("transfer gap (oracle_r) on HELD-OUT cells")
    ax.set_title("T3a — how many pilot reservoirs are enough?", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    for k, ys in zip((1, 2, 3), data):
        ax.scatter([k] * len(ys), ys, color="black", alpha=0.4, s=15, zorder=10)

    # Panel 2: at N_pilot=2, which held-out reservoir is hardest?
    ax = axes[1]
    by_R = defaultdict(list)
    for r in rows:
        if r["n_pilot"] == 2 and not r["held_in_pilot"]:
            held_R = r["held_cell"].split("/")[0]
            by_R[held_R].append(r["gap"])
    labels = sorted(by_R)
    means = [np.mean(by_R[k]) for k in labels]
    stds = [np.std(by_R[k]) for k in labels]
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color="#9467bd", alpha=0.85)
    ax.axhline(NOISE, color="red", ls="--", lw=1, label=f"noise floor +{NOISE}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("gap on held-out reservoir family")
    ax.set_title(
        "T3b — which reservoir family is hardest to transfer to?\n(2-pilot setting)", fontsize=11
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    for b, m, s in zip(bars, means, stds):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + s + 0.001,
            f"{m:+.4f}",
            ha="center",
            fontsize=9,
        )

    # Panel 3: menu agreement: how often does each strategy appear in the menu across LOR splits
    ax = axes[2]
    menu_counts = defaultdict(int)
    n_splits = 0
    for r in rows:
        if r["n_pilot"] == 2:
            n_splits += 1
            for s in r["menu"]:
                menu_counts[s] += 1
    n_splits_unique = len({r["pilot_R"] for r in rows if r["n_pilot"] == 2})
    if menu_counts:
        items = sorted(menu_counts.items(), key=lambda x: -x[1])
        names, counts = zip(*items)
        ax.barh(names, counts, color="#1f77b4", alpha=0.85)
        ax.axvline(
            n_splits_unique,
            color="black",
            ls=":",
            lw=1,
            label=f"all {n_splits_unique} splits agreed",
        )
        ax.set_xlabel(f"# pilot-pair splits selecting this strategy (out of {n_splits_unique})")
        ax.set_title("T3c — menu stability across pilot subsets\n(at 2-pilot, K=5)", fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle(f"T3: cross-reservoir transfer — leave-one-out validation  (D={D})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT, "T3_reservoir_transfer.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"WROTE {out}")


if __name__ == "__main__":
    plot_t1()
    plot_t2()
    plot_t3()
