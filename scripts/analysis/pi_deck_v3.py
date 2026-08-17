"""PI deck v3: time-normalized ranking, batch_size performance table,
val/test paired curves.

  fig12_time_normalized_ranking.png  — rank strategies by best val_pearson
    achieved within a fixed GPU-hour budget. Bars at 5h, 10h, 20h, 50h.
  fig13_bs_perf_table_30k.png        — table of best val Pearson at each
    (D, bs) bin with model counts.
  fig13_bs_perf_table_300k.png       — same at 300k.
  fig14_val_vs_test_curves_30k.png   — paired cumulative-best curves: val
    (per-cell holdout, distribution-specific) vs test (canonical genomic
    chr-test holdout, identical across reservoirs). Shows the val inflation
    for synthetic-reservoir cells.
  fig14_val_vs_test_curves_300k.png  — same at 300k.
"""

import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 18,
        "axes.titleweight": "bold",
    }
)

ROOT = "outputs/hp_step1_bakeoff_e100"
OUT = os.environ.get("PI_OUT", "outputs/analysis/pi_deck")
os.makedirs(OUT, exist_ok=True)

RESERVOIRS_30K = {
    "genomic": ["seed42_0", "seed43_1", "seed44_2"],
    "motif_planted_v2": ["seed42_0", "seed43_1"],
    "dinuc_shuffle": ["seed42_0", "seed43_1"],
}
RESERVOIRS_300K = {"genomic": ["seed42_0", "seed43_1", "seed44_2"]}
K5 = ["optuna_gp", "evo_batch", "llm_explore_nv1", "evo_single", "optuna_tpe"]
FAMILY = {
    "random": "baseline",
    "ray_asha": "baseline",
    "ray_bohb": "baseline",
    "optuna_tpe": "optuna",
    "optuna_gp": "optuna",
    "optuna_cmaes": "optuna",
    "optuna_qmc": "optuna",
    "evo_single": "evo",
    "evo_batch": "evo",
    "evo_explore": "evo",
    "evo_exploit": "evo",
    "evo_adaptive": "evo",
    "evo_massive": "evo",
    "evo_knowledgeable": "evo",
    "llm_explore_nv1": "llm",
    "llm_diverse_nv1": "llm",
    "llm_exploit_nv1": "llm",
    "llm_critic_nv0": "llm",
}


def cell_meta_full(cd):
    """List of (round, val_p, test_p_genomic, train_time_s, batch_size)."""
    rows = []
    for m in sorted(glob.glob(os.path.join(cd, "r*_meta.json"))):
        try:
            d = json.load(open(m))
        except Exception:
            continue
        vp = d.get("val_pearson")
        rd = d.get("round")
        if vp is None or rd is None or not np.isfinite(vp):
            continue
        test_p = d.get("per_set_metrics", {}).get("genomic", {}).get("pearson", np.nan)
        if not isinstance(test_p, (int, float)) or not np.isfinite(test_p):
            test_p = np.nan
        t = d.get("train_time_sec") or 0
        bs = d.get("hp", {}).get("batch_size")
        rows.append((int(rd), float(vp), float(test_p), float(t), bs))
    rows.sort()
    return rows


def collect(D, reservoirs):
    by_strat = defaultdict(dict)
    for R, seeds in reservoirs.items():
        for sd in seeds:
            for cd in sorted(glob.glob(os.path.join(ROOT, f"k562_{R}_d{D}", sd, "*"))):
                if not os.path.isdir(cd):
                    continue
                s = os.path.basename(cd)
                rows = cell_meta_full(cd)
                if rows:
                    by_strat[s][f"{R}/{sd}"] = rows
    return by_strat


# ──────────────────────────────────────────────────────────────────────────
# fig12: time-normalized strategy ranking at multiple wall-time budgets
# ──────────────────────────────────────────────────────────────────────────
def fig12_time_normalized(D, reservoirs, out_path, budgets_h=(2.5, 5, 10, 20)):
    by_strat = collect(D, reservoirs)
    # For each strategy×cell, compute best val_pearson achieved by cumulative
    # train_time <= budget. Average across cells.
    table = {b: {} for b in budgets_h}
    for s, cells in by_strat.items():
        for b in budgets_h:
            bests = []
            for cell_id, rows in cells.items():
                cum_t = 0.0
                best_v = np.nan
                for rd, vp, _, t, _ in rows:
                    cum_t += t
                    if cum_t / 3600.0 <= b:
                        if np.isnan(best_v) or vp > best_v:
                            best_v = vp
                    else:
                        break
                if np.isfinite(best_v):
                    bests.append(best_v)
            if bests:
                table[b][s] = (float(np.mean(bests)), float(np.std(bests)), len(bests))

    strats = sorted(
        {s for b in table.values() for s in b}, key=lambda s: -table[max(budgets_h)].get(s, (0,))[0]
    )
    fig, ax = plt.subplots(figsize=(15, max(6, 0.45 * len(strats))))
    y_pos = np.arange(len(strats))
    width = 0.20
    cmap = plt.get_cmap("viridis")
    for i, b in enumerate(budgets_h):
        means = [table[b].get(s, (np.nan,))[0] for s in strats]
        stds = [table[b].get(s, (0, 0))[1] if s in table[b] else 0 for s in strats]
        ax.barh(
            y_pos + (i - 1.5) * width,
            means,
            xerr=stds,
            height=width,
            color=cmap(i / max(1, len(budgets_h) - 1)),
            alpha=0.85,
            label=f"{b}h budget",
            capsize=2.5,
        )
    ax.set_yticks(y_pos)
    labels_with_star = [f"{s} ★" if s in K5 else s for s in strats]
    ax.set_yticklabels(labels_with_star)
    ax.invert_yaxis()
    ax.set_xlabel("best val Pearson within budget (mean across cells)")
    ax.set_title(
        f"TIME-NORMALIZED strategy ranking — D={D:,}\n"
        f"★ = K=5 deploy menu  |  fair comparison: same GPU-hours, not same #rounds"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {out_path}")


# ──────────────────────────────────────────────────────────────────────────
# fig13: batch_size performance table
# ──────────────────────────────────────────────────────────────────────────
def fig13_bs_table(D, reservoirs, out_path):
    by_strat = collect(D, reservoirs)
    # Per bs: best val, mean val, p90 train_time, count
    by_bs = defaultdict(list)
    for s, cells in by_strat.items():
        for cell_id, rows in cells.items():
            for rd, vp, test_p, t, bs in rows:
                if bs is None:
                    continue
                by_bs[int(bs)].append((vp, test_p, t))

    bss = sorted(by_bs)
    cell_text = []
    for bs in bss:
        vp = np.array([r[0] for r in by_bs[bs] if np.isfinite(r[0])])
        tp = np.array([r[1] for r in by_bs[bs] if np.isfinite(r[1])])
        t = np.array([r[2] for r in by_bs[bs] if r[2] > 0])
        cell_text.append(
            [
                f"{bs}",
                f"{len(vp)}",
                f"{np.max(vp):.4f}" if len(vp) else "—",
                f"{np.mean(vp):.4f}" if len(vp) else "—",
                f"{np.max(tp):.4f}" if len(tp) else "—",
                f"{np.mean(tp):.4f}" if len(tp) else "—",
                f"{np.median(t):.0f}" if len(t) else "—",
                f"{np.percentile(t, 90):.0f}" if len(t) else "—",
            ]
        )

    fig, ax = plt.subplots(figsize=(13, 0.5 + 0.4 * (len(bss) + 1)))
    ax.axis("off")
    cols = [
        "batch_size",
        "n_models",
        "best val",
        "mean val",
        "best test\n(genomic chr-test)",
        "mean test",
        "median time (s)",
        "p90 time (s)",
    ]
    table = ax.table(
        cellText=cell_text,
        colLabels=cols,
        loc="center",
        cellLoc="center",
        colColours=["#e3eaf2"] * len(cols),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    # Highlight rows in the locked D-aware menu
    menu = {30000: [128, 256, 512, 1024], 300000: [256, 512, 1024, 2048]}.get(D, [])
    for i, bs in enumerate(bss):
        if bs in menu:
            for j in range(len(cols)):
                table[(i + 1, j)].set_facecolor("#e3f6e3")
    ax.set_title(
        f"Batch-size performance — D={D:,}   (green rows = locked D-aware menu)\n"
        f"best val = same-distribution holdout, best test = canonical chr-test "
        f"holdout (genomic OOD for synthetic reservoirs)",
        fontsize=14,
        pad=20,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {out_path}")


# ──────────────────────────────────────────────────────────────────────────
# fig14: val vs test paired curves
# ──────────────────────────────────────────────────────────────────────────
def fig14_val_vs_test(D, reservoirs, out_path):
    by_strat = collect(D, reservoirs)
    # Per reservoir family: cumulative best val and cumulative best test
    # (per_set_metrics["genomic"]["pearson"]), aggregated across cells of that R.
    families_R = sorted(reservoirs)
    if len(families_R) == 1:
        nrows, ncols = 1, 1
        figsize = (12, 6)
    else:
        nrows, ncols = 1, len(families_R)
        figsize = (6 * ncols, 6)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    elif hasattr(axes, "shape"):
        axes = axes.reshape(-1).tolist()
    color_map = {
        "optuna_gp": "#2ca02c",
        "evo_batch": "#1f77b4",
        "llm_explore_nv1": "#d62728",
        "evo_single": "#9467bd",
        "optuna_tpe": "#ff7f0e",
    }
    for ax, R in zip(axes, families_R):
        seeds = reservoirs[R]
        for s in K5:
            color = color_map[s]
            curves_v, curves_t = [], []
            for sd in seeds:
                cell_id = f"{R}/{sd}"
                rows = by_strat.get(s, {}).get(cell_id, [])
                if not rows:
                    continue
                vp = np.maximum.accumulate(np.array([r[1] for r in rows]))
                tp_raw = np.array([r[2] for r in rows])
                # cumulative max ignoring NaN
                tp = np.array(tp_raw)
                cur = -np.inf
                for i, v in enumerate(tp_raw):
                    if np.isfinite(v) and v > cur:
                        cur = v
                    tp[i] = cur if np.isfinite(cur) else np.nan
                curves_v.append(vp)
                curves_t.append(tp)
            if curves_v:
                ml = min(len(c) for c in curves_v)
                if ml > 0:
                    mean_v = np.mean([c[:ml] for c in curves_v], axis=0)
                    mean_t = np.nanmean([c[:ml] for c in curves_t], axis=0)
                    x = np.arange(1, ml + 1)
                    ax.plot(x, mean_v, color=color, lw=3, label=f"{s} VAL")
                    ax.plot(x, mean_t, color=color, lw=3, ls="--", label=f"{s} TEST (genomic)")
        ax.set_xlabel("models trained")
        ax.set_title(R)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9, ncol=2)
    axes[0].set_ylabel("Pearson (cumulative best)")
    fig.suptitle(
        f"Val vs Test paired curves — D={D:,}\n"
        f"solid = own holdout (val), dashed = canonical chr-test (genomic OOD for synthetic R)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    fig12_time_normalized(
        30000, RESERVOIRS_30K, os.path.join(OUT, "fig12_time_normalized_ranking_30k.png")
    )
    fig12_time_normalized(
        300000,
        RESERVOIRS_300K,
        os.path.join(OUT, "fig12_time_normalized_ranking_300k.png"),
        budgets_h=(2.5, 5, 10, 20),
    )
    fig13_bs_table(30000, RESERVOIRS_30K, os.path.join(OUT, "fig13_bs_perf_table_30k.png"))
    fig13_bs_table(300000, RESERVOIRS_300K, os.path.join(OUT, "fig13_bs_perf_table_300k.png"))
    fig14_val_vs_test(30000, RESERVOIRS_30K, os.path.join(OUT, "fig14_val_vs_test_30k.png"))
    fig14_val_vs_test(300000, RESERVOIRS_300K, os.path.join(OUT, "fig14_val_vs_test_300k.png"))
    print("DONE")
