"""PI-meeting plot pack. Reads r*_meta.json from outputs/hp_step1_bakeoff_e100
and writes presentation-grade plots to PI_OUT (default ./outputs/analysis/pi_deck).

Plots:
  fig1_cum_best_30k_by_family.png  — cumulative best val_pearson per strategy at 30k,
    faceted by strategy family (evo / optuna / llm / random+ray), one line per cell
    + mean across cells. Justifies how many rounds each family needs.
  fig2_cum_best_300k.png            — same at D=300k.
  fig3_strategy_ranking_30k.png     — bar chart of mean best val_pearson per strategy
    at 30k, K=5 menu highlighted. Why these 5.
  fig4_ensemble_size.png            — from validation_t2c.json: ensemble oracle_r vs
    K (# greedy-picked models), showing 4-8 is the natural plateau.
  fig5_bs_schedule.png              — D-aware batch_size menu + B_crit fit.
"""

import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Presentation-grade text sizes
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
RESERVOIRS_300K = {
    "genomic": ["seed42_0", "seed43_1", "seed44_2"],
}

FAMILY = {
    "random": "random/baseline",
    "ray_asha": "random/baseline",
    "ray_bohb": "random/baseline",
    "optuna_tpe": "optuna (BO)",
    "optuna_gp": "optuna (BO)",
    "optuna_cmaes": "optuna (BO)",
    "optuna_qmc": "optuna (BO)",
    "evo_single": "evolutionary",
    "evo_batch": "evolutionary",
    "evo_explore": "evolutionary",
    "evo_exploit": "evolutionary",
    "evo_adaptive": "evolutionary",
    "evo_massive": "evolutionary",
    "evo_knowledgeable": "evolutionary",
    "llm_explore_nv1": "LLM (Claude)",
    "llm_diverse_nv1": "LLM (Claude)",
    "llm_exploit_nv1": "LLM (Claude)",
    "llm_critic_nv0": "LLM (Claude)",
}

STRAT_COLOR = {
    "optuna_gp": "#2ca02c",
    "evo_batch": "#1f77b4",
    "llm_explore_nv1": "#d62728",
    "evo_single": "#9467bd",
    "optuna_tpe": "#ff7f0e",
    "llm_diverse_nv1": "#e377c2",
    "evo_adaptive": "#17becf",
    "evo_exploit": "#7f7f7f",
    "evo_explore": "#bcbd22",
    "evo_massive": "#8c564b",
    "evo_knowledgeable": "#aec7e8",
    "optuna_cmaes": "#ffbb78",
    "optuna_qmc": "#98df8a",
    "random": "#c5b0d5",
    "ray_asha": "#c49c94",
    "ray_bohb": "#f7b6d3",
    "llm_critic_nv0": "#dbdb8d",
    "llm_exploit_nv1": "#9edae5",
}

K5_MENU = ["optuna_gp", "evo_batch", "llm_explore_nv1", "evo_single", "optuna_tpe"]


def cell_models(cd):
    """Return list of (round, val_pearson) for this cell sorted by round."""
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
        rows.append((int(rd), float(vp)))
    rows.sort()
    return rows


def collect(D, reservoirs):
    """Returns dict[strat] -> dict[cell_id] -> list of (round, val_pearson)."""
    by_strat = defaultdict(dict)
    for R, seeds in reservoirs.items():
        for sd in seeds:
            for cd in sorted(glob.glob(os.path.join(ROOT, f"k562_{R}_d{D}", sd, "*"))):
                if not os.path.isdir(cd):
                    continue
                s = os.path.basename(cd)
                rows = cell_models(cd)
                if rows:
                    by_strat[s][f"{R}/{sd}"] = rows
    return by_strat


def cum_max(rows):
    """Given list of (round, val), return arrays of x (model index 1..N) and
    cumulative max val by model number."""
    if not rows:
        return np.array([]), np.array([])
    y = [r[1] for r in rows]
    return np.arange(1, len(y) + 1), np.maximum.accumulate(np.array(y))


def fig1_cum_best_by_family(D, reservoirs, out_path):
    by_strat = collect(D, reservoirs)
    families = ["evolutionary", "optuna (BO)", "LLM (Claude)", "random/baseline"]
    family_strats = {f: [s for s in by_strat if FAMILY.get(s) == f] for f in families}

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey=True)
    axes = axes.reshape(-1)
    for ax, fam in zip(axes, families):
        strats = sorted(family_strats[fam])
        for s in strats:
            cells = by_strat[s]
            if not cells:
                continue
            color = STRAT_COLOR.get(s, "#888")
            # Per-cell cumulative max
            curves = []
            for cell_id, rows in cells.items():
                x, y = cum_max(rows)
                if len(x):
                    ax.plot(x, y, color=color, alpha=0.18, lw=0.9)
                    curves.append((x, y))
            # Mean curve at min-length
            if curves:
                max_len = min(len(c[0]) for c in curves)
                if max_len > 0:
                    mean = np.mean([c[1][:max_len] for c in curves], axis=0)
                    in_k5 = s in K5_MENU
                    ax.plot(
                        range(1, max_len + 1),
                        mean,
                        color=color,
                        lw=3.5 if in_k5 else 2.0,
                        alpha=1.0 if in_k5 else 0.7,
                        marker="o" if in_k5 else None,
                        ms=4,
                        label=f"{s}{' ★' if in_k5 else ''}",
                    )
        ax.set_title(fam)
        ax.set_xlabel("models trained")
        ax.set_ylabel("best val Pearson (cumulative)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=10, ncol=1)

    fig.suptitle(
        f"Cumulative best val Pearson per HP-search strategy — D={D:,}\n★ = in the K=5 deploy menu",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {out_path}")


def fig3_strategy_ranking(D, reservoirs, out_path):
    by_strat = collect(D, reservoirs)
    rows = []
    for s, cells in by_strat.items():
        # mean across cells of (best val_pearson at first 75 rounds)
        bests = []
        for cell_id, mods in cells.items():
            vp = [v for r, v in mods if r < 75]
            if vp:
                bests.append(max(vp))
        if bests:
            rows.append((s, np.mean(bests), np.std(bests), len(bests)))
    rows.sort(key=lambda x: -x[1])

    fig, ax = plt.subplots(figsize=(14, 7))
    names = [r[0] for r in rows]
    means = [r[1] for r in rows]
    stds = [r[2] for r in rows]
    colors = ["#2ca02c" if n in K5_MENU else "#888" for n in names]
    ax.barh(names, means, xerr=stds, color=colors, alpha=0.85, capsize=4)
    ax.invert_yaxis()
    ax.set_xlabel("best val Pearson per cell (mean ± std across cells)")
    ax.set_title(
        f"HP-search strategy ranking at D={D:,}\n"
        f"green = chosen in K=5 deploy menu  |  gray = dropped"
    )
    # Annotate K=5 marker
    for i, (n, m, _, n_cells) in enumerate(rows):
        ax.text(m + 0.005, i, f"  n={n_cells}", va="center", fontsize=10, color="#555")
    ax.set_xlim(min(means) - 0.03, max(means) + 0.05)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {out_path}")


def fig4_ensemble_size(t2c_path, out_path):
    if not os.path.exists(t2c_path):
        print(f"  [skip] {t2c_path} not found; cannot make ensemble-size plot")
        return
    rows = json.load(open(t2c_path))
    # Build distribution of gaps per k vs full pool (negative gap = greedy beats full)
    gaps = defaultdict(list)
    abs_oracle = defaultdict(list)
    abs_full = []
    for r in rows:
        full = r["full_pool"]
        if not np.isfinite(full):
            continue
        abs_full.append(full)
        for k, val in enumerate(r["curve"], start=1):
            if np.isfinite(val):
                gaps[k].append(full - val)
                abs_oracle[k].append(val)
    ks = sorted(gaps)
    medians = [np.median(abs_oracle[k]) for k in ks]
    iqr_lo = [np.percentile(abs_oracle[k], 25) for k in ks]
    iqr_hi = [np.percentile(abs_oracle[k], 75) for k in ks]
    full_med = float(np.median(abs_full))

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.fill_between(
        ks, iqr_lo, iqr_hi, color="#1f77b4", alpha=0.25, label="IQR across held-out cells"
    )
    ax.plot(
        ks, medians, "o-", color="#1f77b4", lw=3, ms=10, label="median greedy ensemble oracle_r"
    )
    ax.axhline(
        full_med, color="#d62728", lw=2.5, ls="--", label=f"K=5 strategies' FULL pool (~100 models)"
    )
    # Plateau band
    ax.axvspan(4, 8, alpha=0.10, color="green", label="natural plateau: 4-8 models")
    ax.set_xlabel("# greedily-picked ensemble models")
    ax.set_ylabel("ensemble oracle Pearson on held-out cell")
    ax.set_xticks(ks)
    ax.set_title(
        "Deploy ensemble size — performance saturates at 4-8 models\n"
        "greedy-K=5 already matches or beats the full ~100-model pool"
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {out_path}")


def fig5_bs_schedule(out_path):
    SCHEDULE = [
        (10_000, [128, 256, 512, 1024]),
        (30_000, [128, 256, 512, 1024]),
        (100_000, [256, 512, 1024, 2048]),
        (300_000, [256, 512, 1024, 2048]),
        (1_000_000, [512, 1024, 2048, 4096]),
        (3_000_000, [512, 1024, 2048, 4096]),
    ]
    ANCHORS = [(30_000, 512), (300_000, 1024)]
    ALPHA = 0.301
    fig, ax = plt.subplots(figsize=(13, 7))
    Dgrid = np.logspace(np.log10(5_000), np.log10(5_000_000), 200)
    ax.plot(
        Dgrid,
        [512 * (D / 30_000) ** ALPHA for D in Dgrid],
        "k-",
        lw=2.5,
        alpha=0.85,
        label=r"empirical $B_{\rm crit}(D) \propto D^{0.301}$",
    )
    for D, b in ANCHORS:
        ax.plot(D, b, "ko", ms=14, zorder=5)
        ax.annotate(f"  measured B_crit={b}", (D, b), fontsize=12, va="center")
    cmap = plt.get_cmap("viridis")
    for i, (D, menu) in enumerate(SCHEDULE):
        color = cmap(i / (len(SCHEDULE) - 1))
        ax.vlines(D, min(menu), max(menu), colors=color, lw=14, alpha=0.30)
        for bs in menu:
            ax.plot(
                D,
                bs,
                "o",
                color=color,
                ms=11,
                zorder=4,
                markeredgecolor="white",
                markeredgewidth=1.2,
            )
    ax.set_xscale("log")
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Dataset size  D")
    ax.set_ylabel("batch_size menu")
    ax.set_title(
        "D-aware batch_size menu — derived from empirical $B_{\\rm crit}$\n"
        "menus span ¼·B_crit → 2·B_crit, stepped 2× per D-decade"
    )
    ax.set_yticks([16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    ax.set_yticklabels([16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_xlim(5_000, 5_000_000)
    ax.set_ylim(48, 5500)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    print("Building PI deck ...", flush=True)
    fig1_cum_best_by_family(30000, RESERVOIRS_30K, os.path.join(OUT, "fig1_cum_best_30k.png"))
    fig2 = os.path.join(OUT, "fig2_cum_best_300k.png")
    fig1_cum_best_by_family(300000, RESERVOIRS_300K, fig2)
    fig3_strategy_ranking(30000, RESERVOIRS_30K, os.path.join(OUT, "fig3_strategy_ranking_30k.png"))
    fig4_ensemble_size(
        "outputs/analysis/validation_suite/validation_t2c.json",
        os.path.join(OUT, "fig4_ensemble_size_5to8.png"),
    )
    fig5_bs_schedule(os.path.join(OUT, "fig5_bs_schedule.png"))
    print("DONE")
