"""PI deck v2: time-normalized + 300k ranking + LOSO + deploy schematic.

Adds to the existing pi_deck:
  fig6_cum_best_300k_strat_only_K5.png  — clean cumulative-best at 300k for the
    K=5 deploy menu only (drops noise from non-menu strategies).
  fig7_strategy_ranking_300k.png        — bar chart of mean best val Pearson per
    strategy at D=300k (in-progress data).
  fig8_cum_best_vs_walltime_30k.png     — cumulative best val Pearson vs cumulative
    train wall time (FAIR cost comparison across strategies). LLM strategies
    look very different on this axis vs model-count.
  fig9_cum_best_vs_walltime_300k.png    — same at 300k.
  fig10_loso_weight_share.png           — bar chart per strategy of LOSO drop and
    ElasticNet weight share, from existing strategy_contribution analysis.
  fig11_deploy_procedure.png            — visual schematic of Stage 1/2/3 deploy
    methodology (pilot → menu → deploy).
"""

import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV

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
K5_COLOR = {
    "optuna_gp": "#2ca02c",
    "evo_batch": "#1f77b4",
    "llm_explore_nv1": "#d62728",
    "evo_single": "#9467bd",
    "optuna_tpe": "#ff7f0e",
}


def cell_meta(cd):
    """List of (round, val, train_time_s) sorted by round."""
    rows = []
    for m in sorted(glob.glob(os.path.join(cd, "r*_meta.json"))):
        try:
            d = json.load(open(m))
        except Exception:
            continue
        vp = d.get("val_pearson")
        rd = d.get("round")
        t = d.get("train_time_sec")
        if vp is None or rd is None or not np.isfinite(vp):
            continue
        rows.append((int(rd), float(vp), float(t or 0)))
    rows.sort()
    return rows


def collect(D, reservoirs, strats_filter=None):
    by_strat = defaultdict(dict)
    for R, seeds in reservoirs.items():
        for sd in seeds:
            for cd in sorted(glob.glob(os.path.join(ROOT, f"k562_{R}_d{D}", sd, "*"))):
                if not os.path.isdir(cd):
                    continue
                s = os.path.basename(cd)
                if strats_filter and s not in strats_filter:
                    continue
                rows = cell_meta(cd)
                if rows:
                    by_strat[s][f"{R}/{sd}"] = rows
    return by_strat


# ──────────────────────────────────────────────────────────────────────────
# fig6: 300k K=5 cumulative best (clean)
# ──────────────────────────────────────────────────────────────────────────
def fig6_300k_k5(out_path):
    by_strat = collect(300000, RESERVOIRS_300K, strats_filter=set(K5))
    fig, ax = plt.subplots(figsize=(14, 7.5))
    for s in K5:
        if s not in by_strat:
            continue
        color = K5_COLOR[s]
        curves = []
        for cell_id, rows in by_strat[s].items():
            y = np.maximum.accumulate(np.array([r[1] for r in rows]))
            x = np.arange(1, len(y) + 1)
            ax.plot(x, y, color=color, alpha=0.30, lw=1.0)
            curves.append(y)
        max_len = min(len(c) for c in curves) if curves else 0
        if max_len > 0:
            mean = np.mean([c[:max_len] for c in curves], axis=0)
            ax.plot(range(1, max_len + 1), mean, color=color, lw=4, marker="o", ms=6, label=s)
    ax.set_xlabel("models trained")
    ax.set_ylabel("best val Pearson (cumulative)")
    ax.set_title(
        "D=300,000 — cumulative best per K=5 deploy strategy\n"
        "(faint = per cell, bold = cross-cell mean)"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {out_path}")


# ──────────────────────────────────────────────────────────────────────────
# fig7: 300k strategy ranking
# ──────────────────────────────────────────────────────────────────────────
def fig7_ranking_300k(out_path):
    by_strat = collect(300000, RESERVOIRS_300K)
    rows = []
    for s, cells in by_strat.items():
        bests = []
        for cell_id, mods in cells.items():
            vp = [v for r, v, _ in mods]
            if vp:
                bests.append(max(vp))
        if bests:
            rows.append((s, np.mean(bests), np.std(bests), len(bests)))
    rows.sort(key=lambda x: -x[1])
    fig, ax = plt.subplots(figsize=(13, 6.5))
    names = [r[0] for r in rows]
    means = [r[1] for r in rows]
    stds = [r[2] for r in rows]
    colors = ["#2ca02c" if n in K5 else "#888" for n in names]
    ax.barh(names, means, xerr=stds, color=colors, alpha=0.85, capsize=4)
    ax.invert_yaxis()
    ax.set_xlabel("best val Pearson per cell (mean ± std)")
    ax.set_title(
        f"HP-search strategy ranking at D=300,000 (in-progress)\n"
        f"green = K=5 deploy menu  |  gray = dropped"
    )
    for i, (n, m, _, n_cells) in enumerate(rows):
        ax.text(m + 0.002, i, f"  n={n_cells}", va="center", fontsize=10, color="#555")
    ax.set_xlim(min(means) - 0.02, max(means) + 0.04)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {out_path}")


# ──────────────────────────────────────────────────────────────────────────
# fig8 / fig9: cumulative best vs WALL TIME (fair cost comparison)
# ──────────────────────────────────────────────────────────────────────────
def fig_cum_best_vs_walltime(D, reservoirs, out_path, top_strats_only=False):
    strats_filter = set(K5) if top_strats_only else None
    by_strat = collect(D, reservoirs, strats_filter=strats_filter)
    fig, ax = plt.subplots(figsize=(15, 7.5))

    color_pool = plt.get_cmap("tab20").colors
    strat_list = sorted(by_strat)
    for i, s in enumerate(strat_list):
        color = K5_COLOR.get(s, color_pool[i % len(color_pool)])
        in_k5 = s in K5
        curves = []
        for cell_id, rows in by_strat[s].items():
            ts = np.cumsum(np.array([r[2] for r in rows])) / 3600.0
            y = np.maximum.accumulate(np.array([r[1] for r in rows]))
            ax.plot(ts, y, color=color, alpha=0.18 if in_k5 else 0.10, lw=0.9)
            curves.append((ts, y))
        # Aggregate: resample each curve to a common time grid
        usable = [(ts, y) for ts, y in curves if len(ts) >= 2 and ts[-1] > 0]
        if usable:
            t_max = min(ts[-1] for ts, _ in usable)
            if t_max > 0:
                tgrid = np.linspace(0.01, t_max, 60)
                ys = np.array([np.interp(tgrid, ts, y) for ts, y in usable])
                if ys.ndim == 2 and ys.shape[1] == len(tgrid):
                    mean = np.nanmean(ys, axis=0)
                    ax.plot(
                        tgrid,
                        mean,
                        color=color,
                        lw=3.5 if in_k5 else 1.6,
                        alpha=1.0 if in_k5 else 0.7,
                        label=f"{s}{' ★' if in_k5 else ''}",
                    )

    ax.set_xlabel("cumulative GPU-hours used")
    ax.set_ylabel("best val Pearson (cumulative)")
    title = f"Cumulative best val Pearson vs WALL TIME — D={D:,}"
    if top_strats_only:
        title += "  (K=5 menu only)"
    title += "\nfair cost comparison: LLM/evo_batch propose multiple configs/round but train sequentially"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", ncol=2, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {out_path}")


# ──────────────────────────────────────────────────────────────────────────
# fig10: LOSO + weight-share per strategy
# ──────────────────────────────────────────────────────────────────────────
def fig10_loso_weight(out_path):
    """Compute LOSO + weight-share at 30k from existing data."""
    BUDGET, K = 75, 5
    by_R = {}
    for R, seeds in RESERVOIRS_30K.items():
        by_R[R] = []
        for sd in seeds:
            cells = sorted(
                d
                for d in glob.glob(os.path.join(ROOT, f"k562_{R}_d30000", sd, "*"))
                if os.path.isdir(d)
            )
            if not cells:
                continue
            lab = np.load(os.path.join(cells[0], "labels.npz"))
            vy, oy = lab["val_labels"], lab["test_oracle"]
            cols, who = [], []
            for cd in cells:
                strat = os.path.basename(cd)
                metas = []
                for m in sorted(glob.glob(os.path.join(cd, "r*_meta.json"))):
                    try:
                        md = json.load(open(m))
                    except Exception:
                        continue
                    if md.get("val_pearson") is None:
                        continue
                    metas.append((int(md.get("round", -1)), float(md["val_pearson"]), m))
                metas.sort()
                metas = sorted(metas[:BUDGET], key=lambda r: -r[1])[:K]
                for _, _, m in metas:
                    try:
                        z = np.load(m.replace("_meta.json", ".npz"))
                    except Exception:
                        continue
                    if z["val_pred"].shape != vy.shape or z["test_pred"].shape != oy.shape:
                        continue
                    if not (
                        np.all(np.isfinite(z["val_pred"])) and np.all(np.isfinite(z["test_pred"]))
                    ):
                        continue
                    cols.append((z["val_pred"], z["test_pred"]))
                    who.append(strat)
            if cols:
                by_R[R].append(
                    (
                        np.array([c[0] for c in cols]).T,
                        np.array([c[1] for c in cols]).T,
                        vy,
                        oy,
                        np.array(who),
                    )
                )

    all_strats = sorted({s for L in by_R.values() for _, _, _, _, w in L for s in w})
    weight, loso = defaultdict(list), defaultdict(list)
    for R, L in by_R.items():
        for V, T, vy, oy, who in L:
            en = ElasticNetCV(
                l1_ratio=[0.5, 0.9, 0.95, 1.0], positive=True, cv=5, max_iter=20000, n_jobs=1
            )
            en.fit(V, vy)
            r_full = float(pearsonr(en.predict(T), oy)[0])
            l1r, alpha = float(en.l1_ratio_), float(en.alpha_)
            tot = en.coef_.sum() or 1.0
            for s in all_strats:
                mask = who == s
                if not mask.any():
                    weight[s].append(0.0)
                    loso[s].append(0.0)
                    continue
                weight[s].append(float(en.coef_[mask].sum() / tot * 100))
                from sklearn.linear_model import ElasticNet

                en2 = ElasticNet(l1_ratio=l1r, alpha=alpha, positive=True, max_iter=20000)
                en2.fit(V[:, ~mask], vy)
                r_minus = float(pearsonr(en2.predict(T[:, ~mask]), oy)[0])
                loso[s].append(float(r_full - r_minus))

    rows = sorted(all_strats, key=lambda s: -np.mean(loso[s]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    names = rows
    loso_m = [np.mean(loso[s]) for s in names]
    w_m = [np.mean(weight[s]) for s in names]
    colors = ["#2ca02c" if n in K5 else "#888" for n in names]
    ax1.barh(names, loso_m, color=colors, alpha=0.85)
    ax1.invert_yaxis()
    ax1.axvline(0, color="k", lw=0.5)
    ax1.set_xlabel("LOSO oracle_r drop\n(leave-one-strategy-out from ensemble)")
    ax1.set_title("Strategy CONTRIBUTION to ensemble (decision metric)", fontsize=14)
    ax1.grid(axis="x", alpha=0.3)
    ax2.barh(names, w_m, color=colors, alpha=0.85)
    ax2.invert_yaxis()
    ax2.set_xlabel("ElasticNet weight share (%)")
    ax2.set_title("Strategy weight share (diagnostic only)", fontsize=14)
    ax2.grid(axis="x", alpha=0.3)
    fig.suptitle(
        "Strategy contribution: LOSO drives K=5 selection; weight-share diagnoses redundancy",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {out_path}")


# ──────────────────────────────────────────────────────────────────────────
# fig11: deploy procedure schematic
# ──────────────────────────────────────────────────────────────────────────
def fig11_deploy_schematic(out_path):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    stages = [
        (
            "STAGE 1\nPilot (once per D)",
            "Run 18 HP-search strategies\non 3 reservoirs × 3 seeds\n~8,000 models pooled per D",
            "#e3eaf2",
            "#1f77b4",
        ),
        (
            "STAGE 2\nMenu selection (once per D)",
            "Cross-reservoir greedy on\n(mean oracle_r across cells) →\nK=5 strategies LOCKED",
            "#e3f6e3",
            "#2ca02c",
        ),
        (
            "STAGE 3\nDeploy (per R × A cell)",
            "Train K=5 strats on this cell\n→ pool models → greedy K=5-8\n→ ElasticNetCV ensemble",
            "#fff0e0",
            "#ff7f0e",
        ),
    ]
    for i, (title, body, fc, ec) in enumerate(stages):
        cx = 2 + 5 * i
        ax.add_patch(
            patches.FancyBboxPatch(
                (cx - 2.1, 4.5), 4.2, 4.2, boxstyle="round,pad=0.1", fc=fc, ec=ec, lw=2.5
            )
        )
        ax.text(cx, 7.7, title, ha="center", va="center", fontsize=15, fontweight="bold", color=ec)
        ax.text(cx, 5.7, body, ha="center", va="center", fontsize=12, color="#333")
        if i < 2:
            ax.annotate(
                "",
                xy=(cx + 2.6, 6.5),
                xytext=(cx + 2.1, 6.5),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#555"),
            )

    # Key empirical findings as a strip
    ax.text(
        8,
        3.3,
        "Empirical validations (from 30k bake-off)",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color="#333",
    )
    facts = [
        ("K=5 strats cover all-18", "T1 median gap = +0.0009  (sub-noise)"),
        ("5-8 models is the natural ensemble size", "T2c: greedy K=5-8 matches/beats full pool"),
        (
            "Menu transfers across reservoirs",
            "T3 menu Jaccard = 1.0 for 4/5 strats across LOR splits",
        ),
        (
            "Total compute saving vs per-cell HP search",
            "~36x cheaper (~9.6k vs ~345k deploy models)",
        ),
    ]
    for i, (claim, evidence) in enumerate(facts):
        y = 2.4 - i * 0.55
        ax.text(0.5, y, "✓", fontsize=14, color="#2ca02c", fontweight="bold")
        ax.text(1.0, y, claim, fontsize=12, color="#222")
        ax.text(8.5, y, evidence, fontsize=11, color="#555", style="italic")

    fig.suptitle(
        "Deploy procedure — single 3-reservoir pilot per D, locked menu, light per-cell deploy",
        y=0.98,
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    fig6_300k_k5(os.path.join(OUT, "fig6_cum_best_300k_K5_only.png"))
    fig7_ranking_300k(os.path.join(OUT, "fig7_strategy_ranking_300k.png"))
    fig_cum_best_vs_walltime(
        30000, RESERVOIRS_30K, os.path.join(OUT, "fig8_cum_best_vs_walltime_30k.png")
    )
    fig_cum_best_vs_walltime(
        300000, RESERVOIRS_300K, os.path.join(OUT, "fig9_cum_best_vs_walltime_300k.png")
    )
    fig10_loso_weight(os.path.join(OUT, "fig10_loso_weight_share.png"))
    fig11_deploy_schematic(os.path.join(OUT, "fig11_deploy_procedure.png"))
    print("DONE")
