"""Test 2b (revised): does restricting the ensemble to TOP-N models per K=5
strategy generalize better than using ALL intermediates from those strategies?

Leave-one-reservoir-out: pilot on 2 reservoirs to select the K=5 strategy menu,
then in held-out cells compare ensemble oracle_r at increasing restriction:
  - K5_top1   : 5 models     (1 per strategy)
  - K5_top3   : 15 models    (3 per strategy)
  - K5_top5   : 25 models    (5 per strategy)
  - K5_top10  : 50 models    (10 per strategy)
  - K5_full   : ALL available models from K=5 strats (≤ TOPN_FOR_FULL per cell)
  - all18_full: ALL models from all 18 strats — kitchen-sink reference

If K5_full ≈ K5_top1: restriction is FREE (use 5 models). Deploy with K=5 configs.
If K5_full ≫ K5_top1, K5_top3 closes the gap: hedge with 3 per strat (15 models).
If K5_full > all_top tiers consistently: restriction HURTS — must run strategies."""

import glob
import itertools
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV

E100 = os.environ.get("VS_ROOT", "outputs/hp_step1_bakeoff_e100")
D = os.environ.get("VS_D", "30000")
RESERVOIRS = {
    "genomic": ["seed42_0", "seed43_1", "seed44_2"],
    "motif_planted_v2": ["seed42_0", "seed43_1"],
    "dinuc_shuffle": ["seed42_0", "seed43_1"],
}
BUDGET = int(os.environ.get("VS_BUDGET", "75"))
TOPN_FOR_FULL = int(os.environ.get("VS_TOPN_FULL", "20"))  # need enough headroom for top-10
OUT_DIR = os.environ.get("VS_OUT", "outputs/analysis/validation_suite")


def cell_topk(cd, n):
    rows = []
    for m in sorted(glob.glob(os.path.join(cd, "r*_meta.json"))):
        try:
            d = json.load(open(m))
        except Exception:
            continue
        vp = d.get("val_pearson")
        if vp is None:
            continue
        rows.append((int(d.get("round", -1)), float(vp), m))
    rows.sort()
    return sorted(rows[:BUDGET], key=lambda r: -r[1])[:n]


def load_cell(reservoir, seed):
    sd = os.path.join(E100, f"k562_{reservoir}_d{D}", seed)
    cells = sorted(d for d in glob.glob(os.path.join(sd, "*")) if os.path.isdir(d))
    if not cells:
        return None
    lab = np.load(os.path.join(cells[0], "labels.npz"))
    vy, oy = lab["val_labels"], lab["test_oracle"]
    by_strat = {}
    for cd in cells:
        s = os.path.basename(cd)
        for _, vp, m in cell_topk(cd, TOPN_FOR_FULL):
            try:
                z = np.load(m.replace("_meta.json", ".npz"))
            except Exception:
                continue
            if z["val_pred"].shape != vy.shape or z["test_pred"].shape != oy.shape:
                continue
            by_strat.setdefault(s, []).append((z["val_pred"], z["test_pred"], vp))
    return (reservoir, seed, by_strat, vy, oy) if by_strat else None


def ens(cols, vy, oy):
    if not cols:
        return np.nan, 0
    V = np.array([c[0] for c in cols]).T
    T = np.array([c[1] for c in cols]).T
    en = ElasticNetCV(l1_ratio=[0.5, 0.9, 0.95, 1.0], positive=True, cv=5, max_iter=20000, n_jobs=1)
    en.fit(V, vy)
    return float(pearsonr(en.predict(T), oy)[0]), V.shape[1]


def top_n_per_strat(cell, strats, n):
    """Take top-n models per strategy (by val_pearson). If a strategy has fewer
    than n models, return all of them. Returns list of (val_pred, test_pred)."""
    _, _, by_strat, _, _ = cell
    out = []
    for s in strats:
        items = sorted(by_strat.get(s, []), key=lambda x: -x[2])[:n]
        out.extend([(v, t) for v, t, _ in items])
    return out


def all_models(cell, strats=None):
    """All available models from given strats (or all strats if None)."""
    _, _, by_strat, _, _ = cell
    out = []
    src = strats if strats is not None else list(by_strat)
    for s in src:
        for v, t, _ in by_strat.get(s, []):
            out.append((v, t))
    return out


def cross_cell_greedy(cells, candidates, max_K):
    """Greedy at strategy level on cross-cell mean (using top-1 per strat to score)."""
    selected = []
    for _ in range(min(max_K, len(candidates))):
        best, best_s = -np.inf, None
        for cand in candidates:
            if cand in selected:
                continue
            scores = []
            for c in cells:
                cols = top_n_per_strat(c, selected + [cand], n=1)
                if cols:
                    r, _ = ens(cols, c[3], c[4])
                    if np.isfinite(r):
                        scores.append(r)
            if not scores:
                continue
            m = np.nanmean(scores)
            if m > best:
                best, best_s = m, cand
        if best_s is None:
            break
        selected.append(best_s)
    return selected


def all_cells():
    cells = []
    for R, seeds in RESERVOIRS.items():
        for sd in seeds:
            c = load_cell(R, sd)
            if c:
                cells.append(c)
    return cells, sorted({s for _, _, bs, _, _ in cells for s in bs})


def t2b_restriction(cells, all_strats, max_K=5):
    """For each LOR split (2 pilot Rs, 1 held-out R), in each held-out cell,
    score ensembles at multiple restriction levels."""
    R_list = sorted({c[0] for c in cells})
    out = []
    tiers_n = [1, 3, 5, 10, "full"]
    for pilot_R in itertools.combinations(R_list, 2):
        pilot_cells = [c for c in cells if c[0] in pilot_R]
        held_cells = [c for c in cells if c[0] not in pilot_R]
        menu = cross_cell_greedy(pilot_cells, all_strats, max_K)
        held_out_R = [r for r in R_list if r not in pilot_R][0]
        for hc in held_cells:
            row = {
                "pilot_R": ",".join(pilot_R),
                "held_R": held_out_R,
                "held_cell": f"{hc[0]}/{hc[1]}",
                "menu": menu,
            }
            for n in tiers_n:
                if n == "full":
                    cols = all_models(hc, strats=menu)
                else:
                    cols = top_n_per_strat(hc, menu, n=n)
                r, n_models = ens(cols, hc[3], hc[4])
                row[f"top{n}_oracle"] = r
                row[f"top{n}_n"] = n_models
            # Kitchen sink: all 18 strategies × all their models in this cell
            cols_all = all_models(hc, strats=all_strats)
            r, n_models = ens(cols_all, hc[3], hc[4])
            row["all18_full_oracle"] = r
            row["all18_full_n"] = n_models
            out.append(row)
    return out


def plot_t2b(rows, out_png):
    tiers = ["top1", "top3", "top5", "top10", "topfull"]
    tier_labels = [
        "top-1 / strat\n(5 models)",
        "top-3 / strat\n(15)",
        "top-5 / strat\n(25)",
        "top-10 / strat\n(50)",
        "K=5 full pool",
    ]
    tier_colors = ["#d62728", "#ff7f0e", "#9467bd", "#1f77b4", "#2ca02c"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.3), gridspec_kw={"width_ratios": [3, 2]})

    # Panel 1: per held-out cell, absolute oracle_r per tier
    cells_labels = sorted({r["held_cell"] for r in rows})
    by_cell = {c: {t: [] for t in tiers + ["all18_full"]} for c in cells_labels}
    for r in rows:
        c = r["held_cell"]
        for t in tiers:
            by_cell[c][t].append(r[f"{t}_oracle"])
        by_cell[c]["all18_full"].append(r["all18_full_oracle"])

    xx = np.arange(len(cells_labels))
    offset = np.linspace(-0.32, 0.32, len(tiers))
    for i, t in enumerate(tiers):
        means = [np.nanmean(by_cell[c][t]) for c in cells_labels]
        stds = [np.nanstd(by_cell[c][t]) for c in cells_labels]
        ax1.errorbar(
            xx + offset[i],
            means,
            yerr=stds,
            fmt="o",
            capsize=3,
            ms=6,
            color=tier_colors[i],
            label=tier_labels[i],
        )
    a18 = [np.nanmean(by_cell[c]["all18_full"]) for c in cells_labels]
    for x, k in zip(xx, a18):
        ax1.hlines(
            k,
            x - 0.42,
            x + 0.42,
            color="#444",
            lw=2,
            ls="-",
            label="all 18 strats × full pool" if x == xx[0] else None,
        )
    ax1.set_xticks(xx)
    ax1.set_xticklabels(cells_labels, rotation=40, ha="right", fontsize=8)
    ax1.set_ylabel("ensemble oracle_r (held-out cell)")
    ax1.set_title(
        "T2b: how does restriction tier affect held-out oracle_r?\n"
        "(K=5 menu chosen on 2 pilot reservoirs; held-out cells in 3rd reservoir)",
        fontsize=10,
    )
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(axis="y", alpha=0.25)

    # Panel 2: gap to K=5 full pool (positive = full pool wins)
    deltas = {t: [] for t in tiers}
    deltas_all18 = []
    for r in rows:
        ref = r["topfull_oracle"]
        if not np.isfinite(ref):
            continue
        for t in tiers:
            v = r[f"{t}_oracle"]
            if np.isfinite(v):
                deltas[t].append(ref - v)
        if np.isfinite(r["all18_full_oracle"]):
            deltas_all18.append(ref - r["all18_full_oracle"])
    bp_data = [deltas[t] for t in tiers] + [deltas_all18]
    bp_labels = tier_labels + ["all 18\nkitchen sink"]
    bp_colors = tier_colors + ["#444"]
    bp = ax2.boxplot(bp_data, tick_labels=bp_labels, showmeans=True, patch_artist=True, widths=0.55)
    for patch, c in zip(bp["boxes"], bp_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax2.axhline(0.005, color="red", ls="--", lw=1, alpha=0.7, label="noise floor +0.005")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("gap vs K=5 FULL pool (positive = restriction lost quality)")
    ax2.set_title(
        "Gap to K=5 full-pool reference\nIf a restriction-tier box sits at 0 → restriction is free",
        fontsize=10,
    )
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(axis="y", alpha=0.25)
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right", fontsize=8)

    fig.suptitle(
        "Test 2b (revised) — does restricting to top-N models per strategy generalize as well as the full pool?",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"WROTE {out_png}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading pilot at D={D} ...", flush=True)
    cells, all_strats = all_cells()
    print(f"  {len(cells)} cells, {len(all_strats)} strategies", flush=True)
    print("T2b: restriction tiers ...", flush=True)
    rows = t2b_restriction(cells, all_strats, max_K=5)
    json.dump(
        rows,
        open(os.path.join(OUT_DIR, "validation_t2b_revised.json"), "w"),
        indent=2,
        default=float,
    )
    plot_t2b(rows, os.path.join(OUT_DIR, "validation_t2b_revised.png"))

    print("\n=== T2b SUMMARY (gap vs K=5 FULL pool, smaller = better restriction) ===")
    tiers = ["top1", "top3", "top5", "top10"]
    for t in tiers:
        gaps = [
            r["topfull_oracle"] - r[f"{t}_oracle"]
            for r in rows
            if np.isfinite(r["topfull_oracle"]) and np.isfinite(r[f"{t}_oracle"])
        ]
        if gaps:
            print(
                f"  {t:>10s}  median={np.median(gaps):+.4f}  max={np.max(gaps):+.4f}  n={len(gaps)}"
            )
    a18_gaps = [
        r["topfull_oracle"] - r["all18_full_oracle"]
        for r in rows
        if np.isfinite(r["topfull_oracle"]) and np.isfinite(r["all18_full_oracle"])
    ]
    if a18_gaps:
        print(
            f"  {'all18_full':>10s}  median={np.median(a18_gaps):+.4f}  max={np.max(a18_gaps):+.4f}  n={len(a18_gaps)}"
        )


if __name__ == "__main__":
    main()
