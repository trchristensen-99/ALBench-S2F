"""Two additional analyses for the HP strategy-selection question:

Figure 1 — Cross-cell greedy forward
  Pooled-across-reservoirs greedy: at each step pick the strategy that most
  raises the MEAN oracle-r across all (reservoir × seed) cells. This is the
  Level-1 menu-selection objective (reservoir-agnostic). Overlaid with the
  random-subset distribution from the same data so you can see how much the
  pooled-knee shifts when picks are optimal rather than typical.

Figure 2 — Per-reservoir greedy picks
  Heatmap of greedy-rank by reservoir + cross-cell, showing whether early picks
  agree across reservoirs (= one universal recipe works) or differ (= reservoir-
  specific menus needed). Companion line: top-k Jaccard similarity across
  reservoirs as k grows.

Saves PNGs into ~/Downloads/hp_strategy_curves/. Run via sbatch on cpuq."""

import glob
import itertools
import json
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV

E100 = os.environ.get("CURVES_ROOT", "outputs/hp_step1_bakeoff_e100")
D = os.environ.get("CURVES_D", "30000")
RESERVOIRS = {
    "genomic": ["seed42_0", "seed43_1", "seed44_2"],
    "motif_planted_v2": ["seed42_0", "seed43_1"],
    "dinuc_shuffle": ["seed42_0", "seed43_1"],
}
BUDGET = int(os.environ.get("CURVES_BUDGET", "75"))
K = int(os.environ.get("CURVES_TOPK", "5"))
N_SAMPLES = int(os.environ.get("CURVES_SAMPLES", "30"))
MAXK = int(os.environ.get("CURVES_MAXK", "9"))
OUT_DIR = os.environ.get("CURVES_OUT", "outputs/analysis")
random.seed(0)


# ── data loading (mirrors strategy_count_curves.py) ──────────────────────────
def cell_topk(cd):
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
    rows = sorted(rows[:BUDGET], key=lambda r: -r[1])[:K]
    return [r[2] for r in rows]


def load_seed(seed_dir):
    cells = sorted(d for d in glob.glob(os.path.join(seed_dir, "*")) if os.path.isdir(d))
    if not cells:
        return None
    lab = np.load(os.path.join(cells[0], "labels.npz"))
    vy, oy = lab["val_labels"], lab["test_oracle"]
    by_strat = {}
    for cd in cells:
        s = os.path.basename(cd)
        for m in cell_topk(cd):
            try:
                z = np.load(m.replace("_meta.json", ".npz"))
            except Exception:
                continue
            if z["val_pred"].shape != vy.shape or z["test_pred"].shape != oy.shape:
                continue
            by_strat.setdefault(s, []).append((z["val_pred"], z["test_pred"]))
    return (by_strat, vy, oy) if by_strat else None


def score(by_strat, vy, oy, strats):
    cols = [c for s in strats for c in by_strat.get(s, [])]
    if not cols:
        return np.nan
    V = np.array([c[0] for c in cols]).T
    T = np.array([c[1] for c in cols]).T
    en = ElasticNetCV(l1_ratio=[0.5, 0.9, 0.95, 1.0], positive=True, cv=5, max_iter=20000, n_jobs=1)
    en.fit(V, vy)
    return float(pearsonr(en.predict(T), oy)[0])


def load_all():
    """Returns dict: reservoir -> list of (by_strat, vy, oy) per seed, plus union of strategies."""
    out = {}
    allstr = set()
    for R, seeds in RESERVOIRS.items():
        loaded = []
        for sd in seeds:
            res = load_seed(os.path.join(E100, f"k562_{R}_d{D}", sd))
            if res:
                loaded.append(res)
                allstr.update(res[0])
        if loaded:
            out[R] = loaded
    return out, sorted(allstr)


# ── greedy variants ──────────────────────────────────────────────────────────
def per_reservoir_greedy(loaded, allstr):
    """Greedy on each reservoir's own mean-across-seeds oracle-r."""
    selected, path = [], []
    for _ in range(min(MAXK, len(allstr))):
        best, best_s = -np.inf, None
        for cand in allstr:
            if cand in selected:
                continue
            rs = [score(bs, vy, oy, selected + [cand]) for bs, vy, oy in loaded]
            m = np.nanmean(rs)
            if m > best:
                best, best_s = m, cand
        if best_s is None:
            break
        selected.append(best_s)
        path.append((best_s, best))
    return path


def cross_cell_greedy(per_R, allstr):
    """Greedy on MEAN across ALL reservoirs × seeds — Level-1 menu objective."""
    flat = [seed_data for loaded in per_R.values() for seed_data in loaded]
    selected, path = [], []
    for _ in range(min(MAXK, len(allstr))):
        best, best_s = -np.inf, None
        for cand in allstr:
            if cand in selected:
                continue
            rs = [score(bs, vy, oy, selected + [cand]) for bs, vy, oy in flat]
            m = np.nanmean(rs)
            if m > best:
                best, best_s = m, cand
        if best_s is None:
            break
        selected.append(best_s)
        path.append((best_s, best))
    return path


def random_subset_pooled(per_R, allstr):
    """For each k, sample random k-subsets and score each on EVERY reservoir×seed,
    return pooled list. Used to overlay the random-subset cloud beneath the
    cross-cell greedy path."""
    flat = [seed_data for loaded in per_R.values() for seed_data in loaded]
    dist = {}
    for k in range(1, min(MAXK, len(allstr)) + 1):
        combos = list(itertools.combinations(allstr, k))
        if len(combos) > N_SAMPLES:
            combos = random.sample(combos, N_SAMPLES)
        vals = []
        for combo in combos:
            for bs, vy, oy in flat:
                r = score(bs, vy, oy, list(combo))
                if np.isfinite(r):
                    vals.append(r)
        dist[k] = vals
    return dist


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ── plots ────────────────────────────────────────────────────────────────────
def plot_cross_cell_greedy(cc_path, pooled_dist, out_png):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ks = sorted(pooled_dist)
    med = [np.median(pooled_dist[k]) for k in ks]
    p25 = [np.percentile(pooled_dist[k], 25) for k in ks]
    p75 = [np.percentile(pooled_dist[k], 75) for k in ks]
    ax.fill_between(ks, p25, p75, alpha=0.18, color="#888", label="random k-subset IQR")
    ax.plot(ks, med, "o-", color="#555", lw=1.2, ms=4, label="random k-subset median")
    cc_x = list(range(1, len(cc_path) + 1))
    cc_y = [r for _, r in cc_path]
    ax.plot(cc_x, cc_y, "s-", color="#d62728", lw=2, ms=6, label="cross-cell GREEDY")
    # Knee marker for greedy (where increment drops below 0.001)
    knee = None
    for i in range(1, len(cc_y)):
        if cc_y[i] - cc_y[i - 1] < 1e-3:
            knee = i
            break
    if knee is not None:
        ax.axvline(knee, color="k", ls="--", lw=1, alpha=0.4, label=f"greedy knee k={knee}")
    # Annotate the strategy picked at each greedy step
    for x, (s, r) in zip(cc_x, cc_path):
        ax.annotate(
            s,
            (x, r),
            fontsize=7,
            ha="left",
            va="bottom",
            xytext=(3, 3),
            textcoords="offset points",
            rotation=0,
            color="#d62728",
            alpha=0.85,
        )
    ax.set_xlabel("# HP strategies (k)")
    ax.set_ylabel("ensemble oracle-r  (mean across all reservoir×seed cells)")
    ax.set_title(
        f"Cross-cell greedy vs random-subset median — D={D}, pooled across {len(RESERVOIRS)} reservoirs"
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"WROTE {out_png}")
    plt.close(fig)


def plot_reservoir_overlap(per_R_paths, cc_path, out_png):
    Rs = list(per_R_paths)
    strat_pool = sorted(
        {s for path in per_R_paths.values() for s, _ in path} | {s for s, _ in cc_path}
    )
    cols = Rs + ["cross-cell"]
    # rank matrix: rows = strats, cols = reservoirs(+cc); val = greedy rank (1=first), NaN = not selected within MAXK
    M = np.full((len(strat_pool), len(cols)), np.nan)
    for j, R in enumerate(Rs):
        path = per_R_paths[R]
        for rank, (s, _) in enumerate(path, start=1):
            i = strat_pool.index(s)
            M[i, j] = rank
    for rank, (s, _) in enumerate(cc_path, start=1):
        i = strat_pool.index(s)
        M[i, -1] = rank
    # Sort rows by cross-cell rank (first), then earliest rank in any reservoir.
    order = sorted(
        range(len(strat_pool)),
        key=lambda i: (
            M[i, -1] if not np.isnan(M[i, -1]) else 99,
            np.nanmin(M[i, :]) if not np.all(np.isnan(M[i, :])) else 99,
        ),
    )
    M = M[order]
    strat_pool = [strat_pool[i] for i in order]

    fig, (axH, axJ) = plt.subplots(
        1,
        2,
        figsize=(13, max(4.5, 0.25 * len(strat_pool) + 1)),
        gridspec_kw={"width_ratios": [3, 2]},
    )
    im = axH.imshow(M, aspect="auto", cmap="viridis_r", vmin=1, vmax=MAXK)
    axH.set_yticks(range(len(strat_pool)))
    axH.set_yticklabels(strat_pool, fontsize=9)
    axH.set_xticks(range(len(cols)))
    axH.set_xticklabels(cols, rotation=20, ha="right", fontsize=9)
    axH.set_title(
        f"Greedy-pick RANK per reservoir (D={D})\nlower=earlier; gray=never picked within top-{MAXK}",
        fontsize=10,
    )
    for i in range(len(strat_pool)):
        for j in range(len(cols)):
            v = M[i, j]
            if not np.isnan(v):
                axH.text(
                    j,
                    i,
                    f"{int(v)}",
                    ha="center",
                    va="center",
                    color="white" if v <= MAXK / 2 else "black",
                    fontsize=8,
                )
    cb = fig.colorbar(im, ax=axH, fraction=0.04, pad=0.02)
    cb.set_label("greedy rank (1 = first pick)", fontsize=8)

    # Jaccard at top-k across reservoirs (not including cross-cell)
    ks = list(range(1, MAXK + 1))
    j_vals = []
    for k in ks:
        sets = [{s for s, _ in per_R_paths[R][:k]} for R in Rs]
        pairs = list(itertools.combinations(range(len(sets)), 2))
        if not pairs:
            j_vals.append(1.0)
            continue
        j_vals.append(float(np.mean([jaccard(sets[a], sets[b]) for a, b in pairs])))
    axJ.plot(ks, j_vals, "o-", color="#1f77b4", lw=2, ms=6)
    axJ.axhline(1.0, color="k", ls=":", lw=1, alpha=0.5)
    axJ.set_xlabel("top-k greedy picks compared")
    axJ.set_ylabel("mean pairwise Jaccard similarity")
    axJ.set_title("How similar are reservoirs' top-k greedy picks?", fontsize=10)
    axJ.set_ylim(0, 1.05)
    axJ.set_xticks(ks)
    axJ.grid(alpha=0.25)
    for k, j in zip(ks, j_vals):
        axJ.annotate(
            f"{j:.2f}",
            (k, j),
            fontsize=8,
            va="bottom",
            ha="center",
            xytext=(0, 4),
            textcoords="offset points",
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"WROTE {out_png}")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"loading D={D} data ...", flush=True)
    per_R, allstr = load_all()
    print(
        f"  {len(allstr)} strategies, {sum(len(v) for v in per_R.values())} reservoir-seed cells",
        flush=True,
    )

    print("per-reservoir greedy ...", flush=True)
    per_R_paths = {R: per_reservoir_greedy(per_R[R], allstr) for R in per_R}
    for R, p in per_R_paths.items():
        print(f"  {R}: {[s for s, _ in p]}")

    print("cross-cell greedy ...", flush=True)
    cc_path = cross_cell_greedy(per_R, allstr)
    print(f"  cross-cell: {[s for s, _ in cc_path]}")

    print("pooled random subsets ...", flush=True)
    pooled = random_subset_pooled(per_R, allstr)

    out1 = os.path.join(OUT_DIR, "strategy_count_curves_cross_cell.png")
    plot_cross_cell_greedy(cc_path, pooled, out1)

    out2 = os.path.join(OUT_DIR, "strategy_picks_by_reservoir.png")
    plot_reservoir_overlap(per_R_paths, cc_path, out2)


if __name__ == "__main__":
    main()
