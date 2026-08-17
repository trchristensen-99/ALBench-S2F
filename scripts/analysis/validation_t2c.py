"""Test 2c: how many GREEDILY-PICKED models from the K=5 strategy history are
needed to match the full-pool ensemble? Unlike T2b (which forces 1/3/5/10 per
strategy), this lets greedy pick freely — possibly 3 from evo_batch + 1 from
optuna_gp + 0 from others, for example.

Setup:
  - Per held-out cell C (LOR with 2 pilot reservoirs picking the K=5 menu):
    pool all K=5 strats' models in C (~75-100 models per cell)
    greedy: at each step k, add the model whose addition most raises
      ElasticNetCV(positive=True) test_oracle pearson in C
    record curve oracle_r vs k for k = 1..30
  - Compare each k to the K=5 full-pool baseline
  - Knee = smallest k whose oracle_r is within 0.005 (noise floor) of the full
    pool baseline → that's the # of distinct HP configs we need to lock.

Cross-cell stability:
  - For each pilot cell, identify the HP configs (lr, bs, n_layers, ...) chosen
    by greedy in its top-K=8 selection.
  - Across cells, compute Jaccard overlap on (strategy, round) identity AND
    HP-tuple-nearest-neighbor identity.
"""

import glob
import itertools
import json
import os
from collections import defaultdict

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
TOPN_PER_STRAT = int(os.environ.get("VS_TOPN", "20"))  # pool depth per strategy
MAX_K = int(os.environ.get("VS_MAXK", "20"))
OUT_DIR = os.environ.get("VS_OUT", "outputs/analysis/validation_suite")
NOISE = 0.005


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
        rows.append((int(d.get("round", -1)), float(vp), m, d.get("hp", {})))
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
        for rnd, vp, m, hp in cell_topk(cd, TOPN_PER_STRAT):
            try:
                z = np.load(m.replace("_meta.json", ".npz"))
            except Exception:
                continue
            if z["val_pred"].shape != vy.shape or z["test_pred"].shape != oy.shape:
                continue
            by_strat.setdefault(s, []).append(
                {
                    "val": z["val_pred"],
                    "test": z["test_pred"],
                    "val_p": vp,
                    "round": rnd,
                    "strat": s,
                    "hp": hp,
                }
            )
    return (reservoir, seed, by_strat, vy, oy) if by_strat else None


def ens(cols, vy, oy):
    if not cols:
        return np.nan
    V = np.array([c["val"] for c in cols]).T
    T = np.array([c["test"] for c in cols]).T
    en = ElasticNetCV(l1_ratio=[0.5, 0.9, 0.95, 1.0], positive=True, cv=5, max_iter=20000, n_jobs=1)
    en.fit(V, vy)
    return float(pearsonr(en.predict(T), oy)[0])


def greedy_models(cell, candidates, max_k):
    """Greedy model-level selection over `candidates` (list of model dicts) in this cell.
    Returns ordered list of picks and curve of oracle_r."""
    _, _, _, vy, oy = cell
    selected, curve = [], []
    for _ in range(min(max_k, len(candidates))):
        best, best_i = -np.inf, None
        for i, m in enumerate(candidates):
            if i in {idx for idx, _ in selected}:
                continue
            r = ens([m for _, m in selected] + [m], vy, oy)
            if np.isfinite(r) and r > best:
                best, best_i = r, i
        if best_i is None:
            break
        selected.append((best_i, candidates[best_i]))
        curve.append(best)
    return [m for _, m in selected], curve


def cross_cell_strat_greedy(cells, all_strats, max_K=5):
    """STRATEGY-level greedy on cross-cell mean (using top-1 per strat). Returns menu."""
    selected = []
    for _ in range(min(max_K, len(all_strats))):
        best, best_s = -np.inf, None
        for cand in all_strats:
            if cand in selected:
                continue
            scores = []
            for c in cells:
                cols = []
                for s in selected + [cand]:
                    items = sorted(c[2].get(s, []), key=lambda x: -x["val_p"])[:1]
                    cols.extend(items)
                if cols:
                    r = ens(cols, c[3], c[4])
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


def t2c_runs(cells, all_strats):
    """Per LOR (2-pilot) split, in held-out cells:
    - greedy-pick K from K=5 strats' pool
    - record curve oracle_r vs k
    - record top-K=8 picks' identity (strat, round) for cross-cell stability"""
    R_list = sorted({c[0] for c in cells})
    out = []
    for pilot_R in itertools.combinations(R_list, 2):
        pilot_cells = [c for c in cells if c[0] in pilot_R]
        held_cells = [c for c in cells if c[0] not in pilot_R]
        menu = cross_cell_strat_greedy(pilot_cells, all_strats, max_K=5)
        for hc in held_cells:
            pool = []
            for s in menu:
                pool.extend(hc[2].get(s, []))
            full_score = ens(pool, hc[3], hc[4]) if pool else np.nan
            picks, curve = greedy_models(hc, pool, MAX_K)
            ids = [(p["strat"], p["round"]) for p in picks]
            out.append(
                {
                    "pilot_R": ",".join(pilot_R),
                    "held_cell": f"{hc[0]}/{hc[1]}",
                    "menu": menu,
                    "curve": curve,
                    "full_pool": full_score,
                    "picks_ids": ids,
                    "pool_size": len(pool),
                }
            )
    return out


def plot_t2c(rows, out_png):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [3, 2]})

    # Panel 1: oracle_r vs k per cell + full-pool reference
    cmap = plt.get_cmap("viridis")
    cells_uniq = sorted({r["held_cell"] for r in rows})
    color_map = {c: cmap(i / max(1, len(cells_uniq) - 1)) for i, c in enumerate(cells_uniq)}
    for r in rows:
        k_vals = np.arange(1, len(r["curve"]) + 1)
        ax1.plot(
            k_vals,
            r["curve"],
            "o-",
            color=color_map[r["held_cell"]],
            ms=3,
            lw=1.2,
            alpha=0.6,
            label=r["held_cell"]
            if r["pilot_R"]
            == sorted({rr["pilot_R"] for rr in rows if rr["held_cell"] == r["held_cell"]})[0]
            else None,
        )
        ax1.axhline(r["full_pool"], color=color_map[r["held_cell"]], lw=0.7, ls=":", alpha=0.4)
    ax1.axvspan(5, 8, alpha=0.10, color="green", label="target deploy size 5-8")
    ax1.set_xlabel("k (# greedily-picked models, no per-strat constraint)")
    ax1.set_ylabel("ensemble oracle_r in held-out cell")
    ax1.set_title(
        f"T2c: greedy-pick K models vs full K=5 pool  (D={D})\n"
        "dotted lines = full-pool oracle_r per cell",
        fontsize=10,
    )
    ax1.legend(fontsize=7, loc="lower right", ncol=2)
    ax1.grid(alpha=0.25)
    ax1.set_xticks(range(1, MAX_K + 1, 2))

    # Panel 2: gap to full pool at increasing k (box plot across cells × splits)
    gaps_at_k = defaultdict(list)
    for r in rows:
        for k, val in enumerate(r["curve"], start=1):
            gaps_at_k[k].append(r["full_pool"] - val)
    ks = sorted(gaps_at_k)
    data = [gaps_at_k[k] for k in ks]
    bp = ax2.boxplot(
        data, tick_labels=[str(k) for k in ks], showmeans=True, patch_artist=True, widths=0.6
    )
    for i, patch in enumerate(bp["boxes"]):
        k = ks[i]
        patch.set_facecolor("#9467bd" if 5 <= k <= 8 else "#bbb")
        patch.set_alpha(0.6)
    ax2.axhline(NOISE, color="red", ls="--", lw=1, label=f"noise floor +{NOISE}")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("k (# greedy picks)")
    ax2.set_ylabel("gap vs K=5 full pool (smaller = better)")
    ax2.set_title("Distribution of gap-to-full-pool across cells × splits", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("T2c — can 5-8 greedily-picked models match the full K=5 pool?", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"WROTE {out_png}")


def plot_pick_stability(rows, out_png, max_k=8):
    """At each k, plot (a) what fraction of picks come from each strategy and
    (b) how many cell-pair Jaccard agreements on the {(strat, round)} identity."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Composition by strategy at k=5 and k=8 across cells
    counts = {k: defaultdict(int) for k in (5, 8)}
    n_at_k = {k: 0 for k in (5, 8)}
    for r in rows:
        ids = r["picks_ids"]
        for k in (5, 8):
            if len(ids) >= k:
                for s, _ in ids[:k]:
                    counts[k][s] += 1
                n_at_k[k] += 1
    strats = sorted({s for k in counts for s in counts[k]})
    x = np.arange(len(strats))
    width = 0.36
    for i, k in enumerate((5, 8)):
        vals = [counts[k][s] / max(1, n_at_k[k]) for s in strats]
        ax1.bar(
            x + (i - 0.5) * width,
            vals,
            width=width,
            label=f"k={k}",
            color=["#1f77b4", "#ff7f0e"][i],
            alpha=0.85,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(strats, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("mean # picks per cell (out of k)")
    ax1.set_title(
        "Where do greedy picks come from?\n(allocation across K=5 strats at k=5 and k=8)",
        fontsize=10,
    )
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.25)

    # Jaccard of pick identities across cells at each k
    ks = list(range(1, max_k + 1))
    j_vals = []
    for k in ks:
        sets_by_cell = defaultdict(set)
        for r in rows:
            if len(r["picks_ids"]) >= k:
                # Tag each pick by (strategy, round) — proxy for HP-config identity
                sets_by_cell[r["held_cell"]].update(r["picks_ids"][:k])
        cells = sorted(sets_by_cell)
        if len(cells) < 2:
            j_vals.append(np.nan)
            continue
        pair_js = []
        for a, b in itertools.combinations(cells, 2):
            A, B = sets_by_cell[a], sets_by_cell[b]
            if A or B:
                pair_js.append(len(A & B) / len(A | B))
        j_vals.append(float(np.mean(pair_js)) if pair_js else np.nan)
    ax2.plot(ks, j_vals, "o-", color="#2ca02c", lw=2, ms=7)
    ax2.set_xlabel("top-k greedy picks compared")
    ax2.set_ylabel("mean Jaccard across cells\n(identity = (strategy, round))")
    ax2.set_title("How often do held-out cells agree on greedy picks?", fontsize=10)
    ax2.set_xticks(ks)
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.25)
    for k, j in zip(ks, j_vals):
        if np.isfinite(j):
            ax2.annotate(
                f"{j:.2f}",
                (k, j),
                fontsize=8,
                va="bottom",
                ha="center",
                xytext=(0, 4),
                textcoords="offset points",
            )

    fig.suptitle("T2c diagnostic — greedy-pick composition & cross-cell stability", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"WROTE {out_png}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading pilot at D={D} ...", flush=True)
    cells, all_strats = all_cells()
    print(f"  {len(cells)} cells, {len(all_strats)} strategies", flush=True)
    print("T2c: greedy-pick-K vs full pool (LOR transfer) ...", flush=True)
    rows = t2c_runs(cells, all_strats)
    json.dump(
        rows, open(os.path.join(OUT_DIR, "validation_t2c.json"), "w"), indent=2, default=float
    )
    plot_t2c(rows, os.path.join(OUT_DIR, "validation_t2c.png"))
    plot_pick_stability(rows, os.path.join(OUT_DIR, "validation_t2c_stability.png"))

    print("\n=== T2c SUMMARY (gap vs K=5 full pool, smaller = better) ===")
    gaps_at_k = defaultdict(list)
    for r in rows:
        for k, val in enumerate(r["curve"], start=1):
            gaps_at_k[k].append(r["full_pool"] - val)
    for k in sorted(gaps_at_k):
        g = gaps_at_k[k]
        print(f"  k={k:>2d}  median={np.median(g):+.4f}  max={np.max(g):+.4f}  n={len(g)}")


if __name__ == "__main__":
    main()
