"""Ensemble oracle_r as a function of HP-search rounds available.

For each pilot cell, take models in chronological round order. At each
round_cap in a sweep, build the ensemble using only models whose round <=
round_cap (across K=5 strategies). Plot the saturation curve per cell + pooled.

Two ensemble schemes are evaluated at each round_cap:
  - 'full pool' : ElasticNetCV over ALL K=5 strats' models with round <= cap
  - 'greedy K=5': greedy model-level pick of K=5 models, also restricted to
                  rounds <= cap

If the full-pool curve plateaus at round_cap N, you can cut the search budget
to N without losing ensemble quality. If the greedy-K=5 curve also plateaus at
the same N, a small deploy ensemble is enough at that round count."""

import glob
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
ROUND_CAPS = [
    int(x)
    for x in os.environ.get("RE_ROUND_CAPS", "2,4,6,8,10,12,15,18,22,26,30,35,40,50,75").split(",")
]
GREEDY_K = int(os.environ.get("RE_GREEDY_K", "5"))
OUT_DIR = os.environ.get("VS_OUT", "outputs/analysis/validation_suite")


def cell_models_with_rounds(cd):
    """Return list of (round, val_pearson, val_pred, test_pred) for this cell."""
    rows = []
    for m in sorted(glob.glob(os.path.join(cd, "r*_meta.json"))):
        try:
            d = json.load(open(m))
        except Exception:
            continue
        vp = d.get("val_pearson")
        rd = d.get("round")
        if vp is None or rd is None:
            continue
        try:
            z = np.load(m.replace("_meta.json", ".npz"))
        except Exception:
            continue
        rows.append((int(rd), float(vp), z["val_pred"], z["test_pred"]))
    rows.sort()
    return rows


def load_cell(reservoir, seed, k5_strats):
    sd = os.path.join(E100, f"k562_{reservoir}_d{D}", seed)
    cells = sorted(d for d in glob.glob(os.path.join(sd, "*")) if os.path.isdir(d))
    if not cells:
        return None
    lab = np.load(os.path.join(cells[0], "labels.npz"))
    vy, oy = lab["val_labels"], lab["test_oracle"]
    # Collect rounds + preds for each K=5 strategy
    per_strat = {}
    for cd in cells:
        s = os.path.basename(cd)
        if s not in k5_strats:
            continue
        rows = cell_models_with_rounds(cd)
        good = []
        for r in rows:
            if r[2].shape == vy.shape and r[3].shape == oy.shape:
                # Drop NaN-bearing models — some training runs write partial
                # results when they crash, and ElasticNetCV rejects NaN.
                if np.all(np.isfinite(r[2])) and np.all(np.isfinite(r[3])):
                    good.append(r)
        per_strat[s] = good
    return (reservoir, seed, per_strat, vy, oy) if per_strat else None


def ens_oracle(cols, vy, oy):
    if not cols:
        return np.nan
    V = np.array([c[0] for c in cols]).T
    T = np.array([c[1] for c in cols]).T
    en = ElasticNetCV(l1_ratio=[0.5, 0.9, 0.95, 1.0], positive=True, cv=5, max_iter=20000, n_jobs=1)
    en.fit(V, vy)
    return float(pearsonr(en.predict(T), oy)[0])


def models_in_cap(cell, round_cap):
    """Return list of (val_pred, test_pred) from all K=5 strats with round <= cap."""
    _, _, per_strat, _, _ = cell
    out = []
    for s, rows in per_strat.items():
        for rd, _, vp, tp in rows:
            if rd <= round_cap:
                out.append((vp, tp))
    return out


def greedy_pick(cell, candidates, k):
    """Greedy model-level selection, return ensemble oracle_r at final k."""
    if not candidates or k <= 0:
        return np.nan
    _, _, _, vy, oy = cell
    selected = []
    for _ in range(min(k, len(candidates))):
        best, best_i = -np.inf, None
        for i, c in enumerate(candidates):
            if i in {idx for idx, _ in selected}:
                continue
            r = ens_oracle([c for _, c in selected] + [c], vy, oy)
            if np.isfinite(r) and r > best:
                best, best_i = r, i
        if best_i is None:
            break
        selected.append((best_i, candidates[best_i]))
    return ens_oracle([c for _, c in selected], vy, oy)


def cross_cell_strategy_greedy(cells_by_R, all_strats, max_K=5):
    """Strategy-level greedy on cross-cell mean (each strat contributes top-1
    model per cell)."""
    selected = []
    for _ in range(min(max_K, len(all_strats))):
        best, best_s = -np.inf, None
        for cand in all_strats:
            if cand in selected:
                continue
            scores = []
            for R, seeds in cells_by_R.items():
                for cell in seeds:
                    cols = []
                    for s in selected + [cand]:
                        items = sorted(cell[2].get(s, []), key=lambda x: -x[1])[:1]
                        cols.extend([(it[2], it[3]) for it in items])
                    if cols:
                        r = ens_oracle(cols, cell[3], cell[4])
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading all 18 strats per cell to derive K=5 menu ...", flush=True)
    # First pass: load all strategies (no filter) to derive K=5 via cross-cell greedy
    cells_full = {}
    for R, seeds in RESERVOIRS.items():
        cells_full[R] = []
        for sd in seeds:
            sd_full = os.path.join(E100, f"k562_{R}_d{D}", sd)
            cells_dirs = sorted(
                d for d in glob.glob(os.path.join(sd_full, "*")) if os.path.isdir(d)
            )
            all_strats = {os.path.basename(d) for d in cells_dirs}
            cell = load_cell(R, sd, all_strats)
            if cell:
                cells_full[R].append(cell)
    all_strats = sorted({s for L in cells_full.values() for c in L for s in c[2]})
    k5 = cross_cell_strategy_greedy(cells_full, all_strats, max_K=5)
    print(f"  K=5 menu: {k5}", flush=True)

    # Reload restricted to K=5 only (saves memory; we already have it but rescope)
    cells = [c for L in cells_full.values() for c in L]

    # Sweep round_caps; per cell record full-pool and greedy-K oracle_r
    results = {"caps": ROUND_CAPS, "k5_menu": k5, "cells": []}
    for cell in cells:
        cell_id = f"{cell[0]}/{cell[1]}"
        print(f"  scanning {cell_id} ...", flush=True)
        row = {"cell": cell_id, "full_pool": [], "greedy_k": [], "n_models": []}
        for cap in ROUND_CAPS:
            mods = [(c[2], c[3]) for s in k5 for c in cell[2].get(s, []) if c[0] <= cap]
            row["n_models"].append(len(mods))
            row["full_pool"].append(ens_oracle(mods, cell[3], cell[4]))
            row["greedy_k"].append(greedy_pick(cell, mods, k=GREEDY_K))
        results["cells"].append(row)

    json.dump(
        results,
        open(os.path.join(OUT_DIR, "rounds_vs_ensemble.json"), "w"),
        indent=2,
        default=float,
    )

    # --- plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0), sharey=True)

    cmap = plt.get_cmap("tab10")
    color_map = {row["cell"]: cmap(i % 10) for i, row in enumerate(results["cells"])}
    for row in results["cells"]:
        ax1.plot(
            ROUND_CAPS,
            row["full_pool"],
            "o-",
            color=color_map[row["cell"]],
            ms=4,
            lw=1.2,
            alpha=0.85,
            label=row["cell"],
        )
        ax2.plot(
            ROUND_CAPS,
            row["greedy_k"],
            "o-",
            color=color_map[row["cell"]],
            ms=4,
            lw=1.2,
            alpha=0.85,
            label=row["cell"],
        )

    # Pooled mean curves
    fp_mat = np.array([row["full_pool"] for row in results["cells"]])
    gk_mat = np.array([row["greedy_k"] for row in results["cells"]])
    fp_mean = np.nanmean(fp_mat, axis=0)
    gk_mean = np.nanmean(gk_mat, axis=0)
    ax1.plot(ROUND_CAPS, fp_mean, "k-", lw=3, alpha=0.85, label="MEAN across cells")
    ax2.plot(ROUND_CAPS, gk_mean, "k-", lw=3, alpha=0.85, label="MEAN across cells")

    # Annotate plateau: smallest cap where mean is within 0.005 of final
    final_fp, final_gk = fp_mean[-1], gk_mean[-1]
    knee_fp = next(
        (c for c, v in zip(ROUND_CAPS, fp_mean) if v >= final_fp - 0.005), ROUND_CAPS[-1]
    )
    knee_gk = next(
        (c for c, v in zip(ROUND_CAPS, gk_mean) if v >= final_gk - 0.005), ROUND_CAPS[-1]
    )
    ax1.axvline(knee_fp, color="red", ls="--", lw=1.2, alpha=0.7, label=f"plateau at cap={knee_fp}")
    ax2.axvline(knee_gk, color="red", ls="--", lw=1.2, alpha=0.7, label=f"plateau at cap={knee_gk}")

    for ax, title in [
        (ax1, f"Full pool (all K=5 strats × all models so far)"),
        (ax2, f"Greedy K={GREEDY_K} (best 5 models so far)"),
    ]:
        ax.set_xlabel("HP-search round cap (only models from rounds ≤ cap used)")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(ROUND_CAPS)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax1.set_ylabel("ensemble oracle_r")
    fig.suptitle(f"Ensemble oracle_r vs HP-search rounds available  (D={D}, K=5 menu)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, "rounds_vs_ensemble.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"WROTE {out}")

    print("\n=== plateau (smallest cap within 0.005 of cap=75) ===")
    print(f"  full pool  : cap={knee_fp}  (final={final_fp:.4f})")
    print(f"  greedy K=5 : cap={knee_gk}  (final={final_gk:.4f})")


if __name__ == "__main__":
    main()
