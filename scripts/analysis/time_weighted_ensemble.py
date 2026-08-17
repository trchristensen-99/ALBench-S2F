"""Time-weighted ensemble analysis: ensemble oracle_r as a function of total
GPU-hour budget per K=5 strategy.

For each cell:
  for each cap_time T in [0.5, 1, 2, ..., 30] GPU-h:
    pool = []
    for strat in K=5:
      take models from this strat with cumul train_time <= T
      add to pool
    full_pool_score = ElasticNetCV(pool, val_labels) -> test_oracle pearson
    greedy_k5_score = greedy-pick best 5 models from pool

Two flavors plotted side by side:
  - per-strategy budget T (parallel deploy: each strat gets T independently)
  - shared budget T/K (total budget T split equally across K strats)

Plus: x-axis = mean models trained per strategy at time T (combined view).
"""

import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
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
        "figure.titlesize": 17,
        "axes.titleweight": "bold",
    }
)

ROOT = "outputs/hp_step1_bakeoff_e100"
OUT = os.environ.get("PI_OUT", "outputs/analysis/pi_deck")
os.makedirs(OUT, exist_ok=True)

K5 = ["optuna_gp", "evo_batch", "llm_explore_nv1", "evo_single", "optuna_tpe"]
RESERVOIRS_30K = {
    "genomic": ["seed42_0", "seed43_1", "seed44_2"],
    "motif_planted_v2": ["seed42_0", "seed43_1"],
    "dinuc_shuffle": ["seed42_0", "seed43_1"],
}
RESERVOIRS_300K = {"genomic": ["seed42_0", "seed43_1", "seed44_2"]}


def cell_topk(cd, n):
    """Top-n models by val_pearson from first 75 rounds."""
    rows = []
    for m in sorted(glob.glob(os.path.join(cd, "r*_meta.json"))):
        try:
            d = json.load(open(m))
        except Exception:
            continue
        vp = d.get("val_pearson")
        rd = d.get("round")
        t = d.get("train_time_sec") or 0
        if vp is None or rd is None or not np.isfinite(vp):
            continue
        rows.append((int(rd), float(vp), float(t), m))
    rows.sort()
    return [r for r in rows[:75]]


def load_cell(reservoir, seed, D, k5_strats):
    """Returns (R, sd, per_strat, vy, oy). per_strat[strat] = ordered list of
    (round, val_pearson, train_time_s, val_pred, test_pred), sorted by round."""
    sd = os.path.join(ROOT, f"k562_{reservoir}_d{D}", seed)
    cells = sorted(d for d in glob.glob(os.path.join(sd, "*")) if os.path.isdir(d))
    if not cells:
        return None
    lab = np.load(os.path.join(cells[0], "labels.npz"))
    vy, oy = lab["val_labels"], lab["test_oracle"]
    per_strat = {}
    for cd in cells:
        s = os.path.basename(cd)
        if s not in k5_strats:
            continue
        items = []
        for rd, vp, t, m in cell_topk(cd, 75):
            try:
                z = np.load(m.replace("_meta.json", ".npz"))
            except Exception:
                continue
            if z["val_pred"].shape != vy.shape or z["test_pred"].shape != oy.shape:
                continue
            if not (np.all(np.isfinite(z["val_pred"])) and np.all(np.isfinite(z["test_pred"]))):
                continue
            items.append((rd, vp, t, z["val_pred"], z["test_pred"]))
        if items:
            per_strat[s] = items
    return (reservoir, seed, per_strat, vy, oy) if per_strat else None


def ens(cols, vy, oy):
    if not cols:
        return np.nan
    V = np.array([c[0] for c in cols]).T
    T = np.array([c[1] for c in cols]).T
    en = ElasticNetCV(l1_ratio=[0.5, 0.9, 0.95, 1.0], positive=True, cv=5, max_iter=20000, n_jobs=1)
    en.fit(V, vy)
    return float(pearsonr(en.predict(T), oy)[0])


def greedy_pick(candidates, vy, oy, k):
    if not candidates or k <= 0:
        return np.nan
    selected = []
    for _ in range(min(k, len(candidates))):
        best, best_i = -np.inf, None
        for i, c in enumerate(candidates):
            if i in {idx for idx, _ in selected}:
                continue
            r = ens([cc for _, cc in selected] + [c], vy, oy)
            if np.isfinite(r) and r > best:
                best, best_i = r, i
        if best_i is None:
            break
        selected.append((best_i, candidates[best_i]))
    return ens([c for _, c in selected], vy, oy)


def models_within_time(per_strat, T_hours):
    """For each K=5 strat, take models with cumulative time <= T (T in hours).
    Returns list of (val_pred, test_pred) pooled across strats, and the mean
    model-count per strat."""
    pool = []
    counts = []
    for s in K5:
        rows = per_strat.get(s, [])
        cumul = 0.0
        c = 0
        for rd, vp, t, vp_arr, tp_arr in rows:
            cumul += t / 3600.0
            if cumul <= T_hours:
                pool.append((vp_arr, tp_arr))
                c += 1
            else:
                break
        counts.append(c)
    return pool, float(np.mean(counts)) if counts else 0


def time_grid(D):
    if D == 30000:
        return [0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15]
    return [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25]


def analyze(D, reservoirs, out_path):
    cells = []
    for R, seeds in reservoirs.items():
        for sd in seeds:
            c = load_cell(R, sd, D, set(K5))
            if c:
                cells.append(c)
    if not cells:
        print(f"  no cells found for D={D}")
        return
    print(f"  D={D}: {len(cells)} cells loaded")

    T_grid = time_grid(D)
    per_cell = []  # list of dicts with "T", "full", "greedy", "models_per_strat"
    for cell in cells:
        R, sd, per_strat, vy, oy = cell
        cell_id = f"{R}/{sd}"
        row = {"cell": cell_id, "T": [], "full": [], "greedy": [], "mps": []}
        for T in T_grid:
            pool, mps = models_within_time(per_strat, T)
            if not pool:
                continue
            row["T"].append(T)
            row["full"].append(ens(pool, vy, oy))
            row["greedy"].append(greedy_pick(pool, vy, oy, k=5))
            row["mps"].append(mps)
            print(
                f"    {cell_id}  T={T:>4.1f}h  mps={mps:.1f}  full={row['full'][-1]:.4f}  greedy={row['greedy'][-1]:.4f}",
                flush=True,
            )
        per_cell.append(row)

    # ── plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    cmap = plt.get_cmap("tab10")
    color_map = {row["cell"]: cmap(i % 10) for i, row in enumerate(per_cell)}
    # Pooled mean
    fp_pool, gk_pool = defaultdict(list), defaultdict(list)
    for row in per_cell:
        ax1.plot(
            row["T"],
            row["full"],
            "o-",
            color=color_map[row["cell"]],
            ms=4,
            lw=1.2,
            alpha=0.85,
            label=row["cell"],
        )
        ax2.plot(
            row["T"],
            row["greedy"],
            "o-",
            color=color_map[row["cell"]],
            ms=4,
            lw=1.2,
            alpha=0.85,
            label=row["cell"],
        )
        for T, fp, gk in zip(row["T"], row["full"], row["greedy"]):
            if np.isfinite(fp):
                fp_pool[T].append(fp)
            if np.isfinite(gk):
                gk_pool[T].append(gk)
    pooled_T = [T for T in T_grid if len(fp_pool[T]) >= max(2, len(per_cell) // 2)]
    fp_mean = [np.mean(fp_pool[T]) for T in pooled_T]
    gk_mean = [np.mean(gk_pool[T]) for T in pooled_T]
    ax1.plot(
        pooled_T, fp_mean, "k-", lw=3.5, alpha=0.9, label=f"MEAN (n≥{max(2, len(per_cell) // 2)})"
    )
    ax2.plot(
        pooled_T, gk_mean, "k-", lw=3.5, alpha=0.9, label=f"MEAN (n≥{max(2, len(per_cell) // 2)})"
    )

    for ax, title in [(ax1, "Full pool ensemble"), (ax2, "Greedy K=5 ensemble")]:
        ax.set_xlabel("GPU-hours per K=5 strategy (parallel deploy)")
        ax.set_ylabel("ensemble oracle Pearson")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9, ncol=2)

    fig.suptitle(
        f"Time-weighted ensemble — D={D:,}\n"
        f"x = GPU-hours each K=5 strategy is given (assumes parallel-deploy on K GPUs)",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    analyze(30000, RESERVOIRS_30K, os.path.join(OUT, "fig15_time_weighted_ensemble_30k.png"))
    analyze(300000, RESERVOIRS_300K, os.path.join(OUT, "fig15_time_weighted_ensemble_300k.png"))
    print("DONE")
