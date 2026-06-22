"""Does adding more HP-optimization STRATEGIES keep helping the deploy ensemble?

For each reservoir cell we pool top-K models/strategy (within budget), then for
each subset SIZE k we sample MANY random strategy subsets, fit a positive
ElasticNet on that cell's own val, and score oracle-r on its own canonical test.
This shows the FULL distribution of achievable oracle-r at each k (not just the
greedy-optimal path), so the knee is visible regardless of WHICH strategies you
happen to pick. We overlay the greedy-forward path and the all-strategies ceiling.

Run on HPC (data lives there). Writes PNG + JSON; copy PNG to ~/Downloads after.
"""

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
K = int(os.environ.get("CURVES_TOPK", "5"))  # models kept per strategy
N_SAMPLES = int(os.environ.get("CURVES_SAMPLES", "40"))  # random subsets per k
OUT_PNG = os.environ.get("CURVES_PNG", "outputs/analysis/strategy_count_curves.png")
OUT_JSON = OUT_PNG.replace(".png", ".json")
random.seed(0)


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
    return pearsonr(en.predict(T), oy)[0]


def subsets_of_size(allstr, k, cap):
    combos = list(itertools.combinations(allstr, k))
    if len(combos) > cap:
        combos = random.sample(combos, cap)
    return combos


def greedy_path(loaded, allstr):
    selected, path = [], []
    for _ in range(len(allstr)):
        best, best_s = -1, None
        for cand in allstr:
            if cand in selected:
                continue
            rs = [score(bs, vy, oy, selected + [cand]) for bs, vy, oy in loaded]
            m = np.nanmean(rs)
            if m > best:
                best, best_s = m, cand
        selected.append(best_s)
        path.append(best)
    return path


def analyze_reservoir(R, seeds):
    loaded = [load_seed(os.path.join(E100, f"k562_{R}_d{D}", sd)) for sd in seeds]
    loaded = [x for x in loaded if x]
    if not loaded:
        return None
    allstr = sorted({s for bs, _, _ in loaded for s in bs})
    n = len(allstr)
    dist = {}  # k -> list of oracle-r over (subset x seed)
    for k in range(1, n + 1):
        vals = []
        for combo in subsets_of_size(allstr, k, N_SAMPLES):
            for bs, vy, oy in loaded:
                r = score(bs, vy, oy, list(combo))
                if np.isfinite(r):
                    vals.append(r)
        dist[k] = vals
    return {"strats": allstr, "n": n, "dist": dist, "greedy": greedy_path(loaded, allstr)}


def main():
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    results = {}
    for R, seeds in RESERVOIRS.items():
        print(f"[{R}] loading + sampling ...", flush=True)
        res = analyze_reservoir(R, seeds)
        if res:
            results[R] = res
            print(
                f"  n_strats={res['n']}  greedy@knee4={res['greedy'][3] if res['n'] >= 4 else 'NA'}"
            )

    Rs = list(results)
    fig, axes = plt.subplots(1, len(Rs) + 1, figsize=(5.0 * (len(Rs) + 1), 4.2), squeeze=False)
    axes = axes[0]
    mean_by_k = {}
    for ax, R in zip(axes, Rs):
        res = results[R]
        ks = sorted(res["dist"])
        data = [res["dist"][k] for k in ks]
        ax.violinplot(data, positions=ks, showmeans=True, widths=0.8)
        med = [np.median(res["dist"][k]) for k in ks]
        ax.plot(ks, med, "o-", color="#444", ms=3, lw=1, label="random-subset median")
        ax.plot(
            range(1, res["n"] + 1),
            res["greedy"],
            "s--",
            color="#d62728",
            ms=4,
            lw=1.4,
            label="greedy-forward",
        )
        ceil = np.median(res["dist"][res["n"]])
        ax.axhline(ceil, color="#1f77b4", ls=":", lw=1, alpha=0.7, label="all-strats")
        ax.set_title(f"{R}  (D={D})", fontsize=10)
        ax.set_xlabel("# strategies (k)")
        ax.set_ylabel("ensemble oracle-r")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
        for k in ks:
            mean_by_k.setdefault(k, []).extend(res["dist"][k])

    ax = axes[-1]
    ks = sorted(mean_by_k)
    med = [np.median(mean_by_k[k]) for k in ks]
    p25 = [np.percentile(mean_by_k[k], 25) for k in ks]
    p75 = [np.percentile(mean_by_k[k], 75) for k in ks]
    ax.fill_between(ks, p25, p75, alpha=0.2, color="#2ca02c", label="IQR (pooled R)")
    ax.plot(ks, med, "o-", color="#2ca02c", lw=1.6, label="median (pooled R)")
    if len(med) >= 1:
        final = med[-1]
        knee = next((k for k, m in zip(ks, med) if m >= final - 0.001), ks[-1])
        ax.axvline(knee, color="k", ls="--", lw=1, alpha=0.6, label=f"knee k={knee} (within 1e-3)")
    ax.set_title("Pooled across reservoirs", fontsize=10)
    ax.set_xlabel("# strategies (k)")
    ax.set_ylabel("ensemble oracle-r")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)

    fig.suptitle(
        f"Deploy-ensemble oracle-r vs #HP strategies — {N_SAMPLES} random subsets/k, budget {BUDGET}, top-{K}/strat",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=140)
    print(f"WROTE {OUT_PNG}")

    out = {
        R: {
            "strats": results[R]["strats"],
            "greedy": results[R]["greedy"],
            "median_by_k": {k: float(np.median(v)) for k, v in results[R]["dist"].items()},
            "iqr_by_k": {
                k: [float(np.percentile(v, 25)), float(np.percentile(v, 75))]
                for k, v in results[R]["dist"].items()
            },
        }
        for R in Rs
    }
    out["_pooled_median_by_k"] = {k: float(np.median(mean_by_k[k])) for k in ks}
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"WROTE {OUT_JSON}")

    print("\n=== pooled median oracle-r vs k ===")
    prev = None
    for k in ks:
        m = np.median(mean_by_k[k])
        d = "" if prev is None else f"{m - prev:+.4f}"
        prev = m
        print(f"  k={k:2d}  median={m:.4f}  d_prev={d}  n_samples={len(mean_by_k[k])}")


if __name__ == "__main__":
    main()
