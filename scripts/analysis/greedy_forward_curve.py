"""How many HP strategies does the deploy ensemble actually need? Greedy FORWARD
selection at the STRATEGY level: start empty, repeatedly add the strategy whose
addition most raises the (seed-averaged) oracle-r of the pooled ElasticNet ensemble.
Record the curve oracle-r vs #strategies, per reservoir, to find the knee.

This handles redundancy directly (no weighting guess): a strategy only advances the
curve if it adds something the already-selected ones don't cover."""

import glob
import json
import os

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV

E100 = "outputs/hp_step1_bakeoff_e100"
RESERVOIRS = {
    "genomic": ["seed42_0", "seed43_1", "seed44_2"],
    "motif_planted_v2": ["seed42_0", "seed43_1"],
    "dinuc_shuffle": ["seed42_0", "seed43_1"],
}
BUDGET = 75
K = 5
MAXK = 9


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
    return by_strat, vy, oy


def score(by_strat, vy, oy, strats):
    cols = [c for s in strats for c in by_strat.get(s, [])]
    if not cols:
        return np.nan
    V = np.array([c[0] for c in cols]).T
    T = np.array([c[1] for c in cols]).T
    en = ElasticNetCV(l1_ratio=[0.5, 0.9, 0.95, 1.0], positive=True, cv=5, max_iter=20000, n_jobs=1)
    en.fit(V, vy)
    return pearsonr(en.predict(T), oy)[0]


def greedy(reservoir, seeds):
    loaded = [load_seed(os.path.join(E100, f"k562_{reservoir}_d30000", sd)) for sd in seeds]
    loaded = [x for x in loaded if x]
    allstr = sorted({s for bs, _, _ in loaded for s in bs})
    selected, curve = [], []
    for _ in range(min(MAXK, len(allstr))):
        best, best_s = -1, None
        for cand in allstr:
            if cand in selected:
                continue
            rs = [score(bs, vy, oy, selected + [cand]) for bs, vy, oy in loaded]
            m = np.nanmean(rs)
            if m > best:
                best, best_s = m, cand
        selected.append(best_s)
        curve.append((best_s, best))
    return curve


def main():
    curves = {R: greedy(R, sds) for R, sds in RESERVOIRS.items()}
    Rs = list(RESERVOIRS)
    print(f"=== greedy forward: oracle-r vs #strategies (budget {BUDGET}, top-{K}/strat) ===")
    print(f"  {'k':>2s}  " + "  ".join(f"{R[:6]:>16s}" for R in Rs) + "      mean   d_vs_prev")
    prev = None
    for k in range(MAXK):
        cells, ms = [], []
        for R in Rs:
            if k < len(curves[R]):
                s, r = curves[R][k]
                cells.append(f"{s[:11]:11s}{r:5.3f}")
                ms.append(r)
            else:
                cells.append(" " * 16)
        mean = np.mean(ms)
        d = "" if prev is None else f"{mean - prev:+.4f}"
        prev = mean
        print(f"  {k + 1:>2d}  " + "  ".join(cells) + f"    {mean:.4f}  {d}")


if __name__ == "__main__":
    main()
