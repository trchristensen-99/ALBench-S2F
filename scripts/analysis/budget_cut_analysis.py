"""Can we cut MODEL_BUDGET below 100? Mine the deep genomic anchor (budget 200, 3 seeds).

Two questions, two readouts:
  (A) DEPLOY plateau: single-val ElasticNet ensemble oracle-r when each cell is capped
      to its first-N models (by round order). If r is flat past N, the deploy ensemble
      doesn't need more than N — the budget can drop to N.
  (B) SELECTION fairness: for each strategy, the round-fraction at which its top-k (by
      val) models land. Adaptive/LLM strategies whose best models arrive LATE would be
      unfairly penalized by an aggressive cut; cheap samplers that peak early can be cut
      hard. Also report, per strategy, the smallest N at which best-val-so-far is within
      1e-3 of its final-200 best (the strategy's own truncation point).
"""

import glob
import json
import os
import re

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV

ROOT = "outputs/hp_step1_bakeoff_e100/k562_genomic_d30000"
SEEDS = ["seed42_0", "seed43_1", "seed44_2"]
K = 5
BUDGETS = [25, 50, 75, 100, 150, 200]


def cell_rows(cd):
    rows = []
    for m in sorted(glob.glob(os.path.join(cd, "r*_meta.json"))):
        try:
            d = json.load(open(m))
        except Exception:
            continue
        vp = d.get("val_pearson")
        if vp is None:
            continue
        rnd = d.get("round")
        if rnd is None:
            mm = re.search(r"/r(\d+)_", m)
            rnd = int(mm.group(1)) if mm else -1
        rows.append((int(rnd), float(vp), m))
    rows.sort()  # by round ascending (trajectory order)
    return rows


def load_capped(seed, budget):
    cells = sorted(d for d in glob.glob(os.path.join(ROOT, seed, "*")) if os.path.isdir(d))
    if not cells:
        return None
    lab = np.load(os.path.join(cells[0], "labels.npz"))
    vy, oy = lab["val_labels"], lab["oracle_genomic"]
    V, T = [], []
    for cd in cells:
        rows = cell_rows(cd)[:budget]
        rows = sorted(rows, key=lambda r: -r[1])[:K]  # top-k by val within budget
        for _, _, m in rows:
            try:
                z = np.load(m.replace("_meta.json", ".npz"))
            except Exception:
                continue
            if z["val_pred"].shape != vy.shape:
                continue
            V.append(z["val_pred"])
            T.append(z["test_pred_genomic"])
    if not V:
        return None
    return np.array(V).T, np.array(T).T, vy, oy


def stack(V, T, vy, oy):
    en = ElasticNetCV(l1_ratio=[0.5, 0.9, 0.95, 1.0], positive=True, cv=5, max_iter=20000, n_jobs=1)
    en.fit(V, vy)
    return pearsonr(en.predict(T), oy)[0], int((en.coef_ > 1e-8).sum())


def deploy_plateau():
    print("=== (A) DEPLOY: single-val ensemble oracle-r vs per-cell budget (mean over seeds) ===")
    print(f"  {'budget':8s} " + "  ".join(f"{s[:6]:>8s}" for s in SEEDS) + "    mean   d_vs_200")
    means = {}
    for b in BUDGETS:
        rs = []
        for s in SEEDS:
            out = load_capped(s, b)
            if out is None:
                continue
            r, _ = stack(*out)
            rs.append(r)
        means[b] = np.mean(rs) if rs else float("nan")
    for b in BUDGETS:
        rs = []
        for s in SEEDS:
            out = load_capped(s, b)
            if out is None:
                rs.append(float("nan"))
                continue
            r, _ = stack(*out)
            rs.append(r)
        d = means[b] - means[200]
        print(f"  {b:<8d} " + "  ".join(f"{r:8.4f}" for r in rs) + f"   {means[b]:.4f}  {d:+.4f}")


def selection_fairness():
    print("\n=== (B) SELECTION: per-strategy best-val truncation point + top-k round-skew ===")
    print("  (trunc_N = smallest N where best-val within 1e-3 of final-200 best;")
    print("   late_frac = mean round-fraction of the top-k models, 1.0 = all at the end)")
    strat_rows = {}
    for s in SEEDS:
        for cd in sorted(glob.glob(os.path.join(ROOT, s, "*"))):
            if not os.path.isdir(cd):
                continue
            strat = os.path.basename(cd)
            rows = cell_rows(cd)
            if len(rows) < 50:
                continue
            strat_rows.setdefault(strat, []).append(rows)
    print(
        f"  {'strategy':22s} {'n_cells':>7s} {'final_best':>10s} {'trunc_N':>8s} {'late_frac':>9s}"
    )
    for strat in sorted(strat_rows):
        cell_list = strat_rows[strat]
        truncs, lates, finals = [], [], []
        for rows in cell_list:
            vals = [r[1] for r in rows]
            nmax = len(vals)
            best_run = np.maximum.accumulate(vals)
            final_best = best_run[-1]
            finals.append(final_best)
            # smallest N where running-best within 1e-3 of final
            tn = next((i + 1 for i in range(nmax) if best_run[i] >= final_best - 1e-3), nmax)
            truncs.append(tn)
            # round-fraction of the top-k-by-val models
            order = sorted(range(nmax), key=lambda i: -vals[i])[:K]
            lates.append(np.mean([rows[i][0] for i in order]) / max(1, rows[-1][0]))
        print(
            f"  {strat:22s} {len(cell_list):7d} {np.mean(finals):10.4f} "
            f"{int(np.median(truncs)):8d} {np.mean(lates):9.2f}"
        )


if __name__ == "__main__":
    deploy_plateau()
    selection_fairness()
