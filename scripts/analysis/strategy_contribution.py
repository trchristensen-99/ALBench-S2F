"""Rank HP strategies by ENSEMBLE CONTRIBUTION (not solo best-val), and test whether
the contributing set is stable ACROSS reservoirs — i.e. is the deploy menu
reservoir-agnostic or biased to whichever reservoir we fit on?

Per reservoir cell: pool top-K models/strategy (within budget), fit positive
ElasticNet on that reservoir's OWN val (val_pred->val_labels), score on its OWN
canonical oracle test (test_pred->test_oracle). Then:
  - weight_share[s] = summed ElasticNet weight landing on strategy s's models
  - loso[s]         = r_full - r_without_s   (marginal value; ~0 = redundant)
Aggregate mean over seeds within reservoir, then report per-reservoir columns so
we can see if the SAME strategies top the ranking everywhere.
"""

import glob
import json
import os

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNet, ElasticNetCV

E100 = "outputs/hp_step1_bakeoff_e100"
RESERVOIRS = {
    "genomic": ["seed42_0", "seed43_1", "seed44_2"],
    "motif_planted_v2": ["seed42_0", "seed43_1"],
    "dinuc_shuffle": ["seed42_0", "seed43_1"],
}
BUDGET = 75
K = 5


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
    rows = rows[:BUDGET]
    rows = sorted(rows, key=lambda r: -r[1])[:K]
    return [r[2] for r in rows]


def load_cell(seed_dir):
    cells = sorted(d for d in glob.glob(os.path.join(seed_dir, "*")) if os.path.isdir(d))
    if not cells:
        return None
    lab = np.load(os.path.join(cells[0], "labels.npz"))
    vy, oy = lab["val_labels"], lab["test_oracle"]
    cols, who = [], []
    for cd in cells:
        strat = os.path.basename(cd)
        for m in cell_topk(cd):
            try:
                z = np.load(m.replace("_meta.json", ".npz"))
            except Exception:
                continue
            if z["val_pred"].shape != vy.shape or z["test_pred"].shape != oy.shape:
                continue
            cols.append((z["val_pred"], z["test_pred"]))
            who.append(strat)
    if not cols:
        return None
    V = np.array([c[0] for c in cols]).T
    T = np.array([c[1] for c in cols]).T
    return V, T, vy, oy, np.array(who)


def fit_score(V, T, vy, oy, l1=None):
    if l1 is None:
        en = ElasticNetCV(
            l1_ratio=[0.5, 0.9, 0.95, 1.0], positive=True, cv=5, max_iter=20000, n_jobs=1
        )
        en.fit(V, vy)
        return pearsonr(en.predict(T), oy)[0], en.coef_, float(en.l1_ratio_), float(en.alpha_)
    en = ElasticNet(l1_ratio=l1[0], alpha=l1[1], positive=True, max_iter=20000)
    en.fit(V, vy)
    return pearsonr(en.predict(T), oy)[0], en.coef_, l1[0], l1[1]


def analyze():
    strats = sorted(
        {
            os.path.basename(d)
            for d in glob.glob(os.path.join(E100, "k562_genomic_d30000", "seed42_0", "*"))
            if os.path.isdir(d)
        }
    )
    weight = {s: {} for s in strats}
    loso = {s: {} for s in strats}
    full_r = {}
    for R, seeds in RESERVOIRS.items():
        for s in strats:
            weight[s][R], loso[s][R] = [], []
        rs = []
        for sd in seeds:
            out = load_cell(os.path.join(E100, f"k562_{R}_d30000", sd))
            if out is None:
                continue
            V, T, vy, oy, who = out
            r_full, coef, l1r, alpha = fit_score(V, T, vy, oy)
            rs.append(r_full)
            tot_w = coef.sum() or 1.0
            for s in strats:
                mask = who == s
                weight[s][R].append(float(coef[mask].sum() / tot_w))
                if mask.all() or not mask.any():
                    loso[s][R].append(0.0)
                    continue
                r_minus, *_ = fit_score(V[:, ~mask], T[:, ~mask], vy, oy, l1=(l1r, alpha))
                loso[s][R].append(float(r_full - r_minus))
        full_r[R] = np.mean(rs) if rs else float("nan")

    Rs = list(RESERVOIRS)
    print(
        f"full-ensemble oracle-r per reservoir: "
        + "  ".join(f"{R[:6]}={full_r[R]:.4f}" for R in Rs)
    )
    print(
        f"\n{'strategy':22s} | " + " ".join(f"{R[:5]:>13s}" for R in Rs) + " |  mean_w  mean_loso"
    )
    print(f"{'':22s} | " + " ".join(f"{'w%':>6s}{'loso':>7s}" for _ in Rs) + " |")
    agg = {}
    for s in strats:
        mw = np.mean([np.mean(weight[s][R]) for R in Rs if weight[s][R]])
        ml = np.mean([np.mean(loso[s][R]) for R in Rs if loso[s][R]])
        agg[s] = (mw, ml)
    for s in sorted(strats, key=lambda s: -agg[s][1]):
        cells = ""
        for R in Rs:
            w = 100 * np.mean(weight[s][R]) if weight[s][R] else 0
            ls = np.mean(loso[s][R]) if loso[s][R] else 0
            cells += f"{w:6.1f}{ls:7.4f}"
        print(f"{s:22s} | {cells} | {100 * agg[s][0]:6.1f}  {agg[s][1]:+.4f}")


if __name__ == "__main__":
    analyze()
