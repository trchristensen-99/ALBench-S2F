"""Marginal benefit of pooling additional HP-search strategies into the ensemble.

For each bake-off reservoir cell (D=30k), rank HP strategies by their best single model's
val Pearson, then cumulatively pool the top-1, top-2, ... strategies' models and build one
val-selected greedy ElasticNet ensemble at each step; record its genomic test Pearson.
Averaged across reservoirs -> the marginal curve + knee that justifies pooling across strategies.
"""

import os, sys, json, glob
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV

BASE = "outputs/hp_step1_bakeoff_e100"
RES = [
    "genomic",
    "dinuc_shuffle",
    "evoaug_heavy",
    "gc_matched",
    "motif_planted_v2",
    "phylogenetic_zoonomia",
    "uncertainty_guided",
    "diversity_guided",
]
D = 30000
SET = "genomic"
MAX_POOL, MAX_SIZE, MIN_DELTA = 40, 8, 1e-4
EN_KW = dict(l1_ratio=[0.1, 0.5, 0.9, 1.0], positive=True, cv=3, max_iter=5000, n_jobs=1)


def load_cell(R):
    cell = f"{BASE}/k562_{R}_d{D}/seed42_0"
    lf = sorted(glob.glob(f"{cell}/*/labels.npz"))
    if not lf:
        return None
    lz = np.load(lf[0])
    if "val_labels" not in lz.files or f"oracle_{SET}" not in lz.files:
        return None
    vy = lz["val_labels"].astype(np.float64)
    oracle = lz[f"oracle_{SET}"].astype(np.float64)
    models = []
    for mp in sorted(glob.glob(f"{cell}/*/r*_meta.json")):
        strat = os.path.basename(os.path.dirname(mp))
        try:
            mj = json.loads(open(mp).read())
        except Exception:
            continue
        vp = mj.get("val_pearson")
        if vp is None or not np.isfinite(vp):
            continue  # OOM/error stub
        npz = mp.replace("_meta.json", ".npz")
        if not os.path.exists(npz):
            continue
        try:
            d = np.load(npz)
        except Exception:
            continue
        if "val_pred" not in d.files or f"test_pred_{SET}" not in d.files:
            continue
        models.append(
            dict(
                strat=strat,
                val_pearson=float(vp),
                val_pred=d["val_pred"].astype(np.float64),
                test=d[f"test_pred_{SET}"].astype(np.float64),
            )
        )
    if len(models) < 2:
        return None
    return dict(vy=vy, oracle=oracle, models=models)


def greedy_ens(models, vy, oracle):
    pool = sorted(models, key=lambda m: -m["val_pearson"])[:MAX_POOL]
    selected, best_val, best_en, remaining = [], -np.inf, None, list(pool)
    while remaining and len(selected) < MAX_SIZE:
        best = None
        for c in remaining:
            V = np.column_stack([m["val_pred"] for m in selected + [c]])
            en = ElasticNetCV(**EN_KW).fit(V, vy)
            vp = pearsonr(en.predict(V), vy)[0]
            if best is None or vp > best[0]:
                best = (vp, c, en)
        vp, c, en = best
        if vp <= best_val + MIN_DELTA and selected:
            break
        selected.append(c)
        remaining.remove(c)
        best_val = vp
        best_en = en
    if not selected:
        return None
    T = np.column_stack([m["test"] for m in selected])
    return float(pearsonr(best_en.predict(T), oracle)[0])


def marginal(cell, maxk=8):
    # rank strategies by their best single model's val Pearson
    strats = {}
    for m in cell["models"]:
        strats.setdefault(m["strat"], []).append(m)
    strats = {s: ms for s, ms in strats.items() if len(ms) >= 2}
    order = sorted(strats, key=lambda s: -max(m["val_pearson"] for m in strats[s]))
    curve = []
    for k in range(1, min(maxk, len(order)) + 1):
        pooled = [m for s in order[:k] for m in strats[s]]
        tp = greedy_ens(pooled, cell["vy"], cell["oracle"])
        curve.append(tp)
        print(f"    k={k} strat={order[k - 1]:<18} test={tp:.4f}", flush=True)
    return curve, order


results, orders = {}, {}
for R in RES:
    c = load_cell(R)
    if c is None:
        print(f"skip {R} (no/insufficient data)", flush=True)
        continue
    print(
        f"[{R}] {len(c['models'])} models, {len(set(m['strat'] for m in c['models']))} strategies",
        flush=True,
    )
    curve, order = marginal(c)
    results[R] = curve
    orders[R] = order

maxk = max(len(v) for v in results.values())
agg = []
for k in range(maxk):
    vals = [v[k] for v in results.values() if len(v) > k and v[k] is not None]
    agg.append([k + 1, float(np.mean(vals)), float(np.std(vals)), len(vals)])
print("\n=== AGGREGATE (k | mean | std | n | marginal) ===")
marg = []
for i, (k, m, s, n) in enumerate(agg):
    g = m - agg[i - 1][1] if i else 0.0
    marg.append(g)
    print(f"k={k}  mean={m:.4f}  std={s:.4f}  n={n}  marginal={g:+.4f}")

json.dump(
    dict(per_reservoir=results, order=orders, aggregate=agg, marginal=marg),
    open(f"{BASE}/strategy_marginal.json", "w"),
    indent=1,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
for R, v in results.items():
    a1.plot(range(1, len(v) + 1), v, color="0.78", lw=1, alpha=0.8)
ks = [a[0] for a in agg]
ms = [a[1] for a in agg]
ss = [a[2] for a in agg]
a1.errorbar(
    ks, ms, yerr=ss, color="#1d4ed8", lw=2.6, marker="o", capsize=3, label="mean over reservoirs"
)
a1.set_xlabel("# HP-search strategies pooled (best-first)")
a1.set_ylabel("ensemble test Pearson (genomic)")
a1.set_title("Marginal benefit of pooling HP strategies (D=30k)")
a1.legend()
a1.grid(alpha=0.3)
a2.bar(ks[1:], [m * 100 for m in marg[1:]], color="#16a34a")
a2.axhline(0.3, ls="--", color="0.5", label="~noise 0.003")
a2.set_xlabel("strategy added (k)")
a2.set_ylabel("marginal gain x100 (Pearson)")
a2.set_title("Per-step marginal gain")
a2.legend()
a2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{BASE}/strategy_marginal.png", dpi=140)
print("WROTE", f"{BASE}/strategy_marginal.png")
