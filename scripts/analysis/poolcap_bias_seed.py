import numpy as np, json, sys, copy
sys.argv = ["x"]
from scipy.stats import pearsonr
from scripts.analysis.slope_analysis import load_cell, greedy_ensemble

RES = ["genomic", "motif_planted_v2", "dinuc_shuffle", "evoaug_heavy", "gc_matched"]
SEEDS = ["42", "43"]
TAGS = ["gp", "tpe", "evos", "evob"]
SETS = ["genomic", "snv_ref", "ood", "dinuc_shuffle", "random_32k"]
D = 100000

cells = {}
for R in RES:
    for ds in SEEDS:
        c = load_cell("outputs/overnight", R, D, ds, TAGS, round_budget=50)
        if c is not None:
            cells[(R, ds)] = c
print(f"loaded {len(cells)} cells", flush=True)

oracle = {}
for s in SETS:
    for c in cells.values():
        if s in c.oracle:
            oracle[s] = c.oracle[s]
            break
SETS = [s for s in SETS if s in oracle]


def per_strat_filter(cell, n):
    keep = []
    for t in set(m.tag for m in cell.models):
        keep += sorted([m for m in cell.models if m.tag == t], key=lambda m: -m.val_pearson)[:n]
    c2 = copy.copy(cell)
    c2.models = keep
    return c2


def safe_pred(ens, s):
    try:
        return ens.predict(s)
    except Exception:
        return None


out = {}

# (a) pool-cap sweep (genomic/snv/ood)
POOLSETS = [s for s in ["genomic", "snv_ref", "ood"] if s in oracle]


def eval_pool(mode, val):
    preds = {s: [] for s in POOLSETS}
    for (R, ds), cell in cells.items():
        ens = greedy_ensemble(cell, max_pool=val) if mode == "combined" else greedy_ensemble(per_strat_filter(cell, val), max_pool=10 ** 9)
        for s in POOLSETS:
            p = safe_pred(ens, s)
            if p is not None:
                preds[s].append(p)
    return {s: float(pearsonr(np.mean(preds[s], axis=0), oracle[s])[0]) for s in POOLSETS if preds[s]}


out["combined"] = {cap: eval_pool("combined", cap) for cap in [5, 10, 20, 40, 80, 160]}
print("combined caps done", flush=True)
out["per_strategy"] = {n: eval_pool("per_strategy", n) for n in [3, 5, 10, 20, 40]}
print("per-strategy caps done", flush=True)

# per-cell ensembles (cap 40) for seed + coverage
ens40 = {k: greedy_ensemble(c, max_pool=40) for k, c in cells.items()}


def pooled(subsetR, seedset, s):
    ps = []
    for R in subsetR:
        for ds in seedset:
            if (R, ds) in ens40:
                p = safe_pred(ens40[(R, ds)], s)
                if p is not None:
                    ps.append(p)
    return float(pearsonr(np.mean(ps, axis=0), oracle[s])[0]) if ps else None


# (c) seed check
out["seed_check"] = {}
for lbl, seedset in [("1seed_42", ["42"]), ("2seed", ["42", "43"])]:
    out["seed_check"][lbl] = {s: pooled(RES, seedset, s) for s in SETS}

# (b) coverage bias
out["coverage"] = {"full5": {s: pooled(RES, SEEDS, s) for s in SETS}}
for drop in RES:
    sub = [R for R in RES if R != drop]
    out["coverage"][f"drop_{drop}"] = {s: pooled(sub, SEEDS, s) for s in SETS}

json.dump(out, open("outputs/overnight/final_analysis/poolcap_bias_seed.json", "w"), indent=1)
print("WROTE poolcap_bias_seed.json", flush=True)
