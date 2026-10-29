# ruff: noqa
#!/usr/bin/env python
"""Compile the Oct 13-14 reservoir-scaling results (13 reservoirs in 5 families x D in {10k, 30k, 100k}, oracle_v3, llm2x20 HP search, cells in outputs/scaling_canonical_oct13) into tables.

Per cell (reservoir, D), on three fixed eval sets scored against the oracle (genomic test chr7/13, designed high-activity "ood", random_32k):
  n_models        trained models in the cell
  f4_first6       F4 metric: mean single-model Pearson r of the first 6 models of the search (rounds 0-2), equal search budget at every size and strategy
  top8_by_val     mean single-model r of the 8 models with the best validation r (the preliminary plot of Oct 14)
  best_single     r of the single model with the best validation r
  enet_greedy_k5  the deploy ensemble of that period: prefilter the 12 best by validation, greedy forward selection to 5, positive ElasticNetCV fitted on validation
  enet_all_pos    positive ElasticNetCV over all models of the cell (up to the first 40)
Ensembles use up to the first 40 models (the 2 x 20 design budget). Validation labels are the oracle labels of the chr19/21/X set used at the time.

    python scripts/analysis/compile_oct14_scaling.py [--out outputs/compile_oct14_scaling]      (cpuq)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np
from sklearn.linear_model import ElasticNetCV

BASE = "outputs/scaling_canonical_oct13"
FAMILIES = [
    ("genomic", ["genomic", "zoonomia_flank_primates", "zoonomia_flank_mammals"]),
    ("random", ["random", "gc_matched", "dinuc_shuffle"]),
    ("motif", ["motif_jaspar", "motif_ct_enriched", "motif_grammar"]),
    ("augmentation", ["evoaug_paper2023", "evoaug_prm010"]),
    ("model-based", ["d3_steps64", "evo2"]),
]
DS = [10000, 30000, 100000]
EVALS = ["genomic", "ood", "random_32k"]
MAXM = 40
VALSUB = 8000


def pear(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def load(res, D):
    d = os.path.join(BASE, f"{res}_d{D}")
    if not os.path.exists(os.path.join(d, "labels.npz")):
        return None
    lab = np.load(os.path.join(d, "labels.npz"))
    yval = lab["val_labels"].astype(np.float64)
    yte = {e: lab[f"oracle_{e}"].astype(np.float64) for e in EVALS}
    ms = []
    for f in glob.glob(os.path.join(d, "r*_meta.json")):
        if f.endswith("_screen.json"):
            continue
        try:
            m = json.load(open(f))
            if "error" in m:
                continue
            z = np.load(f.replace("_meta.json", ".npz"))
            if "val_pred" not in z.files or not all(f"test_pred_{e}" in z.files for e in EVALS):
                continue
            v = z["val_pred"].astype(np.float64)
            t = {e: z[f"test_pred_{e}"].astype(np.float64) for e in EVALS}
        except Exception:
            continue
        if not np.isfinite(v).all() or any(
            not np.isfinite(x).all() or np.abs(x).max() > 1e3 for x in t.values()
        ):
            continue
        ps = m.get("per_set_metrics", {})
        ms.append(
            {
                "id": m["model_id"],
                "round": m.get("round", 0),
                "val": v,
                "test": t,
                "ps": {e: ps.get(e, {}).get("pearson") for e in EVALS},
                "valr": pear(v, yval),
            }
        )
    ms.sort(key=lambda m: (m["round"], m["id"]))
    return ms, yval, yte


def greedy(V, yv, sub, prefilter=12, kmax=5):
    Vs, ys = V[sub], yv[sub]
    pr = np.array([pear(Vs[:, j], ys) for j in range(Vs.shape[1])])
    cand = list(np.argsort(pr)[::-1][:prefilter])
    chosen, best_val = [], -2.0
    A = np.column_stack([np.ones(len(ys)), Vs])
    while cand and len(chosen) < kmax:
        best = None
        for j in cand:
            cols = [0] + [c + 1 for c in chosen + [j]]
            w, *_ = np.linalg.lstsq(A[:, cols], ys, rcond=None)
            r = pear(A[:, cols] @ w, ys)
            if best is None or r > best[0]:
                best = (r, j)
        if chosen and best[0] <= best_val + 1e-5:
            break
        best_val = best[0]
        chosen.append(best[1])
        cand.remove(best[1])
    return chosen


def enet(cols, V, T, yval, yte, sub):
    en = ElasticNetCV(
        cv=5, n_alphas=40, max_iter=5000, l1_ratio=[0.1, 0.5, 0.9], positive=True
    ).fit(V[np.ix_(sub, cols)], yval[sub])
    return {e: pear(en.predict(T[e][:, cols]), yte[e]) for e in EVALS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/compile_oct14_scaling")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for fam, ress in FAMILIES:
        for res in ress:
            for D in DS:
                c = load(res, D)
                if c is None or not c[0]:
                    print("MISSING", res, D)
                    continue
                ms, yval, yte = c
                n = len(ms)
                first6 = ms[:6]
                top8 = sorted(ms, key=lambda m: -m["valr"])[:8]
                use = ms[:MAXM]
                V = np.column_stack([m["val"] for m in use])
                T = {e: np.column_stack([m["test"][e] for m in use]) for e in EVALS}
                sub = (
                    np.arange(len(yval))
                    if len(yval) <= VALSUB
                    else rng.choice(len(yval), VALSUB, replace=False)
                )
                best = int(np.argmax([m["valr"] for m in use]))
                g = greedy(V, yval, sub)
                eg = enet(g, V, T, yval, yte, sub)
                ea = enet(list(range(V.shape[1])), V, T, yval, yte, sub)
                for e in EVALS:
                    rows.append(
                        {
                            "family": fam,
                            "reservoir": res,
                            "D": D,
                            "eval": e,
                            "n_models": n,
                            "n_in_ensemble": len(use),
                            "f4_first6": np.mean(
                                [m["ps"][e] for m in first6 if m["ps"][e] is not None]
                            )
                            if len(first6)
                            else None,
                            "n_first6": len(first6),
                            "top8_by_val": np.mean(
                                [m["ps"][e] for m in top8 if m["ps"][e] is not None]
                            ),
                            "best_single": pear(T[e][:, best], yte[e]),
                            "enet_greedy_k5": eg[e],
                            "enet_all_pos": ea[e],
                        }
                    )
                print(fam, res, D, "n", n, {e: round(eg[e], 4) for e in EVALS}, flush=True)
    keys = list(rows[0].keys())
    with open(f"{a.out}/scaling_oct14_long.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    json.dump(rows, open(f"{a.out}/scaling_oct14_long.json", "w"), default=float)
    print("wrote", f"{a.out}/scaling_oct14_long.csv", len(rows), "rows")


if __name__ == "__main__":
    main()
