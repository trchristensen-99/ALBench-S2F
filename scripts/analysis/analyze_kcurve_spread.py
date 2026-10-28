# ruff: noqa
#!/usr/bin/env python
"""How much does the number-of-configurations curve depend on WHICH configurations are trained? Random orderings of the retrained menu configurations.

analyze_oct27.an_kcurve reports one ordering of the 30 retrained configurations (the best two of every search run first). Here: for every retrained pool, n_perm random orderings;
for each K the ensemble is the 8 best of the first K configurations by retrained validation r (equal weights), the same construction, and the score is its test r minus the
test r of the 8 best of all configurations. Pearson of a mean of columns is computed from the prediction covariance matrix, so a curve costs microseconds.

    python scripts/analysis/analyze_kcurve_spread.py [--n_perm 300] [--out outputs/justification_oct22]      (cpuq)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_oct26 as b26  # noqa: E402

a = b26.a
EV, MR_, RES, load = b26.EV, b26.MR_, b26.RES, b26.load


def pools(st, D):
    k = D // 1000
    if D == 30000:
        return [f"ind30k_{st}_r{r}" for r in (1, 2, 3)]
    tag = f"un{k}k_{st}_r"
    if all(os.path.exists(f"{MR_}/{tag}{r}/COMPLETE_2of3") for r in (1, 2, 3)):
        return [f"{tag}{r}" for r in (1, 2, 3)]
    return [f"lad_{st}_r1_d{k}k"] if st != "genomic" else [f"nest_genomic_r1_d{k}k"]


def pool_curves(c, cols, n_perm, rng):
    """-> {set: (n_perm, n)} test r of the ensemble of the 8 best (by validation r) of the first K configurations, minus the same for all configurations."""
    val = np.array([a.mj.pear(c["V"][:, j], c["yv"]) for j in cols])
    n = len(cols)
    out = {}
    for e in EV:
        T = c["T"][e][:, cols].astype(np.float64)
        y = np.asarray(c["yt"][e], np.float64)
        T = T - T.mean(0)
        y = y - y.mean()
        C = T.T @ T
        cy = T.T @ y
        vy = float(y @ y)

        def r(sel):
            w = np.zeros(n)
            w[list(sel)] = 1.0 / len(sel)
            return float(w @ cy / np.sqrt(max(float(w @ C @ w), 1e-30) * vy))

        ref = r(np.argsort(-val)[:8])
        curve = np.zeros((n_perm, n))
        for p in range(n_perm):
            perm = rng.permutation(n)
            for k in range(1, n + 1):
                first = perm[:k]
                sel = first[np.argsort(-val[first])[:8]]
                curve[p, k - 1] = r(sel) - ref
        out[e] = curve
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_perm", type=int, default=300)
    ap.add_argument("--out", default="outputs/justification_oct22")
    args = ap.parse_args()
    menu = json.load(open("outputs/menu_retrain_v0/menu_v0.json"))
    skip = {"son55_rep1_r00_arch_00", "son55_rep2_r15_arch_31", "son45_rep1_r11_arch_15"}
    usable = {i for i, e in enumerate(menu) if e["id"] not in skip}
    rng = np.random.default_rng(12345)
    res = {}
    for st in RES:
        for D in (10000, 30000, 100000):
            loaded = [
                c
                for c in (load([d], rounds=[0]) for d in pools(st, D))
                if c is not None and c["V"].shape[1] >= 12
            ]
            if not loaded:
                continue
            common = set.intersection(*[{int(i) for i in c["idx"]} for c in loaded]) & usable
            per_pool = []
            for c in loaded:
                cols = [j for j, i in enumerate(c["idx"]) if int(i) in common]
                per_pool.append(pool_curves(c, cols, args.n_perm, rng))
            n = min(next(iter(pc.values())).shape[1] for pc in per_pool)
            cell = {"n_pools": len(per_pool), "n_configs": n, "n_perm": args.n_perm}
            for e in EV:
                allc = np.concatenate([pc[e][:, :n] for pc in per_pool], 0)
                cell[e] = {
                    "mean": allc.mean(0).tolist(),
                    "sd": allc.std(0).tolist(),
                    "p05": np.percentile(allc, 5, 0).tolist(),
                    "p25": np.percentile(allc, 25, 0).tolist(),
                    "p75": np.percentile(allc, 75, 0).tolist(),
                    "p95": np.percentile(allc, 95, 0).tolist(),
                    "pool_means": [pc[e][:, :n].mean(0).tolist() for pc in per_pool],
                }
            res[f"{st}|{D}"] = cell
            print(
                f"{st} {D // 1000}k: {len(per_pool)} pools x {args.n_perm} orderings, {n} configurations; genomic K=12 mean {cell['genomic']['mean'][11]:+.4f} (sd {cell['genomic']['sd'][11]:.4f}), K=20 {cell['genomic']['mean'][min(19, n - 1)]:+.4f}",
                flush=True,
            )
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(f"{args.out}/data_kspread.json", "w"))
    print("wrote", f"{args.out}/data_kspread.json")


if __name__ == "__main__":
    main()
