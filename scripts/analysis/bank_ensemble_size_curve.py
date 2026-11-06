#!/usr/bin/env python
"""How many tuned-LegNet seeds does an ensemble actually need?

The scaling program trains 20 seeds per cell, which is the expensive part of every reservoir comparison. The seeds differ only in initialisation and data order, so the stack's gain over a single model is pure
variance reduction and should saturate quickly. This measures where: for each size, draw random subsets of K of the 20 seeds, fit the usual positive-ElasticNet stack on V0c and score on the test sets, averaging
over draws. Reported as the fraction of the full 20-seed gain that K seeds recover, which is what decides whether a provisional scaling run can use 3 instead of 20.

    python scripts/analysis/bank_ensemble_size_curve.py [--draws 25]      (cpuq)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "analysis"))
from compare_legnet_vs_loop import MR, SETS, load, pearson, stack_scores  # noqa: E402

BANKS = {
    "10k": "bench38_ln_e800_wd4_10k_c20",
    "30k": "bench38_ln_e500_wd2_30k_c20",
    "100k": "bench38_ln_e500_100k_c20",
    "300k": "bench38_ln_e300_wd05_300k_c20",
}
KS = (1, 2, 3, 4, 5, 6, 8, 10, 15, 20)
METRICS = ("genomic", "ood", "mean18")


def score(ms: list, yv, y) -> dict:
    s = stack_scores(ms, yv, y) if len(ms) > 1 else None
    if s is None:  # a single model is its own prediction, no stack to fit
        t = ms[0][1]
        s = {k: pearson(t[k], y[k]) for k in SETS}
    s["mean18"] = float(np.mean([s[k] for k in SETS]))
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=25)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    res: dict = {}
    for size, d in BANKS.items():
        c = load(f"{MR}/{d}", "r*_menu_*.npz")
        if not c or len(c["ms"]) < 3:
            print(f"{size}: bank not ready ({0 if not c else len(c['ms'])} seeds)")
            continue
        n = len(c["ms"])
        print(f"\n== {size}: {n} seeds")
        print(
            f"{'K':>3s}  "
            + "  ".join(f"{m:>8s}" for m in METRICS)
            + "   % of the full-bank gain (genomic)"
        )
        rows = {}
        for k in [k for k in KS if k <= n]:
            vals = []
            for _ in range(1 if k == n else a.draws):
                idx = rng.choice(n, size=k, replace=False)
                vals.append(score([c["ms"][i] for i in idx], c["yv"], c["y"]))
            rows[k] = {m: float(np.mean([v[m] for v in vals])) for m in METRICS}
        lo, hi = rows[1]["genomic"], rows[max(rows)]["genomic"]
        for k, v in rows.items():
            frac = (v["genomic"] - lo) / (hi - lo) * 100 if hi > lo else float("nan")
            print(f"{k:3d}  " + "  ".join(f"{v[m]:8.4f}" for m in METRICS) + f"   {frac:5.1f}%")
        res[size] = {"n_seeds": n, "rows": rows}
    os.makedirs("outputs/justification_oct22", exist_ok=True)
    json.dump(res, open("outputs/justification_oct22/data_bank_size_curve.json", "w"), indent=1)


if __name__ == "__main__":
    main()
