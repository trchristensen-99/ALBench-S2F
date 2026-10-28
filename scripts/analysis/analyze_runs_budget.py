# ruff: noqa
#!/usr/bin/env python
"""One long AutoResearch run or several shorter ones? Ensemble performance and its variability at a fixed total number of trained models.

Pool: the 12 guided long search runs on the 30k genomic pool (outputs/archsearch_final_oct20: opus55, son45, son55, son55_wide, 3 runs each, 30 rounds x 2 proposals; the unguided control family
is left out). For a total of M models split over r runs, each run contributes its first M/r candidates in proposal order, the pooled candidates are stacked (the study's positive ElasticNetCV on
z-scored validation predictions) and scored on the genomic / designed OOD / random test sets. Which r runs are used is drawn at random from the 12 (all combinations when there are few),
so the spread over draws is the variability of the resulting ensemble from the luck of the runs. Draws share the same 12 runs, so the spread understates the true run-to-run variability
when r is large (at r = 12 there is a single draw).

    python scripts/analysis/analyze_runs_budget.py [--n_draw 24] [--out outputs/justification_oct22]      (cpuq)
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rounds as ar  # noqa: E402

RUNS: dict = {}
SETS = ar.SETS
BUDGETS = (15, 30, 45, 60, 90, 120)
SPLITS = (1, 2, 3, 4, 5, 6)


def score(args):
    combo, per = args
    rows = []
    for k in combo:
        rows += RUNS[k]["rows"][:per]
    c = RUNS[combo[0]]
    return ar.stack_r(c, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_draw", type=int, default=24)
    ap.add_argument("--out", default="outputs/justification_oct22")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    for d in sorted(glob.glob("outputs/archsearch_final_oct20/*/rep*")):
        if "none" in d:
            continue
        c = ar.load(d)
        if c and len(c["rows"]) >= 60:
            RUNS[d] = c
    keys = sorted(RUNS)
    print(len(keys), "guided runs with >= 60 usable candidates", flush=True)
    ref = RUNS[keys[0]]
    for k in keys[1:]:
        assert np.allclose(RUNS[k]["val_y"], ref["val_y"]) and np.allclose(
            RUNS[k]["y"]["genomic"], ref["y"]["genomic"]
        ), "runs do not share validation / test labels"
    rng = np.random.default_rng(2024)
    jobs, meta = [], []
    for M in BUDGETS:
        for r in SPLITS:
            if M % r or M // r > 60 or r > len(keys):
                continue
            per = M // r
            from math import comb

            if comb(len(keys), r) <= a.n_draw:
                combos = list(itertools.combinations(range(len(keys)), r))
            else:
                seen = set()
                while len(seen) < a.n_draw:
                    seen.add(tuple(sorted(rng.choice(len(keys), r, replace=False).tolist())))
                combos = sorted(seen)
            for cb in combos:
                jobs.append(([keys[i] for i in cb], per))
                meta.append((M, r))
    print(len(jobs), "stacks to fit", flush=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = pool.map(score, jobs, chunksize=1)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
    out = {}
    for (M, r), v in zip(meta, res):
        cell = out.setdefault(
            f"{M}|{r}", {"M": M, "r": r, "per": M // r, "vals": {s: [] for s in SETS}}
        )
        for s in SETS:
            cell["vals"][s].append(v[s])
    for cell in out.values():
        for s in SETS:
            x = np.array(cell["vals"][s])
            cell[s] = {
                "mean": float(x.mean()),
                "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                "n_draw": len(x),
                "p05": float(np.percentile(x, 5)),
                "p95": float(np.percentile(x, 95)),
            }
        print(
            f"M={cell['M']:3d} r={cell['r']} ({cell['per']:2d} each): genomic {cell['genomic']['mean']:.4f} sd {cell['genomic']['sd']:.4f} (n={cell['genomic']['n_draw']})",
            flush=True,
        )
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/data_runs_budget.json", "w"))
    print("wrote", f"{a.out}/data_runs_budget.json")


if __name__ == "__main__":
    main()
