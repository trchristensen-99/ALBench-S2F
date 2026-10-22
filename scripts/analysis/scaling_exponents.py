"""Fit err = a * D^(-b) per (arm, eval set) and report the exponent b.

b is the thing the scaling curves exist to measure: how fast error falls as the
training set grows. Fitted by OLS on log(1-r) vs log(D). R^2 is reported beside
it because a 3-point fit can look tidy while hiding curvature -- a low R^2 means
the power-law read is not safe for that arm/eval combination.
"""

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

rows = [r for r in csv.DictReader(open("outputs/figures/eval_matrix.csv")) if r["complete"] == "1"]
EVALS = ["genomic", "ood", "snv_delta", "random_32k", "inversion"]

by = defaultdict(lambda: defaultdict(dict))
for r in rows:
    for e in EVALS:
        if r.get(e):
            by[r["arm"]][e][int(r["D"])] = float(r[e])

print(f"{'arm':<22}" + "".join(f"{e[:11]:>15}" for e in EVALS))
print(f"{'':22}" + "".join(f"{'b (R2)':>15}" for _ in EVALS))
out = {}
for arm in sorted(by):
    cells = []
    for e in EVALS:
        pts = sorted(by[arm][e].items())
        if len(pts) < 3:
            cells.append(f"{'--':>15}")
            continue
        D = np.log10([p[0] for p in pts])
        y = np.log10([max(1e-6, 1.0 - p[1]) for p in pts])
        b, a = np.polyfit(D, y, 1)
        pred = a + b * D
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        out[(arm, e)] = (-b, r2)
        cells.append(f"{-b:>8.3f} ({r2:.2f})")
    print(f"{arm:<22}" + "".join(c.rjust(15) for c in cells))

print("\n  b = decay exponent (higher = error falls faster with scale)")
for e in EVALS:
    vs = [(a, v[0]) for (a, ee), v in out.items() if ee == e]
    if vs:
        vs.sort(key=lambda t: -t[1])
        med = np.median([v for _, v in vs])
        print(f"  {e:<12} median b={med:.3f}   fastest: {vs[0][0]} ({vs[0][1]:.3f})   "
              f"slowest: {vs[-1][0]} ({vs[-1][1]:.3f})")
