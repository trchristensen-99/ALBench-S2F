#!/usr/bin/env python
"""Single networks and small averages of the tuned LegNet on the same two genomic-test label sets the collaborator reports: oracle labels ("Genomic r") and experimental labels ("Measured genomic r").

For each size: the mean over seeds of a single network, the single network picked by validation r, and the plain prediction average of 2 and of 3 random seeds (his "ensembles" are two or three networks inside one
model), averaged over 300 random draws. Test labels are never used for a choice.

    python scripts/analysis/measured_singles.py      (cpuq)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "analysis"))
from analyze_extras38 import load, pearson  # noqa: E402

CELLS = {
    "10k e100 (published)": "bench38_ln_e100_10k_c20",
    "10k e800_wd4 (tuned)": "bench38_ln_e800_wd4_10k_c20",
    "30k e100 (published)": "bench38_ln_e100_30k_c20",
    "30k e500_wd2 (tuned)": "bench38_ln_e500_wd2_30k_c20",
    "30k e300_wd2": "bench38_ln_e300_wd2_30k_c20",
    "100k e100 (published)": "bench38_ln_e100_100k_c20",
    "100k e500 (tuned)": "bench38_ln_e500_100k_c20",
}


def main() -> None:
    rng = np.random.default_rng(0)
    for name, cell in CELLS.items():
        c = load(cell)
        if not c or not c["ms"]:
            continue
        keys = sorted(c["ms"])
        yo = c["lab"]["oracle_genomic"].astype(np.float64)
        ym = c["lab"]["test_true"].astype(np.float64)
        T = np.stack([c["ms"][k]["t"]["genomic"] for k in keys])
        valr = np.array([c["ms"][k]["valr"] for k in keys])
        so = np.array([pearson(t, yo) for t in T])
        sm = np.array([pearson(t, ym) for t in T])
        b = int(np.argmax(valr))
        out = [
            f"{name:24s} n={len(keys):2d} single mean oracle {so.mean():.4f} measured {sm.mean():.4f} | best-by-val {so[b]:.4f} / {sm[b]:.4f}"
        ]
        for n in (2, 3):
            ro, rm = [], []
            for _ in range(300):
                p = T[rng.choice(len(keys), n, replace=False)].mean(axis=0)
                ro.append(pearson(p, yo))
                rm.append(pearson(p, ym))
            out.append(f"avg of {n}: {np.mean(ro):.4f} / {np.mean(rm):.4f}")
        print(" | ".join(out), flush=True)


if __name__ == "__main__":
    main()
