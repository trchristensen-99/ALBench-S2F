#!/usr/bin/env python
"""Early-stopping patience beyond 16: ensemble and member r of the retrained menu (v0, 33 configurations, one seed, bare-insert students, pool r1) with patience 4, 16, 24 and 32, paired by configuration.

python scripts/analysis/analyze_patience.py      (cpuq)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "analysis"))
from analyze_extras38 import SETS, load, pearson  # noqa: E402

from albench.ensemble.stack import fit_zscored_stack  # noqa: E402

for size in (10, 30, 100):
    for res in ("genomic", "random", "motif_ct_enriched"):
        base = load(f"tr{size}k_pat16_{res}_r1")
        if not base or not base["ms"]:
            continue
        y = {s: base["lab"][f"oracle_{s}"].astype(np.float64) for s in SETS}
        print(f"\n{size}k {res} (pool r1)")
        for pat in (4, 16, 24, 32):
            c = load(f"tr{size}k_pat{pat}_{res}_r1")
            if not c or len(c["ms"]) < 3:
                continue
            keys = sorted(set(c["ms"]) & set(base["ms"]))
            st = fit_zscored_stack(np.stack([c["ms"][k]["v"] for k in keys]), c["yv"])
            stack = {
                s: pearson(st.predict(np.stack([c["ms"][k]["t"][s] for k in keys])), y[s])
                for s in SETS
            }
            d = np.array(
                [
                    pearson(c["ms"][k]["t"]["genomic"], y["genomic"])
                    - pearson(base["ms"][k]["t"]["genomic"], y["genomic"])
                    for k in keys
                ]
            )
            print(
                f"  patience {pat:2d}: n={len(keys):2d} stack genomic {stack['genomic']:.4f} ood {stack['ood']:.4f} random {stack['random_32k']:.4f} | member genomic vs patience 16: {d.mean():+.4f} (SE {d.std(ddof=1) / np.sqrt(len(d)):.4f})"
            )
