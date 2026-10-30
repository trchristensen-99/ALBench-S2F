#!/usr/bin/env python
"""Does pooling variants of the same configurations help the ensemble? On the canonical 30k genomic pool the menu configurations were retrained under several input and augmentation variants
(context 4 to 200 bases, shift variants, weight averaging, oracle-like augmentation, menu v1c). Each variant is a different input window or training pipeline of the same architectures, so its
predictions are partly decorrelated from the others. For growing sets of variants: number of models, mean pairwise correlation of the genomic-test predictions, and the positive-ElasticNet stack
(fitted on V0c) on the genomic test, designed set and random sequences. A control with the same number of models from one variant is not possible (one seed each), so the comparison is
"15 models of the standard variant" against "the same plus other variants".

    python scripts/analysis/analyze_pool_variants.py [--out outputs/justification_oct22]      (cpuq)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "analysis"))
from analyze_extras38 import SETS, load, pearson  # noqa: E402

from albench.ensemble.stack import fit_zscored_stack  # noqa: E402

STD = ["fx35_c20s10_30k"]
STEPS = [
    ("standard input (20 flank bases, shift 10): 15 configurations", STD),
    (
        "+ other context sizes (4, 8, 12, 60, 100, 200)",
        STD + [f"fx35_c{c}s10_30k" for c in (4, 8, 12, 60, 100, 200)],
    ),
    ("+ shift variants (no shift, +-30, per-sample, oracle-like, shift-TTA)", None),
    ("+ weight averaging and batch-norm recalculation", None),
    ("+ menu v1c (20 configurations, 5 rewritten)", None),
]
ADD = {
    2: [
        "fx35_c20s0_30k",
        "fx35_c20s30_30k",
        "fx36_c20_ps10_30k",
        "fx36_c20_oracle_30k",
        "fx36_c20_tta8_30k",
    ],
    3: ["fe35_c20s10_emabn_30k"],
    4: ["mv1c_c20_30k"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    a = ap.parse_args()
    cells = {}

    def get(d):
        if d not in cells:
            cells[d] = load(d)
        return cells[d]

    base = get(STD[0])
    yv = base["yv"]
    names, out = list(STD), []
    for i, (label, first) in enumerate(STEPS):
        if first is not None:
            names = list(first)
        else:
            names = names + ADD[i]
        rows = []
        for d in names:
            c = get(d)
            if c is None or not np.allclose(c["yv"], yv):
                continue
            for k, m in sorted(c["ms"].items()):
                rows.append((d, k, m))
        if len(rows) < 5:
            continue
        V = np.stack([m["v"] for _, _, m in rows])
        st = fit_zscored_stack(V, yv)
        res = {"step": label, "n_models": len(rows), "n_cells": len(set(d for d, _, _ in rows))}
        G = np.stack([m["t"]["genomic"] for _, _, m in rows])
        C = np.corrcoef(G)
        res["mean_pairwise_corr_genomic"] = float(C[np.triu_indices(len(rows), 1)].mean())
        for s in SETS:
            T = np.stack([m["t"][s] for _, _, m in rows])
            res[f"stack_{s}"] = pearson(
                st.predict(T), base["lab"][f"oracle_{s}"].astype(np.float64)
            )
            res[f"mean_single_{s}"] = float(
                np.mean([pearson(t, base["lab"][f"oracle_{s}"].astype(np.float64)) for t in T])
            )
        out.append(res)
        print(
            f"{label[:70]:70s} n={res['n_models']:3d} corr {res['mean_pairwise_corr_genomic']:.3f} stack genomic {res['stack_genomic']:.4f} ood {res['stack_ood']:.4f} random {res['stack_random_32k']:.4f}",
            flush=True,
        )
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/data_pool_variants.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
