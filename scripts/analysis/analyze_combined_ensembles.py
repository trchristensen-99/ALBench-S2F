#!/usr/bin/env python
"""Ensembles that combine the published MPRA-LegNet seeds (with the 20/28-base context input) and the LLM-found menu configurations, all trained on the canonical 30k genomic pool.

Steps add model sets to one positive-ElasticNet stack fitted on V0c; for each: number of models, mean pairwise correlation of the genomic-test predictions, stack r on the genomic test, designed
set and random sequences (oracle labels) and on the genomic test against the EXPERIMENTAL labels. Test labels are never used for a choice.

    python scripts/analysis/analyze_combined_ensembles.py [--out outputs/justification_oct22]      (cpuq)
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

STEPS = [
    ("LegNet e100 + context 28 (20 seeds)", ["bench38_ln_e100_30k_c28"]),
    ("LegNet b256 + context 28 (20 seeds)", ["bench38_ln_b256_30k_c28"]),
    (
        "LegNet e100 + b256, context 20 and 28 (80 models)",
        [
            "bench38_ln_e100_30k_c28",
            "bench38_ln_e100_30k_c20",
            "bench38_ln_b256_30k_c28",
            "bench38_ln_b256_30k_c20",
        ],
    ),
    ("menu v1c (20 configurations)", ["mv1c_c20_30k"]),
    ("menu v1c + LegNet e100 c28", ["mv1c_c20_30k", "bench38_ln_e100_30k_c28"]),
    (
        "menu v1c + all four LegNet cells",
        [
            "mv1c_c20_30k",
            "bench38_ln_e100_30k_c28",
            "bench38_ln_e100_30k_c20",
            "bench38_ln_b256_30k_c28",
            "bench38_ln_b256_30k_c20",
        ],
    ),
    (
        "everything: menu variants (context sizes, shifts, aug, v1c) + four LegNet cells",
        [
            "mv1c_c20_30k",
            "bench38_ln_e100_30k_c28",
            "bench38_ln_e100_30k_c20",
            "bench38_ln_b256_30k_c28",
            "bench38_ln_b256_30k_c20",
            "fx35_c20s10_30k",
        ]
        + [f"fx35_c{c}s10_30k" for c in (4, 8, 12, 60, 100, 200)]
        + [
            "fx35_c20s0_30k",
            "fx35_c20s30_30k",
            "fx36_c20_ps10_30k",
            "fx36_c20_oracle_30k",
            "fx36_c20_tta8_30k",
            "fe35_c20s10_emabn_30k",
        ],
    ),
]


LN10 = [f"bench38_ln_{r}_10k_c{c}" for r in ("e100", "b256", "pub") for c in (20, 28)]
STEPS_10K = [
    ("LegNet (e100, b256, pub; context 20 and 28; 120 models)", LN10),
    (
        "collaborator-style LegNet reference (lg38, 3 seeds, + weight averaging)",
        ["lg38_c20_10k", "lg38_c20_emabn_10k"],
    ),
    ("menu v1c (hp38_base, 20 configurations)", ["hp38_base_genomic_10k"]),
    ("menu v1c + LegNet cells", ["hp38_base_genomic_10k"] + LN10),
    (
        "menu v1c + LegNet cells + lg38 reference",
        ["hp38_base_genomic_10k"] + LN10 + ["lg38_c20_10k", "lg38_c20_emabn_10k"],
    ),
    (
        "everything at 10k (adds the recipe-sweep variants of the menu)",
        ["hp38_base_genomic_10k"]
        + LN10
        + ["lg38_c20_10k", "lg38_c20_emabn_10k"]
        + [f"hp38_{x}_genomic_10k" for x in ("lrx05", "lrx2", "bs128", "bs512", "wd1e3", "wd1e2")],
    ),
]


LN100 = [f"bench38_ln_{r}_100k_c20" for r in ("e100", "b256")]
STEPS_100K = [
    ("LegNet e100 (10 seeds)", ["bench38_ln_e100_100k_c20"]),
    ("LegNet e100 + b256 (20 models)", LN100),
    (
        "collaborator-style LegNet reference (lg38, 3 seeds, + weight averaging)",
        ["lg38_c20_100k", "lg38_c20_emabn_100k"],
    ),
    ("menu v1c (hp38_base, 20 configurations)", ["hp38_base_genomic_100k"]),
    ("menu v2 (20 configurations)", ["v2m38_genomic_100k"]),
    ("menu v1c with single-pass RC augmentation (rc38_rca)", ["rc38_rca_genomic_100k"]),
    ("menu v2 + LegNet e100", ["v2m38_genomic_100k", "bench38_ln_e100_100k_c20"]),
    (
        "menu v1c + menu v2 + LegNet e100 + b256",
        ["hp38_base_genomic_100k", "v2m38_genomic_100k"] + LN100,
    ),
    (
        "everything at 100k (adds the recipe-sweep variants, rc38_rca and lg38)",
        ["hp38_base_genomic_100k", "v2m38_genomic_100k", "rc38_rca_genomic_100k"]
        + LN100
        + ["lg38_c20_100k", "lg38_c20_emabn_100k"]
        + [f"hp38_{x}_genomic_100k" for x in ("lrx05", "lrx2", "bs128", "bs512", "wd1e3")],
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    ap.add_argument("--size", default="30k", choices=["30k", "10k", "100k"])
    a = ap.parse_args()
    steps = {"30k": STEPS, "10k": STEPS_10K, "100k": STEPS_100K}[a.size]
    cache, out = {}, []
    ref = None
    for label, dirs in steps:
        rows = []
        for d in dirs:
            if d not in cache:
                cache[d] = load(d)
            c = cache[d]
            if c is None:
                continue
            if ref is None:
                ref = c
            if not np.allclose(c["yv"], ref["yv"]):
                continue
            rows += [(d, k, m) for k, m in sorted(c["ms"].items())]
        if len(rows) < 5:
            continue
        V = np.stack([m["v"] for _, _, m in rows])
        st = fit_zscored_stack(V, ref["yv"])
        res = {"step": label, "n_models": len(rows)}
        G = np.stack([m["t"]["genomic"] for _, _, m in rows])
        res["mean_pairwise_corr_genomic"] = float(
            np.corrcoef(G)[np.triu_indices(len(rows), 1)].mean()
        )
        for s in SETS:
            T = np.stack([m["t"][s] for _, _, m in rows])
            res[f"stack_{s}"] = pearson(st.predict(T), ref["lab"][f"oracle_{s}"].astype(np.float64))
        if "test_true" in ref["lab"].files:
            res["stack_genomic_measured"] = pearson(
                st.predict(G), ref["lab"]["test_true"].astype(np.float64)
            )
        out.append(res)
        print(
            f"{label[:74]:74s} n={res['n_models']:3d} corr {res['mean_pairwise_corr_genomic']:.3f} genomic {res['stack_genomic']:.4f} (measured {res.get('stack_genomic_measured', float('nan')):.4f}) ood {res['stack_ood']:.4f} random {res['stack_random_32k']:.4f}",
            flush=True,
        )
    os.makedirs(a.out, exist_ok=True)
    stem = "data_combined_ensembles" if a.size == "30k" else f"data_combined_ensembles_{a.size}"
    json.dump(out, open(f"{a.out}/{stem}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
