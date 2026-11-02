#!/usr/bin/env python
"""What the AlphaGenome-FT students are, and how they score against oracle and experimental labels.

For each bench33_ag cell (S1: head on the frozen encoder; S2: fully fine-tuned from the S1 checkpoint; both trained on the same nested 30k or 10k genomic pool, selected on V0c): training size, whether the S1
checkpoint was loaded, mean single-model r and stack r on the genomic test set against the oracle labels and against the experimental (measured) labels, and on the other test sets that the cell holds (oracle labels only:
no experimental labels exist for them).

    python scripts/analysis/ag_check.py      (cpuq)
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from albench.ensemble.stack import fit_zscored_stack  # noqa: E402

MR = "outputs/menu_retrain_v0"
SETS = (
    "genomic",
    "ood",
    "random_32k",
    "gc_sweep",
    "dinuc_shuffle",
    "sub_med",
    "ins_med",
    "del_med",
    "translocation",
    "inversion",
)


def pr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.corrcoef(a, b)[0, 1])


for cell in (
    "bench33_agS1_oracle_10k",
    "bench33_agS2_oracle_10k",
    "bench33_agS1_oracle_30k",
    "bench33_agS2_oracle_30k",
    "bench33_agS2_winner_30k",
    "bench33_agS2_oracle_30k_w256",
):
    d = f"{MR}/{cell}"
    if not os.path.exists(f"{d}/labels.npz"):
        continue
    lab = np.load(f"{d}/labels.npz")
    yv = lab["val_labels"].astype(np.float64)
    ms, metas = [], []
    for mf in sorted(glob.glob(f"{d}/r*_menu_*_meta.json")):
        try:
            z = np.load(mf.replace("_meta.json", ".npz"))
            ms.append(
                (
                    z["val_pred"].astype(np.float64),
                    {
                        s: z[f"test_pred_{s}"].astype(np.float64)
                        for s in SETS
                        if f"test_pred_{s}" in z.files
                    },
                )
            )
            metas.append(json.load(open(mf)))
        except Exception:
            continue
    if len(ms) < 3:
        continue
    m0 = metas[0]
    have = [s for s in SETS if all(s in m[1] for m in ms)]
    st = fit_zscored_stack(np.stack([m[0] for m in ms]), yv)
    print(
        f"\n{cell}: n={len(ms)} n_train={m0.get('n_train')} stage={m0.get('stage')} pool={os.path.basename(str(m0.get('pool')))} s1_loaded={m0.get('s1_loaded')} s1_ckpt={str(m0.get('s1_checkpoint'))[-45:]}"
    )
    yo, ym = lab["oracle_genomic"].astype(np.float64), lab["test_true"].astype(np.float64)
    G = np.stack([m[1]["genomic"] for m in ms])
    print(
        f"   genomic  oracle labels: single {np.mean([pr(g, yo) for g in G]):.4f}  stack {pr(st.predict(G), yo):.4f} | measured labels: single {np.mean([pr(g, ym) for g in G]):.4f}  stack {pr(st.predict(G), ym):.4f}"
    )
    for s in have:
        if s == "genomic":
            continue
        T = np.stack([m[1][s] for m in ms])
        y = lab[f"oracle_{s}"].astype(np.float64)
        print(
            f"   {s:14s} oracle labels: single {np.mean([pr(t, y) for t in T]):.4f}  stack {pr(st.predict(T), y):.4f}"
        )
