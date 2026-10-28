# ruff: noqa
#!/usr/bin/env python
"""Cadence study at an equal number of models: ensemble score against the number of trained models, per arm and reservoir.

Each cadence cell (outputs/cadence/<reservoir>_<arm>) trained its models in order, so the deploy ensemble over the first k models is what that arm would have delivered on a budget of k.
Same selector and scorer as scripts/analysis/cadence_results.py (greedy 5-member ensemble fitted on validation predictions, scored on the genomic test set), computed afresh from the current
cell directories, budgets 6, 12, ..., 60. The three reservoirs (genomic, motif CT-enriched, random) are the replicates.

    python scripts/analysis/analyze_cadence_budget.py [--out outputs/justification_oct22]      (cpuq)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "analysis"))
from scripts.analysis.hp_protocol_study import greedy, load_cell  # noqa: E402
from scripts.analysis.cadence_results import ordered  # noqa: E402

BUDGETS = list(range(6, 61, 6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    a = ap.parse_args()
    res = {}
    for cd in sorted(glob.glob("outputs/cadence/*_*/")):
        cell = os.path.basename(cd.rstrip("/"))
        if cell.startswith("_"):
            continue
        models, lab = load_cell(cd.rstrip("/"))
        if not models or len(models) < 6:
            continue
        yv, yt = lab
        order = {
            os.path.basename(f).replace("_meta.json", ""): i
            for i, f in enumerate(ordered(cd.rstrip("/")))
        }
        models.sort(key=lambda m: order.get(m["id"], 999))
        curve = []
        for k in BUDGETS:
            if k > len(models):
                break
            _, s = greedy(models[:k], yv, yt)
            curve.append([k, float(s)])
        res[cell] = {"n": len(models), "curve": curve}
        print(
            f"{cell:<34} n={len(models):3d} " + "  ".join(f"{k}:{s:.4f}" for k, s in curve[-4:]),
            flush=True,
        )
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/data_cadence_budget.json", "w"), indent=1)
    print("wrote", f"{a.out}/data_cadence_budget.json")


if __name__ == "__main__":
    main()
