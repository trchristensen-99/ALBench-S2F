# ruff: noqa
#!/usr/bin/env python
"""Context-size and shift-augmentation sweeps (cells sw38_<D>k_<reservoir>_<arm>, see scripts/program_oct22/register_sw38.py).

Per cell: the first configurations of menu v1 that take wider input (indices 1 to 6; 1 to 4 at 100k), one seed. For each (size, reservoir, arm) and evaluation set (genomic test, designed high-activity, random sequences, SNV
effect) the single-model r of each configuration is computed from the saved predictions and compared, configuration by configuration, with the no-context arm and with the 20-base arm of the same reservoir and size.
Aggregates pool the (reservoir, configuration) pairs of a size: mean paired change, standard error over pairs, number of pairs.

    python scripts/analysis/analyze_sw38.py [--out outputs/justification_oct22]      (cpuq)
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

MR = "outputs/menu_retrain_v0"
SETS = ["genomic", "ood", "random_32k", "snv_ref", "snv_alt"]
OUT = ["genomic", "ood", "random_32k", "snv_delta"]
RES = ["genomic", "random", "motif_ct_enriched", "evoaug_paper2023", "evo2"]
DS = [10000, 30000, 100000]
ARMS = [
    "base",
    "c12",
    "c16",
    "c20",
    "c24",
    "c28",
    "c32",
    "c40",
    "c48",
    "c60",
    "s0",
    "s5",
    "s15",
    "s20",
    "s30",
    "orc",
    "rca",
]
CTX_F = {
    "base": 0,
    "c12": 12,
    "c16": 16,
    "c20": 20,
    "c24": 24,
    "c28": 28,
    "c32": 32,
    "c40": 40,
    "c48": 48,
    "c60": 60,
}
SHIFT_S = {"s0": 0, "s5": 5, "c20": 10, "s15": 15, "s20": 20, "s30": 30}


def pearson(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def load(d, Y):
    out = {}
    for mf in glob.glob(f"{MR}/{d}/r00_menu_*_meta.json"):
        i = int(os.path.basename(mf).split("_")[2])
        if i < 1:
            continue
        try:
            z = np.load(mf.replace("_meta.json", ".npz"))
            t = {s: z[f"test_pred_{s}"].astype(np.float64) for s in SETS}
        except Exception:
            continue
        if any(not np.isfinite(x).all() or np.abs(x).max() > 1e3 for x in t.values()):
            continue
        r = {s: pearson(t[s], Y[s]) for s in SETS[:3]}
        r["snv_delta"] = pearson(t["snv_alt"] - t["snv_ref"], Y["snv_alt"] - Y["snv_ref"])
        out[i] = r
    return out


def agg(pairs):
    """pairs: list of paired differences per evaluation set -> mean, se, n"""
    res = {}
    for s in OUT:
        d = np.array([p[s] for p in pairs], np.float64)
        d = d[np.isfinite(d)]
        res[s] = {
            "mean": float(d.mean()) if len(d) else float("nan"),
            "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
            "n": int(len(d)),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    a = ap.parse_args()
    cells, labels = {}, {}
    for D in DS:
        for res in RES:
            for arm in ARMS:
                d = f"sw38_{D // 1000}k_{res}_{arm}"
                if not os.path.exists(f"{MR}/{d}/labels.npz"):
                    continue
                if (D, res) not in labels:
                    lab = np.load(f"{MR}/{d}/labels.npz")
                    labels[(D, res)] = {s: lab[f"oracle_{s}"].astype(np.float64) for s in SETS}
                cells[(D, res, arm)] = load(d, labels[(D, res)])
    out = {"cells": {}, "vs_base": {}, "vs_c20": {}, "by_reservoir_vs_base": {}}
    for (D, res, arm), ms in cells.items():
        out["cells"][f"{D}|{res}|{arm}"] = {
            "n": len(ms),
            "single": {s: float(np.nanmean([m[s] for m in ms.values()])) for s in OUT}
            if ms
            else None,
        }
    for D in DS:
        for arm in ARMS:
            for ref, key in (("base", "vs_base"), ("c20", "vs_c20")):
                pairs = []
                for res in RES:
                    A, B = cells.get((D, res, arm)), cells.get((D, res, ref))
                    if not A or not B or arm == ref:
                        continue
                    for i in sorted(set(A) & set(B)):
                        pairs.append({s: A[i][s] - B[i][s] for s in OUT})
                if pairs:
                    out[key][f"{D}|{arm}"] = agg(pairs)
            for res in RES:
                A, B = cells.get((D, res, arm)), cells.get((D, res, "base"))
                if A and B and arm != "base":
                    pairs = [{s: A[i][s] - B[i][s] for s in OUT} for i in sorted(set(A) & set(B))]
                    if pairs:
                        out["by_reservoir_vs_base"][f"{D}|{res}|{arm}"] = agg(pairs)
    for k, v in out["vs_base"].items():
        print(
            k,
            {s: (round(v[s]["mean"], 4), v[s]["n"]) for s in ("genomic", "snv_delta")},
            flush=True,
        )
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/data_sw38.json", "w"), indent=1, default=float)
    print(
        "wrote", f"{a.out}/data_sw38.json", "cells with data:", sum(1 for v in cells.values() if v)
    )


if __name__ == "__main__":
    main()
