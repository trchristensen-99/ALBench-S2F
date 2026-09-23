#!/usr/bin/env python
"""Which HP axes vary with RESERVOIR, and which with DATASET SIZE?

Three things this answers, in order of how much they are worth trusting:

1. SINGLE BEST CONFIG per cell, variance decomposed by axis. Cleanest to interpret,
   but one config per cell is a noisy estimator.

2. DOES THE SPLIT SURVIVE MORE SEARCH? Recompute (1) using only k of each cell's
   models, sweeping k. If apparent reservoir-variance shrinks as k grows, the effect
   was search noise. If it holds, it is real. This is the control for the fact that
   our cells had unequal budgets (3-53 models against a target of 52).

3. ENSEMBLE-WEIGHTED HP values. We deploy an N=5 ElasticNet-weighted ensemble, not a
   single model, so the deployed "configuration" is a weighted blend. For each numeric
   axis we take the weight-weighted mean across ensemble members. A reservoir effect
   that exists for the single best model but vanishes under weighting is not one that
   affects what we actually ship.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import ElasticNetCV

NUMERIC = [
    "lr",
    "batch_size",
    "conv_dropout",
    "dense_dropout",
    "n_layers",
    "width_base",
    "ks",
    "weight_decay",
    "pct_start",
    "width_ratio",
    "pool_downsample",
]
LOGGED = {"lr", "weight_decay"}


def cell_models(cell: Path):
    """(hp dict, val score, val_pred) per trained model in a cell."""
    out = []
    for meta in sorted(glob.glob(str(cell / "**" / "*_meta.json"), recursive=True)):
        try:
            d = json.load(open(meta))
        except Exception:
            continue
        hp = d.get("hp")
        v = d.get("best_val_pearson") or d.get("val_pearson")
        if not hp or v is None:
            continue
        npz = meta.replace("_meta.json", ".npz")
        vp = None
        if os.path.exists(npz):
            try:
                with np.load(npz) as z:
                    vp = z["val_pred"] if "val_pred" in z else None
            except Exception:
                pass
        out.append((hp, float(v), vp))
    return out


def val_of(hp, k):
    v = hp.get(k)
    if v is None:
        return None
    return float(np.log10(v)) if (k in LOGGED and v > 0) else float(v)


def decompose(vals, arms, ds):
    """Share of total variance explained by the D marginal vs the reservoir marginal."""
    arr = np.array(list(vals.values()), float)
    if arr.size < 6 or arr.std() == 0:
        return None
    tot = arr.var()
    byD = [
        np.mean([v for (a, d), v in vals.items() if d == D])
        for D in ds
        if any(d == D for (a, d) in vals)
    ]
    byR = [
        np.mean([v for (a, d), v in vals.items() if a == A])
        for A in arms
        if any(a == A for (a, d) in vals)
    ]
    return np.var(byD) / tot, np.var(byR) / tot


def ensemble_weighted_hp(models, n=5):
    """Weight-weighted mean of each numeric axis across the selected N=5 ensemble."""
    usable = [(hp, v, vp) for hp, v, vp in models if vp is not None]
    if len(usable) < 2:
        return None
    L = min(len(vp) for _, _, vp in usable)
    V = np.vstack([vp[:L] for _, _, vp in usable])
    # proxy target: the mean of the top-quartile models, since labels are not needed
    # for a RELATIVE weighting and this keeps the function self-contained
    order = np.argsort([-v for _, v, _ in usable])
    target = V[order[: max(1, len(order) // 4)]].mean(axis=0)
    chosen: list[int] = []
    for _ in range(min(n, V.shape[0])):
        best, bm = None, np.inf
        for c in range(V.shape[0]):
            if c in chosen:
                continue
            m = float(np.mean((V[chosen + [c]].mean(axis=0) - target) ** 2))
            if m < bm:
                best, bm = c, m
        chosen.append(best)
    try:
        en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=5000, n_jobs=1)
        en.fit(V[chosen].T, target)
        w = np.abs(en.coef_)
    except Exception:
        w = np.ones(len(chosen))
    if w.sum() == 0:
        w = np.ones(len(chosen))
    w = w / w.sum()
    out = {}
    for k in NUMERIC:
        vs = [val_of(usable[i][0], k) for i in chosen]
        keep = [(wi, v) for wi, v in zip(w, vs) if v is not None]
        if keep:
            out[k] = float(sum(wi * v for wi, v in keep) / sum(wi for wi, _ in keep))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/stage4_base0")
    ap.add_argument(
        "--arms", default="random,motif_syntax_core,motif_ct_enriched,evoaug,mutagenesis"
    )
    ap.add_argument("--ds", default="10000,30000,100000,300000")
    ap.add_argument("--ks", default="5,10,20,40")
    ap.add_argument("--out", default="outputs/analysis/hp_axis_variance.json")
    args = ap.parse_args()
    ARMS = args.arms.split(",")
    DS = [int(x) for x in args.ds.split(",")]
    KS = [int(x) for x in args.ks.split(",")]
    rng = np.random.default_rng(0)

    cells = {}
    for a in ARMS:
        for D in DS:
            m = cell_models(Path(args.root) / f"{a}_d{D}")
            if m:
                cells[(a, D)] = m
    print(f"cells: {len(cells)}   models: {sum(len(v) for v in cells.values())}\n")

    res = {"n_models": {f"{a}_d{D}": len(v) for (a, D), v in cells.items()}}

    print("1. SINGLE BEST CONFIG per cell")
    print(f"   {'axis':16s}{'var by D':>10s}{'var by res':>12s}   verdict")
    best_hp = {c: max(v, key=lambda t: t[1])[0] for c, v in cells.items()}
    single = {}
    for k in NUMERIC:
        vals = {c: val_of(h, k) for c, h in best_hp.items() if val_of(h, k) is not None}
        dec = decompose(vals, ARMS, DS)
        if dec:
            fd, fr = dec
            single[k] = {"by_D": fd, "by_reservoir": fr}
            verdict = "D" if fd > 2 * fr else ("RESERVOIR" if fr > 2 * fd else "both/neither")
            print(f"   {k:16s}{fd:>10.2f}{fr:>12.2f}   {verdict}")
    res["single_best"] = single

    print("\n2. DOES THE SPLIT SURVIVE MORE SEARCH? (mean over 20 random subsets of size k)")
    print(f"   {'axis':16s}" + "".join(f"{'k=' + str(k):>16s}" for k in KS))
    print(f"   {'':16s}" + "".join(f"{'D / res':>16s}" for k in KS))
    sweep = {}
    for axis in NUMERIC:
        row, ok = f"   {axis:16s}", False
        sweep[axis] = {}
        for k in KS:
            fds, frs = [], []
            for _ in range(20):
                sub = {}
                for c, ms in cells.items():
                    if len(ms) < k:
                        continue
                    pick = rng.choice(len(ms), size=k, replace=False)
                    b = max((ms[i] for i in pick), key=lambda t: t[1])[0]
                    v = val_of(b, axis)
                    if v is not None:
                        sub[c] = v
                dec = decompose(sub, ARMS, DS) if len(sub) >= 6 else None
                if dec:
                    fds.append(dec[0])
                    frs.append(dec[1])
            if fds:
                ok = True
                sweep[axis][k] = {
                    "by_D": float(np.mean(fds)),
                    "by_reservoir": float(np.mean(frs)),
                    "n_cells": len([c for c, m in cells.items() if len(m) >= k]),
                }
                row += f"{np.mean(fds):>7.2f} /{np.mean(frs):>7.2f}"
            else:
                row += f"{'-':>16s}"
        if ok:
            print(row)
    res["budget_sweep"] = sweep

    print("\n3. ENSEMBLE-WEIGHTED HP (N=5, ElasticNet weights)")
    print(f"   {'axis':16s}{'var by D':>10s}{'var by res':>12s}   verdict")
    ens = {c: ensemble_weighted_hp(v) for c, v in cells.items()}
    ens = {c: e for c, e in ens.items() if e}
    print(f"   (cells with a usable ensemble: {len(ens)})")
    ens_res = {}
    for k in NUMERIC:
        vals = {c: e[k] for c, e in ens.items() if k in e}
        dec = decompose(vals, ARMS, DS)
        if dec:
            fd, fr = dec
            ens_res[k] = {"by_D": fd, "by_reservoir": fr}
            verdict = "D" if fd > 2 * fr else ("RESERVOIR" if fr > 2 * fd else "both/neither")
            print(f"   {k:16s}{fd:>10.2f}{fr:>12.2f}   {verdict}")
    res["ensemble_weighted"] = ens_res

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
