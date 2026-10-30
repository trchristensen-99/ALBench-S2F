#!/usr/bin/env python
"""Summary of every menu-retrain cell of the Oct 29-30 follow-ups in one table, plus paired differences against their controls.

For each cell (directory under outputs/menu_retrain_v0 with labels.npz and r*_menu_*_meta.json): number of models, mean single-model test r, best single model by validation, and the
positive-ElasticNet stack of all its models fitted on V0c (genomic test, designed set, random sequences; all scored against the oracle labels). Paired differences compare the SAME configuration
(seed round and menu index) in two cells, so differences between configurations cancel: mean over configurations, SE, number of configurations that improve.

Groups: oracle-like augmentation (fx36_), auxiliary target (mt38_), menu v1c with context (mv1c_), baselines with context (bench38_, lg38_), and the Oct-30 follow-ups (hp38_, rc38_, v2m38_, v1c38_).

    python scripts/analysis/analyze_extras38.py [--out outputs/justification_oct22]      (cpuq)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from albench.ensemble.stack import fit_zscored_stack  # noqa: E402

MR = "outputs/menu_retrain_v0"
SETS = ("genomic", "ood", "random_32k")
PATTERNS = [
    "bench33_agS2_oracle_*",
    "bench33_ln_*",
    "fx36_*",
    "mt38_*",
    "mv1c_*",
    "fx35_c20s10_30k",
    "hp34_base_30k",
    "bench38_*",
    "lg38_*",
    "lg35c_*",
    "hp38_*",
    "rc38_*",
    "v2m38_*",
    "v1c38_*",
]
PAIRS = [  # (cell, control)
    ("fx36_c20_ps10_30k", "fx35_c20s10_30k"),
    ("fx36_c20_oracle_30k", "fx35_c20s10_30k"),
    ("fx36_c20_tta8_30k", "fx35_c20s10_30k"),
    ("mv1c_c20_30k", "fx35_c20s10_30k"),
    ("mt38_aux05_r1_30k", "mt38_k562only_r1_30k"),
    ("mt38_aux10_r1_30k", "mt38_k562only_r1_30k"),
]
for arm in ("lrx05", "lrx2", "bs128", "bs512", "wd1e3", "wd1e2"):
    PAIRS += [
        (f"hp38_{arm}_genomic_10k", "hp38_base_genomic_10k"),
        (f"hp38_{arm}_genomic_100k", "hp38_base_genomic_100k"),
    ]
PAIRS += [
    ("rc38_rca_genomic_10k", "hp38_base_genomic_10k"),
    ("rc38_rca_genomic_100k", "hp38_base_genomic_100k"),
    ("rc38_rca_genomic_30k", "mv1c_c20_30k"),
    ("rc38_rca_random_30k", "rc38_c20_random_30k"),
    ("rc38_rca_motif_ct_enriched_30k", "rc38_c20_motif_ct_enriched_30k"),
]


def pearson(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def load(d: str):
    p = f"{MR}/{d}"
    if not os.path.exists(f"{p}/labels.npz"):
        return None
    lab = np.load(f"{p}/labels.npz")
    yv = lab["val_labels"].astype(np.float64)
    ms = {}
    for mf in sorted(glob.glob(f"{p}/r*_menu_*_meta.json")):
        try:
            json.load(open(mf))  # skip models whose meta file is unreadable
            z = np.load(mf.replace("_meta.json", ".npz"))
            v = z["val_pred"].astype(np.float64)
            t = {s: z[f"test_pred_{s}"].astype(np.float64) for s in SETS}
        except Exception:
            continue
        if not np.isfinite(v).all() or any(
            not np.isfinite(x).all() or np.abs(x).max() > 1e3 for x in t.values()
        ):
            continue
        ms[os.path.basename(mf).replace("_meta.json", "")] = {
            "v": v,
            "t": t,
            "valr": pearson(v, yv),
        }
    return {"lab": lab, "yv": yv, "ms": ms}


def summarise(c) -> dict:
    ms, lab, yv = c["ms"], c["lab"], c["yv"]
    out = {"n_models": len(ms)}
    if len(ms) < 1:
        return out
    keys = sorted(ms)
    out["mean_val_r"] = float(np.mean([ms[k]["valr"] for k in keys]))
    out["best_val_r"] = float(max(ms[k]["valr"] for k in keys))
    st = None
    if len(ms) >= 3:
        st = fit_zscored_stack(np.stack([ms[k]["v"] for k in keys]), yv)
    b = max(keys, key=lambda k: ms[k]["valr"])
    for s in SETS:
        y = lab[f"oracle_{s}"].astype(np.float64)
        T = np.stack([ms[k]["t"][s] for k in keys])
        out[f"mean_single_{s}"] = float(np.mean([pearson(t, y) for t in T]))
        out[f"best_val_single_{s}"] = pearson(ms[b]["t"][s], y)
        if st is not None:
            out[f"stack_{s}"] = pearson(st.predict(T), y)
    if st is not None and "test_true" in lab.files:
        T = np.stack([ms[k]["t"]["genomic"] for k in keys])
        out["stack_genomic_measured"] = pearson(st.predict(T), lab["test_true"].astype(np.float64))
    return out


def paired(a, b) -> dict:
    common = sorted(set(a["ms"]) & set(b["ms"]))
    out = {"n": len(common)}
    if len(common) < 3:
        return out
    for s in SETS:
        y = a["lab"][f"oracle_{s}"].astype(np.float64)
        d = np.array(
            [pearson(a["ms"][k]["t"][s], y) - pearson(b["ms"][k]["t"][s], y) for k in common]
        )
        out[s] = {
            "mean": float(d.mean()),
            "se": float(d.std(ddof=1) / np.sqrt(len(d))),
            "n_better": int((d > 0).sum()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    a = ap.parse_args()
    names = sorted({os.path.basename(p) for pat in PATTERNS for p in glob.glob(f"{MR}/{pat}")})
    cells, summ = {}, {}
    for n in names:
        c = load(n)
        if c and c["ms"]:
            cells[n] = c
            summ[n] = summarise(c)
            sm = summ[n]
            print(
                f"{n:44s} n={sm['n_models']:3d} "
                + " ".join(
                    f"{k[6:]}={sm[k]:.4f}"
                    for k in ("stack_genomic", "stack_ood", "stack_random_32k")
                    if k in sm
                )
                + (
                    f" single={sm['mean_single_genomic']:.4f}"
                    if "mean_single_genomic" in sm
                    else ""
                ),
                flush=True,
            )
    pr = {}
    for x, y in PAIRS:
        if x in cells and y in cells:
            pr[f"{x} - {y}"] = paired(cells[x], cells[y])
            p = pr[f"{x} - {y}"]
            if "genomic" in p:
                print(
                    f"PAIRED {x} - {y}: n={p['n']} genomic {p['genomic']['mean']:+.4f} (SE {p['genomic']['se']:.4f}, {p['genomic']['n_better']}/{p['n']} better) ood {p['ood']['mean']:+.4f} random {p['random_32k']['mean']:+.4f}",
                    flush=True,
                )
    os.makedirs(a.out, exist_ok=True)
    json.dump(
        {"cells": summ, "paired": pr},
        open(f"{a.out}/data_extras38.json", "w"),
        indent=1,
        default=float,
    )
    print("wrote", f"{a.out}/data_extras38.json")


if __name__ == "__main__":
    main()
