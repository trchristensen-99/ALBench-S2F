#!/usr/bin/env python
"""The student families side by side on all 18 evaluation sets, at 10k, 30k and 100k (genomic pools, oracle labels, V0c selection):

  LegNet published   the out-of-the-box MPRA-LegNet recipe (e100), 20 seeds                                (bench38_ln_e100_<D>k_c20)
  LegNet tuned       the per-size recipe of docs/TUNED_LEGNET.md, 20 seeds                                  (bench38_ln_{e800_wd4,e500_wd2,e500}_<D>k_c20)
  menu v2            the 20 LLM-found configurations of the frozen menu, one seed each                      (v2m38_genomic_<D>k)
  AutoResearch run   one run of the loop (architecture + recipe search, 20 models), averaged over its runs  (archsearch_reps_oct22/st38_a0_genomic_<D>k/rep*)
  AlphaGenome S1/S2  AlphaGenome-FT students, frozen encoder (S1) and fully fine-tuned (S2), 10 seeds, 10k and 30k only (baselines only, AGENTS.md)

Per family and size: the mean single-model r and the positive-ElasticNet stack r (fitted on V0c) on every test set, and the SNV effect r (alt minus ref predictions against the oracle's alt minus ref). Runs of the loop are
stacked separately and then averaged. Test labels are never used for a choice.

    python scripts/analysis/analyze_baselines_allsets.py [--out outputs/justification_oct22]      (cpuq)
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
AR = "outputs/archsearch_reps_oct22"
TUNED = {10: "e800_wd4", 30: "e500_wd2", 100: "e500"}
SETS = (
    "genomic", "ood", "random_32k", "snv_ref", "snv_alt", "gc_sweep", "dinuc_shuffle",
    "sub_low", "sub_med", "sub_high", "ins_low", "ins_med", "ins_high",
    "del_low", "del_med", "del_high", "translocation", "inversion",
)  # fmt: skip


def pearson(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def load_dir(d: str, pattern: str, sets=SETS):
    """(val_labels, oracle labels, [(val_pred, {set: test_pred})]) of the models of one directory that trained to finite predictions."""
    if not os.path.exists(f"{d}/labels.npz"):
        return None
    lab = np.load(f"{d}/labels.npz")
    ms = []
    for mf in sorted(glob.glob(f"{d}/{pattern}")):
        try:
            z = np.load(mf.replace("_meta.json", ".npz"))
            v = z["val_pred"].astype(np.float64)
            t = {s: z[f"test_pred_{s}"].astype(np.float64) for s in sets}
        except Exception:
            continue
        if not np.isfinite(v).all() or any(
            not np.isfinite(x).all() or np.abs(x).max() > 1e3 for x in t.values()
        ):
            continue
        ms.append((v, t))
    return {"yv": lab["val_labels"].astype(np.float64), "lab": lab, "ms": ms}


def score(g, sets=SETS) -> dict:
    ms, lab, yv = g["ms"], g["lab"], g["yv"]
    out = {"n": len(ms)}
    y = {s: lab[f"oracle_{s}"].astype(np.float64) for s in sets}
    snv = "snv_ref" in sets and "snv_alt" in sets
    dy = y["snv_alt"] - y["snv_ref"] if snv else None
    for s in sets:
        out[f"single_{s}"] = float(np.mean([pearson(m[1][s], y[s]) for m in ms]))
    if snv:
        out["single_snv_delta"] = float(
            np.mean([pearson(m[1]["snv_alt"] - m[1]["snv_ref"], dy) for m in ms])
        )
    if len(ms) >= 3:
        st = fit_zscored_stack(np.stack([m[0] for m in ms]), yv)
        P = {s: st.predict(np.stack([m[1][s] for m in ms])) for s in sets}
        for s in sets:
            out[f"stack_{s}"] = pearson(P[s], y[s])
        if snv:
            out["stack_snv_delta"] = pearson(P["snv_alt"] - P["snv_ref"], dy)
    return out


def average(rows: list) -> dict:
    keys = [k for k in rows[0] if k != "n"]
    out = {k: float(np.nanmean([r[k] for r in rows if k in r])) for k in keys}
    out["n"] = rows[0]["n"]
    out["runs"] = len(rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    a = ap.parse_args()
    res: dict = {}
    for k in (10, 30, 100):
        fams = {
            "LegNet published": [(f"{MR}/bench38_ln_e100_{k}k_c20", "r*_menu_*_meta.json")],
            "LegNet tuned": [(f"{MR}/bench38_ln_{TUNED[k]}_{k}k_c20", "r*_menu_*_meta.json")],
            "menu v2": [(f"{MR}/v2m38_genomic_{k}k", "r*_menu_*_meta.json")],
            "AutoResearch run": [
                (d, "r*_arch_*_meta.json")
                for d in sorted(glob.glob(f"{AR}/st38_a0_genomic_{k}k/rep*"))
            ],
        }
        if k <= 30:
            fams["AlphaGenome S1 (frozen encoder)"] = [
                (
                    f"{MR}/bench33_agS1_oracle_{k}k",
                    "r*_menu_*_meta.json",
                    ("genomic", "ood", "random_32k"),
                )
            ]
            fams["AlphaGenome S2 (fine-tuned)"] = [
                (f"{MR}/bench33_agS2_oracle_{k}k", "r*_menu_*_meta.json")
            ]
        for name, dirs in fams.items():
            rows = []
            for spec in dirs:
                d, pat = spec[0], spec[1]
                sets = spec[2] if len(spec) > 2 else SETS
                g = load_dir(d, pat, sets)
                if g and len(g["ms"]) >= 3:
                    rows.append(score(g, sets))
            if not rows:
                continue
            res.setdefault(f"{k}k", {})[name] = average(rows)
            r = res[f"{k}k"][name]
            print(
                f"{k:3d}k {name:32s} n={r['n']:2d} runs={r['runs']} | stack: "
                + " ".join(
                    f"{m} {r[f'stack_{m}']:.4f}"
                    for m in (
                        "genomic",
                        "ood",
                        "random_32k",
                        "snv_delta",
                        "gc_sweep",
                        "dinuc_shuffle",
                    )
                    if f"stack_{m}" in r
                ),
                flush=True,
            )
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/data_baselines_allsets.json", "w"), indent=1)


if __name__ == "__main__":
    main()
