#!/usr/bin/env python
"""Why is the loop's ensemble below the tuned LegNet bank, mostly on OOD? And what exactly did the mixed validation set (vc39) change, set by set?

Part 1 (30k, every model of the fixed-harness loop arms): single-model genomic, OOD and random r against the epochs each model trained, and against the bank's own epoch curve (the published LegNet at
100-800 epochs, `bench38_ln_*_30k_c20`): if the loop's OOD deficit is a schedule effect, loop models at a given genomic r should sit on the bank's OOD-against-genomic curve and their OOD should rise with
epochs; if it is the architectures or the harness, they sit below the curve at every schedule length. Also the stack of each arm on every set.

Part 2: `vc39` (mixed validation B1 for stopping, feedback and stack) against `rr39` (V0c), same pools: stack and single-model differences on all 18 sets.

    python scripts/analysis/analyze_loop_ood_gap.py      (cpuq)
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "analysis"))
from compare_legnet_vs_loop import (  # noqa: E402
    AR,
    MR,
    SETS,
    legnet_derived,
    load,
    pearson,
    stack_scores,
)

ARMS = ("rr40", "lv40", "ar41", "ln41", "is40", "qt40", "rl40", "af42", "ln42")
BANK_ARMS = ("e100", "e150_wd2", "e200_wd2", "e300_wd2", "e500_wd2", "e800_wd2")
KEY = ("genomic", "ood", "random_32k")


def singles(c: dict) -> list:
    return [{s: pearson(t[s], c["y"][s]) for s in SETS} for _, t, _ in c["ms"]]


def part1() -> dict:
    curve = []
    for arm in BANK_ARMS:
        c = load(f"{MR}/bench38_ln_{arm}_30k_c20", "r*_menu_*.npz")
        if c:
            r = singles(c)
            curve.append(
                (
                    float(np.mean([x["genomic"] for x in r])),
                    float(np.mean([x["ood"] for x in r])),
                    arm,
                )
            )
    curve.sort()
    gx, oy = np.array([g for g, _, _ in curve]), np.array([o for _, o, _ in curve])
    fit = np.polyfit(gx, oy, 1)  # OOD against genomic along the bank's epoch axis
    print("bank epoch curve (genomic, OOD):", [(round(g, 4), round(o, 3), a) for g, o, a in curve])
    print(f"bank line: OOD = {fit[0]:.2f} x genomic {fit[1]:+.3f}")
    rows, stacks = [], {}
    for arm in ARMS:
        for d in sorted(glob.glob(f"{AR}/genomic_30k_{arm}/rep*")):
            c = load(d, "r*_arch_*.npz")
            if not c or len(c["ms"]) < 5:
                continue
            st = stack_scores(c["ms"], c["yv"], c["y"])
            stacks.setdefault(arm, []).append(st)
            for (_, t, mid), sc in zip(c["ms"], singles(c)):
                try:
                    m = json.load(open(f"{d}/{mid}_meta.json"))
                except Exception:
                    continue
                rows.append(
                    {
                        "arm": arm,
                        "epochs": m.get("epochs_trained") or 0,
                        "legnet": legnet_derived(d, mid),
                        "opt": m.get("optimizer_used"),
                        **sc,
                        "ood_resid": sc["ood"] - float(np.polyval(fit, sc["genomic"])),
                    }
                )
    print(f"\n{len(rows)} loop models")
    for lo, hi in ((0, 100), (100, 200), (200, 300), (300, 10000)):
        sel = [r for r in rows if lo <= r["epochs"] < hi]
        if sel:
            print(
                f"epochs {lo}-{hi}: n={len(sel)} genomic {np.mean([r['genomic'] for r in sel]):.4f} OOD {np.mean([r['ood'] for r in sel]):.3f} "
                f"OOD below the bank line at equal genomic {np.mean([r['ood_resid'] for r in sel]):+.3f} (SE {np.std([r['ood_resid'] for r in sel]) / np.sqrt(len(sel)):.3f})"
            )
    good = [r for r in rows if r["genomic"] >= 0.86]
    for name, sel in (
        ("models >= 0.86 genomic", good),
        ("  of them LegNet-derived", [r for r in good if r["legnet"]]),
        ("  of them other", [r for r in good if not r["legnet"]]),
        ("  of them >= 300 epochs", [r for r in good if r["epochs"] >= 300]),
    ):
        if sel:
            print(
                f"{name}: n={len(sel)} genomic {np.mean([r['genomic'] for r in sel]):.4f} OOD {np.mean([r['ood'] for r in sel]):.3f} resid {np.mean([r['ood_resid'] for r in sel]):+.3f}"
            )
    corr = np.corrcoef([r["epochs"] for r in rows], [r["ood_resid"] for r in rows])[0, 1]
    print(f"correlation of epochs with the OOD residual over all loop models: {corr:+.2f}")
    print("\nstack per arm (mean over pools): genomic / OOD / random / mean of 18")
    for arm, sts in stacks.items():
        print(
            f"  {arm:5s} n={len(sts)} "
            + " / ".join(f"{np.mean([s[k] for s in sts]):.4f}" for k in KEY)
            + f" / {np.mean([np.mean([s[k] for k in SETS]) for s in sts]):.4f}"
        )
    return {"curve": curve, "fit": fit.tolist(), "rows": rows}


def part2() -> dict:
    out = {}
    pairs = []
    for r in (1, 2, 3):
        a = load(f"{AR}/genomic_30k_vc39/rep{r}", "r*_arch_*.npz")
        b = load(f"{AR}/genomic_30k_rr39/rep{r}", "r*_arch_*.npz")
        if a and b and len(a["ms"]) >= 5 and len(b["ms"]) >= 5:
            pairs.append(
                (
                    stack_scores(a["ms"], a["yv"], a["y"]),
                    stack_scores(b["ms"], b["yv"], b["y"]),
                    singles(a),
                    singles(b),
                )
            )
    print(
        f"\nvc39 (mixed validation B1) minus rr39 (V0c), {len(pairs)} paired pools: stack / mean single model"
    )
    for s in (*SETS, "snv_effect"):
        ds = [p[0][s] - p[1][s] for p in pairs]
        dm = (
            [np.mean([x[s] for x in p[2]]) - np.mean([x[s] for x in p[3]]) for p in pairs]
            if s != "snv_effect"
            else [np.nan]
        )
        print(
            f"  {s:14s} stack {np.mean(ds):+.4f} (pools {', '.join(f'{x:+.3f}' for x in ds)})   single {np.nanmean(dm):+.4f}"
        )
        out[s] = {"stack": ds, "single": dm}
    return out


def main() -> None:
    res = {"part1": part1(), "part2": part2()}
    os.makedirs("outputs/justification_oct22", exist_ok=True)
    json.dump(
        res,
        open("outputs/justification_oct22/data_loop_ood_gap.json", "w"),
        indent=1,
        default=float,
    )


if __name__ == "__main__":
    main()
