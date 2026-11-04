#!/usr/bin/env python
"""Scaling curves at 10k / 30k / 100k: the tuned LegNet (single seeds and the 20-seed stack) against the fixed-harness loop (`rr40`: its validation-selected best model and its 30-model stack).

Single models are never chosen on test sets: the loop's "best single" is the model with the highest V0c r in its run, and the bank's is its best seed by V0c (its mean seed is also given). Loop numbers are the
mean over the replicate pools (3 at 10k and 30k, 2 at 100k); the bank is given on the canonical pool (20 seeds) and, where trained, on the same replicate pools as the loop (`register_nov4c.py`). Also fits,
per line and metric, the slope of r per decade of D (log-linear) and the exponent b of (1 - r) ~ D^-b.

    python scripts/analysis/scaling_curves_legnet_vs_loop.py      (cpuq; writes outputs/justification_oct22/data_scaling_curves.json)
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "analysis"))
from compare_legnet_vs_loop import AR, MR, SETS, TUNED, load, pearson, stack_scores  # noqa: E402

SIZES = {"10k": 10000, "30k": 30000, "100k": 100000}
METRICS = ("genomic", "ood", "random_32k", "mean18")


def per_set(t: dict, y: dict) -> dict:
    out = {s: pearson(t[s], y[s]) for s in SETS}
    out["mean18"] = float(np.mean([out[s] for s in SETS]))
    return out


def summarise(c: dict) -> dict:
    """Stack, mean single, and the single model with the best validation r, on every metric."""
    st = stack_scores(c["ms"], c["yv"], c["y"])
    st["mean18"] = float(np.mean([st[s] for s in SETS]))
    singles = [per_set(t, c["y"]) for _, t, _ in c["ms"]]
    val = [pearson(v, c["yv"]) for v, _, _ in c["ms"]]
    best = singles[int(np.argmax(val))]
    return {
        "stack": {m: st[m] for m in METRICS},
        "mean_single": {m: float(np.mean([s[m] for s in singles])) for m in METRICS},
        "best_single": {m: best[m] for m in METRICS},
        "n": len(c["ms"]),
    }


def mean_of(items: list, key: str) -> dict:
    return {m: float(np.mean([i[key][m] for i in items])) for m in METRICS}


def slopes(points: dict) -> dict:
    """points: size -> value. Slope of r per decade of D, and b in (1 - r) ~ D^-b."""
    xs = np.log10([SIZES[s] for s in points])
    ys = np.array(list(points.values()))
    return {
        "per_decade": float(np.polyfit(xs, ys, 1)[0]),
        "b": float(-np.polyfit(xs, np.log10(1 - ys), 1)[0]),
    }


def main() -> None:
    res: dict = {"lines": {}, "per_pool": {}, "n": {}}
    lines: dict = {
        k: {}
        for k in (
            "bank_single_mean",
            "bank_best_single",
            "bank_stack",
            "loop_best_single",
            "loop_stack",
            "loop_mean_single",
            "same_pool_bank_stack",
            "same_pool_bank_single_mean",
        )
    }
    for size in SIZES:
        bank = load(f"{MR}/bench38_ln_{TUNED[size]}_{size}_c20", "r*_menu_*.npz")
        b = summarise(bank)
        lines["bank_single_mean"][size] = b["mean_single"]
        lines["bank_best_single"][size] = b["best_single"]
        lines["bank_stack"][size] = b["stack"]
        runs = []
        for d in sorted(glob.glob(f"{AR}/genomic_{size}_rr40/rep*")):
            c = load(d, "r*_arch_*.npz")
            if c and len(c["ms"]) >= 5:
                runs.append(summarise(c))
        res["per_pool"][size] = {"loop": runs}
        lines["loop_best_single"][size] = mean_of(runs, "best_single")
        lines["loop_stack"][size] = mean_of(runs, "stack")
        lines["loop_mean_single"][size] = mean_of(runs, "mean_single")
        sp = []
        for r in (1, 2, 3):
            c = load(f"{MR}/bench38_ln_{TUNED[size]}_{size}_c20_genomic_r{r}", "r*_menu_*.npz")
            if c and len(c["ms"]) >= 10:  # a partial bank of at least 10 seeds
                sp.append(summarise(c))
        res["per_pool"][size]["same_pool_bank"] = sp
        if sp:
            lines["same_pool_bank_stack"][size] = mean_of(sp, "stack")
            lines["same_pool_bank_single_mean"][size] = mean_of(sp, "mean_single")
        res["n"][size] = {
            "bank": b["n"],
            "loop_runs": len(runs),
            "loop_models": [r["n"] for r in runs],
            "same_pool_banks": [s["n"] for s in sp],
        }
    res["lines"] = lines
    res["fits"] = {
        name: {m: slopes({s: v[m] for s, v in pts.items()}) for m in METRICS}
        for name, pts in lines.items()
        if len(pts) == 3
    }
    print("n:", json.dumps(res["n"]))
    for m in METRICS:
        print(f"\n{m}: " + "  ".join(f"{s:>6s}" for s in SIZES) + "   r/decade      b")
        for name, pts in lines.items():
            vals = "  ".join(f"{pts[s][m]:.4f}" if s in pts else "   -  " for s in SIZES)
            f = res["fits"].get(name, {}).get(m)
            print(
                f"  {name:27s} {vals}" + (f"   {f['per_decade']:+.4f}  {f['b']:.3f}" if f else "")
            )
    os.makedirs("outputs/justification_oct22", exist_ok=True)
    json.dump(
        res,
        open("outputs/justification_oct22/data_scaling_curves.json", "w"),
        indent=1,
        default=float,
    )


if __name__ == "__main__":
    main()
