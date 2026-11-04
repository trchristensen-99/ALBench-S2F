#!/usr/bin/env python
"""The loop against the tuned LegNet bank on the SAME training pool (the correction of Nov 4: loop runs train on `k562_reps` r1-r3, the original bank on the canonical pool).

Per size and replicate pool: the bank trained on that pool (`register_nov4c.py`, `bench38_ln_<cfg>_<D>k_c20_genomic_r<r>`), the loop runs on that pool (`rr40`, `lv40`), their union, and the original bank on the
canonical pool, on every test set. Partial banks are used as they are (the number of seeds is printed), so the read-out can be repeated while the seeds finish.

    python scripts/analysis/compare_same_pool.py [--sizes 10k,30k,100k] [--loops rr40,lv40]      (cpuq)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "analysis"))
from compare_legnet_vs_loop import AR, MR, SETS, TUNED, load, stack_scores  # noqa: E402

COLS = ("genomic", "ood", "random_32k")


def fmt(s: dict) -> str:
    return (
        " / ".join(f"{s[k]:.4f}" for k in COLS)
        + f" / {np.mean([s[k] for k in SETS]):.4f}  (n={s['n_models']}, eff {s['effective_members']:.1f})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="10k,30k,100k")
    ap.add_argument("--loops", default="rr40,lv40")
    a = ap.parse_args()
    res: dict = {}
    for size in a.sizes.split(","):
        can = load(f"{MR}/bench38_ln_{TUNED[size]}_{size}_c20", "r*_menu_*.npz")
        print(f"\n== {size}   genomic / OOD / random / mean of 18")
        if can and can["ms"]:
            print(
                f"bank, canonical pool          {fmt(stack_scores(can['ms'], can['yv'], can['y']))}"
            )
        res[size] = {}
        for r in (1, 2, 3):
            b = load(f"{MR}/bench38_ln_{TUNED[size]}_{size}_c20_genomic_r{r}", "r*_menu_*.npz")
            if not b or len(b["ms"]) < 3:
                print(f"r{r}: same-pool bank not ready ({len(b['ms']) if b else 0} seeds)")
                continue
            row = {"bank": stack_scores(b["ms"], b["yv"], b["y"])}
            print(f"r{r} bank, same pool           {fmt(row['bank'])}")
            for tag in a.loops.split(","):
                c = load(f"{AR}/genomic_{size}_{tag}/rep{r}", "r*_arch_*.npz")
                if not c or len(c["ms"]) < 5:
                    continue
                assert len(c["yv"]) == len(b["yv"])
                row[tag] = stack_scores(c["ms"], c["yv"], c["y"])
                row[f"{tag}+bank"] = stack_scores(c["ms"] + b["ms"], c["yv"], c["y"])
                print(f"r{r} {tag:5s} loop               {fmt(row[tag])}")
                print(f"r{r} {tag:5s} loop + same-pool bank {fmt(row[f'{tag}+bank'])}")
            res[size][f"r{r}"] = row
        for tag in a.loops.split(","):
            pairs = [(v[tag], v["bank"], v[f"{tag}+bank"]) for v in res[size].values() if tag in v]
            if pairs:
                d = {k: [p[0][k] - p[1][k] for p in pairs] for k in COLS}
                u = {k: [p[2][k] - p[1][k] for p in pairs] for k in COLS}
                print(
                    f"{tag} minus same-pool bank over {len(pairs)} pools: "
                    + ", ".join(f"{k} {np.mean(d[k]):+.4f}" for k in COLS)
                    + " | union minus bank: "
                    + ", ".join(f"{k} {np.mean(u[k]):+.4f}" for k in COLS)
                )
    os.makedirs("outputs/justification_oct22", exist_ok=True)
    json.dump(
        res, open("outputs/justification_oct22/data_same_pool.json", "w"), indent=1, default=float
    )


if __name__ == "__main__":
    main()
