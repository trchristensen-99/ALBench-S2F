#!/usr/bin/env python
"""Write albench/ensemble/menus/per_size.json: the training-priority order of the menu and how many configurations to train at each dataset size.

K at a size is the smallest number of configurations (taken in priority order) for which the final 8-member ensemble (the 8 best of the K trained, by
validation r) is within a tolerance of the 8-member ensemble of all of them, on the mean curve over the reservoirs and pools tested: 0.003 on the genomic
test set and 0.005 on the designed and random-sequence sets (the noise floor of an ensemble comparison is about 0.005). Reads data.json (analyze_oct27.py
--only kcurve).

python scripts/analysis/make_per_size.py --data outputs/justification_oct22/data.json --out albench/ensemble/menus/per_size.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from albench.ensemble.menu import MAX_MINUTES_H100, load_menu  # noqa: E402

SIZES = (10_000, 30_000, 100_000)
TOL = {"genomic": 0.003, "ood": 0.005, "random_32k": 0.005}


def k_needed(curve: np.ndarray, tol: float) -> int:
    return int(next((k + 1 for k in range(len(curve)) if curve[k] >= -tol), len(curve)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="albench/ensemble/menus/per_size.json")
    ap.add_argument(
        "--patience-small",
        type=int,
        default=8,
        help="patience for 10k and 30k (16 if the replicate study confirms it)",
    )
    args = ap.parse_args()
    K = json.load(open(args.data))["kcurve"]
    menu = load_menu("v0")
    sizes: dict[str, dict] = {}
    for D in SIZES:
        cells = {k: v for k, v in K.items() if k.startswith("menu33|") and k.endswith(f"|{D}")}
        ev: dict[str, dict] = {}
        for e, tol in TOL.items():
            per = []
            for v in cells.values():
                c = np.mean([p["first2"]["top8"][e] for p in v["pools"]], 0)
                ref = np.mean([p["val"]["top8"][e][-1] for p in v["pools"]], 0)
                per.append(c - ref)
            n = min(len(x) for x in per)
            ev[e] = {"k": k_needed(np.mean([x[:n] for x in per], 0), tol), "tolerance": tol}
        k = max(x["k"] for x in ev.values())
        s3 = K.get(f"seeds3|genomic|{D}", {}).get("seeds")
        k3 = None
        if s3:
            ref = s3["full_top8"]["genomic"]
            k3 = int(
                next(
                    (
                        i + 1
                        for i, r in enumerate(s3["top8"]["genomic"])
                        if r >= ref - TOL["genomic"]
                    ),
                    len(s3["top8"]["genomic"]),
                )
            )
        sizes[str(D)] = {
            "k": k,
            "k_by_evaluation_set": ev,
            "k_with_3_seeds_genomic_test": k3,
            "seeds": 1,
            "patience": args.patience_small if D <= 30_000 else 8,
            "max_minutes_h100": MAX_MINUTES_H100[D],
            "cells": sorted(cells),
            "pools": {c: v["n_pools"] for c, v in cells.items()},
        }
    out = {
        "menu": "v0",
        "rule": "first2",
        "criterion": "smallest K whose 8-member ensemble (8 best of the K by validation r) is within 0.003 (genomic test) / 0.005 (designed set, random sequences) of the 8-member ensemble of all 33, mean over genomic, random and motif pools",
        "order": [e.id for e in menu.ordered("first2")],
        "sizes": sizes,
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}: " + ", ".join(f"{D // 1000}k K={sizes[str(D)]['k']}" for D in SIZES))


if __name__ == "__main__":
    main()
