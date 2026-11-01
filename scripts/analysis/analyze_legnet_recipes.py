#!/usr/bin/env python
"""One-knob neighbours of the published LegNet recipe (e100) at 10k, 30k and 100k (lr38_ cells; five seeds per arm) against the 20-seed e100 control.

Per arm and size: mean single-model validation r (V0c), genomic-test, OOD and random-sequence r (oracle labels), the difference to the control mean with its standard error (arm SD / sqrt(n) and
control SD / sqrt(n) combined; seeds are independent draws, so this is an unpaired comparison), and the stack of the arm's five seeds. A knob is only worth adopting if its genomic difference is several
standard errors, because the seed-to-seed SD of one LegNet is 0.002 to 0.01 depending on size.

    python scripts/analysis/analyze_legnet_recipes.py [--out outputs/justification_oct22]      (cpuq)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "analysis"))
from analyze_extras38 import SETS, load, pearson  # noqa: E402

from albench.ensemble.stack import fit_zscored_stack  # noqa: E402

ARMS = ("e100_lr5", "e100_lr20", "e150", "e100_b2048", "e100_wd2", "e100_wd05")


def singles(c):
    keys = sorted(c["ms"])
    y = {s: c["lab"][f"oracle_{s}"].astype(np.float64) for s in SETS}
    r = {s: np.array([pearson(c["ms"][k]["t"][s], y[s]) for k in keys]) for s in SETS}
    r["val"] = np.array([c["ms"][k]["valr"] for k in keys])
    return keys, r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    a = ap.parse_args()
    res: dict = {}
    for D in (10, 30, 100):
        ctl = load(f"bench38_ln_e100_{D}k_c20")
        if not ctl or not ctl["ms"]:
            continue
        _, rc = singles(ctl)
        res[f"{D}k"] = {
            "e100 (control)": {"n": len(rc["val"]), **{m: float(rc[m].mean()) for m in rc}}
        }
        print(
            f"\nD={D}k  control e100 n={len(rc['val'])}: "
            + " ".join(f"{m} {rc[m].mean():.4f}" for m in ("val", *SETS)),
            flush=True,
        )
        for arm in ARMS:
            c = load(f"bench38_ln_{arm}_{D}k_c20")
            if not c or len(c["ms"]) < 2:
                continue
            keys, r = singles(c)
            row = {"n": len(keys)}
            for m in ("val", *SETS):
                diff = float(r[m].mean() - rc[m].mean())
                se = float(np.sqrt(r[m].var(ddof=1) / len(r[m]) + rc[m].var(ddof=1) / len(rc[m])))
                row[m] = {"mean": float(r[m].mean()), "diff": diff, "se": se}
            if len(keys) >= 3:
                st = fit_zscored_stack(np.stack([c["ms"][k]["v"] for k in keys]), c["yv"])
                row["stack"] = {
                    s: pearson(
                        st.predict(np.stack([c["ms"][k]["t"][s] for k in keys])),
                        c["lab"][f"oracle_{s}"].astype(np.float64),
                    )
                    for s in SETS
                }
            res[f"{D}k"][arm] = row
            print(
                f"  {arm:11s} n={len(keys)} "
                + " ".join(
                    f"{m} {row[m]['mean']:.4f} ({row[m]['diff']:+.4f}+-{row[m]['se']:.4f})"
                    for m in ("val", *SETS)
                )
                + (f" | stack genomic {row['stack']['genomic']:.4f}" if "stack" in row else ""),
                flush=True,
            )
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/data_legnet_recipes.json", "w"), indent=1)


if __name__ == "__main__":
    main()
