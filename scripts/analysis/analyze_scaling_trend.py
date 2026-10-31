#!/usr/bin/env python
"""How does the tuned MPRA-LegNet (and the menus) scale with training size, and what does that say about a "known achievable" target at sizes we have not trained?

For D = 10k, 30k, 100k (canonical genomic nested pools, study input standard, V0c validation) the score of
  - the single tuned LegNet (mean over its seeds, recipes e100 and b256 separately and pooled),
  - the stack of all LegNet seeds (positive ElasticNet on V0c, test sets never used for a choice),
  - the stack of menu v1c and of menu v2 (one seed per configuration)
on the genomic test set, OOD and random, as Pearson r and as error 1 - r. For each series a power law in the error, (1 - r) = a D^-b, is fitted to the three sizes and extrapolated to 300k and 1M;
the spread of the three leave-one-out fits (two sizes each) is reported as the honest uncertainty, because three points cannot identify a floor.

    python scripts/analysis/analyze_scaling_trend.py [--out outputs/justification_oct22]      (cpuq)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "analysis"))
from analyze_extras38 import SETS, load, pearson  # noqa: E402

from albench.ensemble.stack import fit_zscored_stack  # noqa: E402

SIZES = (10_000, 30_000, 100_000)
EXTRAP = (300_000, 1_000_000)
CELLS = {
    "legnet_e100": lambda k: [f"bench38_ln_e100_{k}k_c20"],
    "legnet_b256": lambda k: [f"bench38_ln_b256_{k}k_c20"],
    "legnet_all": lambda k: [f"bench38_ln_e100_{k}k_c20", f"bench38_ln_b256_{k}k_c20"],
    "menu_v1c": lambda k: [
        {10: "hp38_base_genomic_10k", 30: "mv1c_c20_30k", 100: "hp38_base_genomic_100k"}[k]
    ],
    "menu_v2": lambda k: [f"v2m38_genomic_{k}k"],
}


def arrays(cells):
    V, T = [], {s: [] for s in SETS}
    for c in cells:
        keys = sorted(c["ms"])
        V += [c["ms"][x]["v"] for x in keys]
        for s in SETS:
            T[s] += [c["ms"][x]["t"][s] for x in keys]
    return np.stack(V), {s: np.stack(T[s]) for s in SETS}


def power_fit(d, e):
    """log e = log a - b log d by least squares; returns (a, b)."""
    b, la = np.polyfit(np.log(d), np.log(e), 1)
    return float(np.exp(la)), float(-b)


def predict(a, b, d):
    return a * d ** (-b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    args = ap.parse_args()
    res: dict = {}
    for name, fn in CELLS.items():
        row = {"D": [], "single_mean": {}, "single_sd": {}, "stack": {}, "n": []}
        for D in SIZES:
            k = D // 1000
            loaded = [load(d) for d in fn(k)]
            loaded = [c for c in loaded if c and c["ms"]]
            if not loaded:
                continue
            ref = loaded[0]
            y = {s: ref["lab"][f"oracle_{s}"].astype(np.float64) for s in SETS}
            V, T = arrays(loaded)
            st = fit_zscored_stack(V, ref["yv"])
            row["D"].append(D)
            row["n"].append(int(len(V)))
            for s in SETS:
                r1 = [pearson(T[s][i], y[s]) for i in range(len(V))]
                row["single_mean"].setdefault(s, []).append(float(np.mean(r1)))
                row["single_sd"].setdefault(s, []).append(float(np.std(r1)))
                row["stack"].setdefault(s, []).append(float(pearson(st.predict(T[s]), y[s])))
            print(
                f"{name:12s} D={D:7d} n={len(V):3d} "
                + " ".join(
                    f"{s} single {row['single_mean'][s][-1]:.4f} stack {row['stack'][s][-1]:.4f}"
                    for s in SETS
                ),
                flush=True,
            )
        row["fits"] = {}
        for kind in ("single_mean", "stack"):
            for s in SETS:
                if len(row["D"]) < 3:
                    continue
                d = np.array(row["D"], float)
                e = 1 - np.array(row[kind][s])
                a, b = power_fit(d, e)
                loo = []
                for i, j in itertools.combinations(range(3), 2):
                    aa, bb = power_fit(d[[i, j]], e[[i, j]])
                    loo.append({"pair": [int(d[i]), int(d[j])], "b": bb})
                    for D in EXTRAP:
                        loo[-1][str(D)] = 1 - predict(aa, bb, D)
                row["fits"][f"{kind}/{s}"] = {
                    "a": a,
                    "b": b,
                    "resid_at_sizes": [float(1 - predict(a, b, x) - r) for x, r in zip(d, 1 - e)],
                    **{str(D): float(1 - predict(a, b, D)) for D in EXTRAP},
                    "leave_one_out": loo,
                }
                f = row["fits"][f"{kind}/{s}"]
                lo = [x[str(300_000)] for x in loo]
                hi = [x[str(1_000_000)] for x in loo]
                print(
                    f"   fit {kind}/{s}: err = {a:.3f} D^-{b:.3f}  r(300k) {f['300000']:.4f} (loo {min(lo):.4f}..{max(lo):.4f})  r(1M) {f['1000000']:.4f} (loo {min(hi):.4f}..{max(hi):.4f})",
                    flush=True,
                )
        res[name] = row
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(f"{args.out}/data_scaling_trend.json", "w"), indent=1)


if __name__ == "__main__":
    main()
