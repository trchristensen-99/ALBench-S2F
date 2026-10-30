#!/usr/bin/env python
"""How many independent seeds of the published MPRA-LegNet (with the context input) does an ensemble need, and does variety across recipes matter more than more seeds of one recipe?

For a training size (10k or 30k, canonical genomic pool): the stack r (positive ElasticNet on V0c, test sets never used for a choice) of random subsets of n members, averaged over draws, for
  - one recipe at one context (e100, context 28): n of its 20 seeds
  - the pooled LegNet cells (e100, b256, pub at context 20 and 28; 120 models): n random members, so recipes and contexts are mixed
  - the menu (20 LLM-found configurations, one seed each): n random configurations, for comparison
and the single-model mean and SD. Draws are seeded; n = 1 is the mean single-model r.

    python scripts/analysis/analyze_seed_scaling.py --size 30k [--out outputs/justification_oct22]      (cpuq)
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

NS = (1, 2, 3, 5, 8, 12, 16, 20)
DRAWS = 12


def cells(size: str):
    k = "10" if size == "10k" else "30"
    ln = {
        f"{r}_c{c}": f"bench38_ln_{r}_{size}_c{c}"
        for r in ("e100", "b256", "pub")
        for c in (20, 28)
    }
    menu = "hp38_base_genomic_10k" if size == "10k" else "mv1c_c20_30k"
    return k, ln, menu


def arrays(c, yv_ref):
    ms = c["ms"]
    keys = sorted(ms)
    V = np.stack([ms[x]["v"] for x in keys])
    T = {s: np.stack([ms[x]["t"][s] for x in keys]) for s in SETS}
    return V, T


def curve(V, T, y, yv, rng):
    out = {}
    for n in NS:
        if n > len(V):
            continue
        vals = {s: [] for s in SETS}
        for _ in range(1 if n == len(V) else DRAWS):
            idx = rng.choice(len(V), n, replace=False)
            if n == 1:
                for s in SETS:
                    vals[s].append(pearson(T[s][idx[0]], y[s]))
                continue
            st = fit_zscored_stack(V[idx], yv)
            for s in SETS:
                vals[s].append(pearson(st.predict(T[s][idx]), y[s]))
        out[n] = {s: {"mean": float(np.mean(v)), "sd": float(np.std(v))} for s, v in vals.items()}
        print(
            f"  n={n:3d} "
            + " ".join(f"{s} {out[n][s]['mean']:.4f}+-{out[n][s]['sd']:.4f}" for s in SETS),
            flush=True,
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="30k", choices=["10k", "30k"])
    ap.add_argument("--out", default="outputs/justification_oct22")
    a = ap.parse_args()
    _, ln, menu = cells(a.size)
    rng = np.random.default_rng(0)
    loaded = {n: load(d) for n, d in ln.items()}
    loaded = {n: c for n, c in loaded.items() if c and c["ms"]}
    ref = next(iter(loaded.values()))
    yv = ref["yv"]
    y = {s: ref["lab"][f"oracle_{s}"].astype(np.float64) for s in SETS}
    res = {}
    for name, label in (
        ("e100_c28", "one recipe: LegNet e100, context 28"),
        ("b256_c28", "one recipe: LegNet b256, context 28"),
    ):
        if name in loaded:
            print(label, flush=True)
            V, T = arrays(loaded[name], yv)
            res[label] = curve(V, T, y, yv, rng)
    parts = [arrays(c, yv) for c in loaded.values()]
    V = np.concatenate([p[0] for p in parts])
    T = {s: np.concatenate([p[1][s] for p in parts]) for s in SETS}
    label = f"mixed: members drawn from all {len(V)} LegNet models (3 recipes x 2 contexts)"
    print(label, flush=True)
    res[label] = curve(V, T, y, yv, rng)
    m = load(menu)
    if m and m["ms"]:
        label = f"menu: {len(m['ms'])} LLM-found configurations, one seed each"
        print(label, flush=True)
        Vm, Tm = arrays(m, yv)
        res[label] = curve(Vm, Tm, y, yv, rng)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/data_seed_scaling_{a.size}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
