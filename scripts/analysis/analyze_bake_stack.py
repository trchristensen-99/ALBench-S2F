#!/usr/bin/env python
"""Do independent (non-LegNet-trunk) designs add to the ensemble once they are trained with a long recipe?

Takes the 84 models of the recipe-matched bake-off (`register_nov3c.py`: 8 independent, 4 hybrid, 2 plain LegNet designs x 3 recipes x 2 seeds) and the 20-seed tuned LegNet bank at 30k, fits the study's
positive-ElasticNet stack on V0c for several subsets and scores it on the 18 test sets. An equal-count control (the same number of LegNet-derived models) separates "more good models" from "different models".

    python scripts/analysis/analyze_bake_stack.py [--out outputs/justification_nov3/data_bake_stack.json]     (cpuq)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "analysis"))
from compare_legnet_vs_loop import MR, SETS, load, stack_scores  # noqa: E402

RECIPES = ("own", "tuned", "mid")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_nov3/data_bake_stack.json")
    a = ap.parse_args()
    bank = load(f"{MR}/bench38_ln_e500_wd2_30k_c20", "r*_menu_*.npz")
    kinds = {e["id"]: e["kind"] for e in json.load(open("outputs/menus_nov3/BAKE_own_30k.json"))}
    pool = []  # (val, test, id, kind, recipe)
    for rec in RECIPES:
        d = f"{MR}/bake_{rec}_30k"
        c = load(d, "r*_menu_*.npz")
        assert len(c["yv"]) == len(bank["yv"]) and np.allclose(c["yv"], bank["yv"]), (
            "validation sets differ"
        )
        for v, t, mid in c["ms"]:
            meta = json.load(open(f"{d}/{mid}_meta.json"))
            pool.append((v, t, f"{rec}:{mid}", kinds[meta["menu_entry"]], rec))
    print("bake models loaded:", len(pool), "of 84; bank", len(bank["ms"]))
    yv, y = bank["yv"], bank["y"]
    B = [(v, t, i) for v, t, i in bank["ms"]]

    def sel(kind=None, recipes=RECIPES):
        return [(v, t, i) for v, t, i, k, r in pool if (kind is None or k in kind) and r in recipes]

    ind, leg = ("independent",), ("hybrid", "plain LegNet")
    groups = {
        "bank (20 seeds)": B,
        "independent, tuned+mid only": sel(ind, ("tuned", "mid")),
        "independent, all recipes": sel(ind),
        "LegNet-derived (hybrid+plain), tuned+mid": sel(leg, ("tuned", "mid")),
        "LegNet-derived, all recipes": sel(leg),
        "bank + independent (tuned+mid)": B + sel(ind, ("tuned", "mid")),
        "bank + independent (own only)": B + sel(ind, ("own",)),
        "bank + LegNet-derived (tuned+mid)": B + sel(leg, ("tuned", "mid")),
        "bank + all 84 bake models": B + sel(),
        "bank + independent(t+m) + LegNet-derived(t+m)": B
        + sel(ind, ("tuned", "mid"))
        + sel(leg, ("tuned", "mid")),
    }
    res: dict = {}
    print(
        f"{'subset':50s}{'n':>4s}{'genomic':>9s}{'OOD':>8s}{'random':>8s}{'mean18':>8s}{'eff.m':>7s}"
    )
    for name, ms in groups.items():
        s = stack_scores(ms, yv, y)
        res[name] = {
            k: s[k] for k in ("genomic", "ood", "random_32k", "effective_members", "n_models")
        } | {"mean18": float(np.mean([s[x] for x in SETS]))}
        print(
            f"{name:50s}{s['n_models']:4d}{s['genomic']:9.4f}{s['ood']:8.4f}{s['random_32k']:8.4f}{res[name]['mean18']:8.4f}{s['effective_members']:7.1f}"
        )
    # weight carried by the independent models inside the full union
    from albench.ensemble.stack import fit_zscored_stack

    ms = B + sel()
    st = fit_zscored_stack(np.stack([m[0] for m in ms]), yv)
    w = np.asarray(st.coef, float)
    w = w / max(w.sum(), 1e-12)
    kind_of = {i: k for _, _, i, k, _ in pool}
    share = {}
    for (_, _, i), wi in zip(ms, w):
        share[kind_of.get(i, "bank")] = share.get(kind_of.get(i, "bank"), 0.0) + float(wi)
    res["weight_share_in_bank_plus_all_bake"] = share
    print(
        "stack weight share in bank + all bake models:", {k: round(v, 3) for k, v in share.items()}
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
