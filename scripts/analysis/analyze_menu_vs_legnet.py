# ruff: noqa
#!/usr/bin/env python
"""Is it a coincidence that the 20-architecture menu and 20 tuned MPRA-LegNets give the same ensemble r at 30k (0.782 against 0.782)?

For 30k and 10k (genomic pool): V0c-fitted stacks of the menu (sc31_{D}k_genomic) and of the tuned LegNets (bench33_ln_b256_{D}k); the stack of both families together (40 models); the correlation of the two ensembles'
test predictions and of their errors; and a paired bootstrap over test sequences of the difference in ensemble r (menu minus LegNet), so the difference can be read against its sampling noise.

    python scripts/analysis/analyze_menu_vs_legnet.py [--out outputs/justification_oct22]      (cpuq)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_oct32 as a32  # noqa: E402
from albench.ensemble.stack import fit_zscored_stack  # noqa: E402

SH = "/grid/koo/home/shared/albench/data"
SETS = ("genomic", "ood", "random_32k")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    a = ap.parse_args()
    vlab = np.load(f"{SH}/val_V0c.npz", allow_pickle=True)["oracle_labels"].astype(float)
    rng = np.random.default_rng(0)
    out = {}
    for D in (30, 10):
        fam = {}
        for name, d in (("menu", f"sc31_{D}k_genomic"), ("legnet", f"bench33_ln_b256_{D}k")):
            c = a32.load_all(d)
            V = np.stack([r["val"] for r in c["rows"]])
            T = {s: np.stack([r["test"][s] for r in c["rows"]]) for s in SETS}
            fam[name] = (V, T, c)
        y = {s: fam["menu"][2]["y"][s] for s in SETS}
        res = {"n": {k: v[0].shape[0] for k, v in fam.items()}}
        P = {}
        for name, (V, T, _) in fam.items():
            st = fit_zscored_stack(V, vlab)
            P[name] = {s: np.asarray(st.predict(T[s]), float) for s in SETS}
        Vb = np.concatenate([fam["menu"][0], fam["legnet"][0]])
        st = fit_zscored_stack(Vb, vlab)
        P["both"] = {
            s: np.asarray(
                st.predict(np.concatenate([fam["menu"][1][s], fam["legnet"][1][s]])), float
            )
            for s in SETS
        }
        P["mean_of_two_ensembles"] = {
            s: (P["menu"][s] - P["menu"][s].mean()) / P["menu"][s].std()
            + (P["legnet"][s] - P["legnet"][s].mean()) / P["legnet"][s].std()
            for s in SETS
        }
        for s in SETS:
            r = lambda p: float(np.corrcoef(p, y[s])[0, 1])
            e1, e2 = y[s] - P["menu"][s] * 0, None
            zs = lambda p: (p - p.mean()) / p.std()

            # errors of the two ensembles after rescaling each to the labels by regression
            def resid(p):
                b = np.polyfit(p, y[s], 1)
                return y[s] - np.polyval(b, p)

            n = len(y[s])
            diffs = []
            for _ in range(1000):
                i = rng.integers(0, n, n)
                diffs.append(
                    np.corrcoef(P["menu"][s][i], y[s][i])[0, 1]
                    - np.corrcoef(P["legnet"][s][i], y[s][i])[0, 1]
                )
            res[s] = {
                "r_menu": r(P["menu"][s]),
                "r_legnet": r(P["legnet"][s]),
                "r_both_stack": r(P["both"][s]),
                "r_mean_of_two": r(P["mean_of_two_ensembles"][s]),
                "corr_between_ensembles": float(np.corrcoef(P["menu"][s], P["legnet"][s])[0, 1]),
                "corr_of_errors": float(
                    np.corrcoef(resid(P["menu"][s]), resid(P["legnet"][s]))[0, 1]
                ),
                "diff_menu_minus_legnet": float(r(P["menu"][s]) - r(P["legnet"][s])),
                "diff_boot_ci95": [
                    float(np.percentile(diffs, 2.5)),
                    float(np.percentile(diffs, 97.5)),
                ],
            }
            print(
                D,
                s,
                {
                    k: (
                        round(v, 4)
                        if isinstance(v, float)
                        else [round(x, 4) for x in v]
                        if isinstance(v, list)
                        else v
                    )
                    for k, v in res[s].items()
                },
                flush=True,
            )
        out[f"{D}k"] = res
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/data_menu_vs_legnet.json", "w"), indent=1)
    print("wrote", f"{a.out}/data_menu_vs_legnet.json")


if __name__ == "__main__":
    main()
