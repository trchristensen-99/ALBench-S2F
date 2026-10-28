# ruff: noqa
#!/usr/bin/env python
"""Benchmark of our genomic ensembles (menu v1, 20 architectures) against MPRA-LegNet and AlphaGenome-FT-MPRA students trained on the same pool (Oct 28).

Every arm is read in the menu-retrain layout (labels.npz + r*_menu_*_meta.json/.npz) and scored by one code path: ensemble r on every test set for
  - the positive ElasticNetCV stack on z-scored validation predictions (albench.ensemble.stack.fit_zscored_stack), fitted on V0c for EVERY arm (the menu
    retrains early-stopped on a different labelling of the same validation sequences; their stored val_pred is re-used with the V0c labels), and
  - the plain mean of the members,
for subsets of N = 5, 10 (random subsets, mean over draws) and all members; single-model r (mean, sd, the model best on V0c); forward-only versus
reverse-complement-averaged for the full ensemble; and r against the EXPERIMENTAL labels on the genomic test set.

    python scripts/analysis/analyze_bench33.py [--out outputs/justification_oct22]      (cpuq)
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
from albench.paths import resolve  # noqa: E402

MR = "outputs/menu_retrain_v0"
MAIN = ("genomic", "ood", "random_32k")
N_SUB = {5: 8, 10: 8}  # subset size -> number of random draws


def pearson(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def load_dir(d):
    p = f"{MR}/{d}"
    if not os.path.exists(f"{p}/labels.npz"):
        return None
    lab = np.load(f"{p}/labels.npz")
    sets = [k[len("oracle_") :] for k in lab.files if k.startswith("oracle_")]
    rows = []
    for mf in sorted(glob.glob(f"{p}/r*_menu_*_meta.json")):
        try:
            meta = json.load(open(mf))
            z = np.load(mf.replace("_meta.json", ".npz"))
        except Exception:
            continue
        keys = [f"test_pred_{s}" for s in sets if f"test_pred_{s}" in z.files]
        if (
            not keys
            or not np.isfinite(z["val_pred"]).all()
            or any((not np.isfinite(z[k]).all()) or np.abs(z[k]).max() > 1e3 for k in keys)
        ):
            continue
        rows.append(
            {
                "meta": meta,
                "val": z["val_pred"].astype(np.float32),
                "test": {k[len("test_pred_") :]: z[k].astype(np.float32) for k in keys},
                "fwd": {
                    k[len("fwd_test_pred_") :]: z[k].astype(np.float32)
                    for k in z.files
                    if k.startswith("fwd_test_pred_")
                },
                "fwd_val": z["fwd_val_pred"].astype(np.float32)
                if "fwd_val_pred" in z.files
                else None,
            }
        )
    return {
        "rows": rows,
        "y": {s: lab[f"oracle_{s}"] for s in sets},
        "true": lab["test_true"],
        "sets": sets,
    }


def score(rows, c, yv, sets, sel, stack=True):
    """Ensemble r per set for the members `sel`: positive stack fitted on V0c and plain mean."""
    out = {"mean_pred": {}, "stack": {}}
    V = np.stack([rows[i]["val"] for i in sel])
    st = fit_zscored_stack(V, yv) if stack else None
    for s in sets:
        T = np.stack([rows[i]["test"][s] for i in sel])
        out["mean_pred"][s] = pearson(T.mean(0), c["y"][s])
        if stack:
            out["stack"][s] = pearson(st.predict(T), c["y"][s])
    return out


def arm_report(c, yv, label, rng):
    rows = c["rows"]
    sets = [s for s in c["sets"] if all(s in r["test"] for r in rows)]
    n = len(rows)
    rep = {"label": label, "n_models": n, "sets": sets, "ens": {}, "single": {}}
    val_r = np.array([pearson(r["val"], yv) for r in rows])
    rep["single"]["val_r_mean"] = float(val_r.mean())
    S = {s: np.array([pearson(r["test"][s], c["y"][s]) for r in rows]) for s in sets}
    rep["single"]["mean"] = {s: float(S[s].mean()) for s in sets}
    rep["single"]["sd"] = {s: float(S[s].std(ddof=1)) if n > 1 else None for s in sets}
    rep["single"]["best_val"] = {s: float(S[s][int(val_r.argmax())]) for s in sets}
    rep["single"]["exp_genomic_mean"] = float(
        np.mean([pearson(r["test"]["genomic"], c["true"]) for r in rows])
    )
    sizes = [k for k in sorted(N_SUB) if k < n] + [n]
    for k in sizes:
        draws = N_SUB.get(k, 1) if k < n else 1
        accs = []
        for _ in range(draws):
            sel = list(rng.choice(n, size=k, replace=False)) if k < n else list(range(n))
            accs.append(score(rows, c, yv, sets, sel, stack=(k >= 2)))
        rep["ens"][str(k)] = {
            kind: {s: float(np.nanmean([a[kind][s] for a in accs])) for s in sets}
            if (kind == "mean_pred" or k >= 2)
            else {}
            for kind in ("mean_pred", "stack")
        }
        rep["ens"][str(k)]["draws"] = draws
    # full ensemble: forward-only mean, and the experimental-label correlation
    Tg = np.stack([r["test"]["genomic"] for r in rows])
    rep["ens"][str(n)]["exp_genomic_mean_pred"] = pearson(Tg.mean(0), c["true"])
    if all("genomic" in r["fwd"] for r in rows):
        rep["ens"][str(n)]["fwd_mean_pred"] = {
            s: pearson(np.stack([r["fwd"][s] for r in rows]).mean(0), c["y"][s])
            for s in sets
            if all(s in r["fwd"] for r in rows)
        }
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    a = ap.parse_args()
    yv = np.asarray(np.load(resolve("val_v0c"), allow_pickle=True)["oracle_labels"], np.float32)
    rng = np.random.default_rng(20261028)
    data = {"val": "V0c (outputs/val_variants/val_V0.npz)", "arms": {}}
    for D in (30000, 10000):
        k = f"{D // 1000}k"
        arms = {f"menu_v1": (f"sc31_{k}_genomic", "menu v1, 20 architectures")}
        for cfg in ("pub", "e100", "b256"):
            arms[f"ln_{cfg}"] = (f"bench33_ln_{cfg}_{k}", f"MPRA-LegNet, {cfg}")
        for rec in ("oracle", "winner"):
            arms[f"ag2_{rec}"] = (
                f"bench33_agS2_{rec}_{k}",
                f"AlphaGenome-FT-MPRA S2 from S1, {rec} recipe",
            )
            arms[f"ag1_{rec}"] = (
                f"bench33_agS1_{rec}_{k}",
                f"AlphaGenome-FT-MPRA S1 (frozen encoder), {rec}",
            )
        res = {}
        for key, (d, label) in arms.items():
            c = load_dir(d)
            if c is None or not c["rows"]:
                continue
            res[key] = arm_report(c, yv, label, rng)
            r = res[key]
            n = r["n_models"]
            print(
                f"D={D} {key}: n={n} single genomic {r['single']['mean'].get('genomic', float('nan')):.4f} | ens(all) stack {r['ens'][str(n)]['stack'].get('genomic', float('nan')):.4f} mean {r['ens'][str(n)]['mean_pred'].get('genomic', float('nan')):.4f}",
                flush=True,
            )
        data["arms"][str(D)] = res
    os.makedirs(a.out, exist_ok=True)
    json.dump(data, open(f"{a.out}/data_bench33.json", "w"), indent=1, default=float)
    print("wrote", f"{a.out}/data_bench33.json")


if __name__ == "__main__":
    main()
