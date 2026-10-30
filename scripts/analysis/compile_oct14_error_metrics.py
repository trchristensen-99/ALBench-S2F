#!/usr/bin/env python
"""Same cells and the same five aggregations as ``compile_oct14_scaling.py``, with the mean squared error against the oracle labels next to Pearson r.

Why a second script: the published table has r only, and MSE cannot be recovered from r (it needs the prediction scale and offset on each evaluation
set). The ensembles are rebuilt exactly as before (same model order, same validation subsample seed), so the ``<series>`` (r) columns reproduce the
published table and the ``<series>_mse`` columns belong to the same predictors. MSE of the SNV effect is the error of (alt - ref) against the
oracle's (alt - ref).

Columns per (reservoir, D, eval): for each series in f4_first6, top8_by_val, best_single, enet_greedy_k5, enet_all_pos: ``<series>`` (r) and
``<series>_mse``; plus ``label_var`` (variance of the oracle labels of the set, so MSE / label_var is a scale-free error) and ``label_mean``.
``cell_rows`` is reused by ``compile_nested_genomic_error_metrics.py`` for cells that are not AutoResearch searches.

    python scripts/analysis/compile_oct14_error_metrics.py [--out outputs/compile_oct14_scaling]      (cpuq, about as long as the original compile)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
from sklearn.linear_model import ElasticNetCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compile_oct14_scaling import (  # noqa: E402
    DS,
    EVALS,
    FAMILIES,
    MAXM,
    OUTSETS,
    VALSUB,
    greedy,
    load,
    pear,
)

SERIES = ("f4_first6", "top8_by_val", "best_single", "enet_greedy_k5", "enet_all_pos")


def mse(a, b) -> float:
    return float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))


def enet_both(cols, V, T, yval, yte, sub):
    en = ElasticNetCV(
        cv=5, n_alphas=40, max_iter=5000, l1_ratio=[0.1, 0.5, 0.9], positive=True
    ).fit(V[np.ix_(sub, cols)], yval[sub])
    out = {}
    for e in OUTSETS:
        if e == "snv_delta":
            p = en.predict(T["snv_alt"][:, cols]) - en.predict(T["snv_ref"][:, cols])
            y = yte["snv_alt"] - yte["snv_ref"]
        else:
            p, y = en.predict(T[e][:, cols]), yte[e]
        out[e] = (pear(p, y), mse(p, y))
    return out


def single_both(model_preds, e, yte):
    """(r, mse) of one model's predictions on evaluation set e."""
    if e == "snv_delta":
        p, y = model_preds["snv_alt"] - model_preds["snv_ref"], yte["snv_alt"] - yte["snv_ref"]
    else:
        p, y = model_preds[e], yte[e]
    return pear(p, y), mse(p, y)


def cell_rows(fam, res, D, ms, yval, yte, rng):
    """All rows (one per evaluation set) of one cell. ``ms`` are the cell's models in training order, each with ``valr``, ``val`` and ``test``
    (dict of predictions per set); ``yval`` and ``yte`` the labels. ``rng`` draws the validation subsample (consumed only for large validation sets)."""
    first6, top8 = ms[:6], sorted(ms, key=lambda m: -m["valr"])[:8]
    use = ms[:MAXM]
    V = np.column_stack([m["val"] for m in use])
    T = {e: np.column_stack([m["test"][e] for m in use]) for e in EVALS}
    sub = (
        np.arange(len(yval))
        if len(yval) <= VALSUB
        else rng.choice(len(yval), VALSUB, replace=False)
    )
    best = int(np.argmax([m["valr"] for m in use]))
    eg = enet_both(greedy(V, yval, sub), V, T, yval, yte, sub)
    ea = enet_both(list(range(V.shape[1])), V, T, yval, yte, sub)
    rows = []
    for e in OUTSETS:
        f6 = [single_both(m["test"], e, yte) for m in first6]
        t8 = [single_both(m["test"], e, yte) for m in top8]
        bp, bm = single_both({k: T[k][:, best] for k in EVALS}, e, yte)
        y = yte["snv_alt"] - yte["snv_ref"] if e == "snv_delta" else yte[e]
        rows.append(
            {
                "family": fam,
                "reservoir": res,
                "D": D,
                "eval": e,
                "n_models": len(ms),
                "label_mean": float(np.mean(y)),
                "label_var": float(np.var(y)),
                "f4_first6": float(np.mean([x[0] for x in f6])),
                "f4_first6_mse": float(np.mean([x[1] for x in f6])),
                "top8_by_val": float(np.mean([x[0] for x in t8])),
                "top8_by_val_mse": float(np.mean([x[1] for x in t8])),
                "best_single": bp,
                "best_single_mse": bm,
                "enet_greedy_k5": eg[e][0],
                "enet_greedy_k5_mse": eg[e][1],
                "enet_all_pos": ea[e][0],
                "enet_all_pos_mse": ea[e][1],
            }
        )
    gm = float(np.mean(T["genomic"]))
    print(
        fam,
        res,
        D,
        "n",
        len(ms),
        "genomic r/mse (deploy)",
        round(eg["genomic"][0], 4),
        round(eg["genomic"][1], 4),
        "label mean",
        round(float(yte["genomic"].mean()), 3),
        "mean member pred",
        round(gm, 3),
        flush=True,
    )
    return rows


def write_rows(rows, out_dir, stem="scaling_error_cells"):
    keys = list(rows[0].keys())
    with open(f"{out_dir}/{stem}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    json.dump(rows, open(f"{out_dir}/{stem}.json", "w"), default=float)
    print("wrote", f"{out_dir}/{stem}.csv", len(rows), "rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/compile_oct14_scaling")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for fam, ress in FAMILIES:
        for res in ress:
            for D in DS:
                c = load(res, D)
                if c is None or not c[0]:
                    print("MISSING", res, D)
                    continue
                ms, yval, yte = c
                rows += cell_rows(fam, res, D, ms, yval, yte, rng)
    write_rows(rows, a.out)


if __name__ == "__main__":
    main()
