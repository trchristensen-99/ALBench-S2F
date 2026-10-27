#!/usr/bin/env python
"""Oct-27 analyses: consistency of the coordinator effect, the extrapolation under other metrics, reduced menus and ensemble methods across reservoirs and sizes,
and the overnight experiments (training-recipe sensitivity, validation labels and HepG2 across sizes, loop-element controls, 600k retune).

    python scripts/analysis/analyze_oct27.py --only coord,extrap,reduced,ensmethods[,recipe,valdep,hgdep,loopctl,hptune] [--out outputs/justification_oct22]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats
from scipy.optimize import nnls

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_oct26 as b26  # noqa: E402

a = b26.a
EV, MR_ = b26.EV, b26.MR_
load, ens, common = b26.load, b26.ens, b26.common
RES = b26.RES
DATA: dict = {}
O22 = "outputs/archsearch_reps_oct22"


def dirs_for(st, D):
    k = D // 1000
    if D == 30000:
        return [f"ind30k_{st}_r1"]
    if D in (600000, 1000000):
        return [f"anchor_{st}_r1_d{k}k"]
    if D == 300000:
        return [f"nest_{st}_r1_d300k"]
    return [f"nest_genomic_r1_d{k}k"] if st == "genomic" else [f"lad_{st}_r1_d{k}k"]


def dl_random_effects(m, se):
    """DerSimonian-Laird random-effects pooling of per-cell mean differences."""
    m, se = np.asarray(m, float), np.asarray(se, float)
    w = 1 / se**2
    fe = float((w * m).sum() / w.sum())
    Q = float((w * (m - fe) ** 2).sum())
    k = len(m)
    C = w.sum() - (w**2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / C)
    ws = 1 / (se**2 + tau2)
    re = float((ws * m).sum() / ws.sum())
    return dict(
        k=k,
        fixed=fe,
        random=re,
        se_random=float(np.sqrt(1 / ws.sum())),
        tau=float(np.sqrt(tau2)),
        Q=Q,
        p_heterogeneity=float(stats.chi2.sf(Q, k - 1)),
        I2=float(max(0.0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0,
    )


# ---------------------------------------------------------------------------------------------------------------------- coord
def an_coord():
    """How consistent is the coordinator (and the explore prompt) effect across reservoirs and sizes? Random-effects meta-analysis over the grid cells."""
    o = json.load(open("outputs/justification_oct22/data.json"))
    res = {}
    for key, name in (("agent_grid", "coordinator"), ("grid", "explore")):
        cells = o.get(key, {}).get("cells", {})
        if not cells:
            continue
        sd_pool = float(
            np.sqrt(
                np.mean(
                    [v["genomic"]["sd_diff"] ** 2 for v in cells.values() if v["genomic"]["n"] > 1]
                )
            )
        )
        names = list(cells)
        m = [cells[k]["genomic"]["mean"] for k in names]
        n = [cells[k]["genomic"]["n"] for k in names]
        se = [sd_pool / np.sqrt(x) for x in n]
        r = {
            "cells": {k: dict(mean=mm, se=s, n=nn) for k, mm, s, nn in zip(names, m, se, n)},
            "sd_pool": sd_pool,
            "overall": dl_random_effects(m, se),
        }
        for label, sel in (
            ("30k", lambda k: k.endswith("_30k")),
            ("10k", lambda k: k.endswith("_10k")),
        ):
            idx = [i for i, k in enumerate(names) if sel(k)]
            if idx:
                w = np.array([1 / se[i] ** 2 for i in idx])
                mm = float(np.sum([w[j] * m[i] for j, i in enumerate(idx)]) / w.sum())
                r[f"size_{label}"] = dict(
                    mean=mm,
                    se=float(np.sqrt(1 / w.sum())),
                    k=len(idx),
                    n_pairs=int(sum(n[i] for i in idx)),
                )
        if "size_30k" in r and "size_10k" in r:
            dd = r["size_30k"]["mean"] - r["size_10k"]["mean"]
            sd_ = float(np.hypot(r["size_30k"]["se"], r["size_10k"]["se"]))
            r["size_difference_30k_minus_10k"] = dict(
                mean=dd, se=sd_, z=dd / sd_, p=float(2 * stats.norm.sf(abs(dd / sd_)))
            )
        r["sign_consistency"] = dict(positive=sum(1 for x in m if x > 0), of=len(m))
        res[name] = r
        ov = r["overall"]
        print(
            f"[coord] {name}: {len(names)} cells, {sum(n)} pairs; random-effects {ov['random']:+.4f} (SE {ov['se_random']:.4f}), fixed {ov['fixed']:+.4f}; tau {ov['tau']:.4f}, I2 {ov['I2']:.0%}, heterogeneity p {ov['p_heterogeneity']:.2f}; "
            + "; ".join(
                f"{k} {r[k]['mean']:+.4f} (SE {r[k]['se']:.4f})"
                for k in ("size_30k", "size_10k")
                if k in r
            )
            + (
                f"; 30k minus 10k {r['size_difference_30k_minus_10k']['mean']:+.4f} (p {r['size_difference_30k_minus_10k']['p']:.2f})"
                if "size_difference_30k_minus_10k" in r
                else ""
            )
            + f"; {r['sign_consistency']['positive']} of {r['sign_consistency']['of']} cells positive",
            flush=True,
        )
    DATA["coord"] = res


# -------------------------------------------------------------------------------------------------------------------- extrap
def test_pred(c, cols, e):
    sub = a.mj.subidx(len(c["yv"]))
    A = c["V"][sub][:, cols]
    mu, sd = A.mean(0), A.std(0) + 1e-12
    w, _ = nnls((A - mu) / sd, c["yv"][sub] - c["yv"][sub].mean())
    return ((c["T"][e][:, cols] - mu) / sd) @ w + c["yv"][sub].mean()


def an_extrap():
    """Measured 600k and 1M points against the 10k-300k power law, under several metrics: 1 - r, calibrated MSE (variance left after the best linear recalibration), raw MSE of the validation-fitted predictor, and 1 - Spearman."""
    res = {}
    for st in ("random", "motif_ct_enriched"):
        cs = {
            D: load(dirs_for(st, D), rounds=[0])
            for D in (10000, 30000, 100000, 300000, 600000, 1000000)
        }
        cs = {D: c for D, c in cs.items() if c is not None}
        idx = list(common(list(cs.values())))
        out = {"n_arch": len(idx), "points": {}, "fit": {}}
        for e in EV:
            met = {}
            for D, c in cs.items():
                cols = np.flatnonzero(np.isin(c["idx"], idx))
                p, y = test_pred(c, cols, e), c["yt"][e]
                r = a.mj.pear(p, y)
                met[D] = {
                    "one_minus_r": 1 - r,
                    "mse_calibrated": float(np.var(y) * (1 - r**2)),
                    "mse_raw": float(np.mean((p - y) ** 2)),
                    "one_minus_spearman": 1 - float(stats.spearmanr(p, y)[0]),
                }
            out["points"][e] = {str(D): v for D, v in met.items()}
            fitD = [D for D in (10000, 30000, 100000, 300000) if D in met]
            out["fit"][e] = {}
            for mname in ("one_minus_r", "mse_calibrated", "mse_raw", "one_minus_spearman"):
                bexp, c0 = np.polyfit(np.log(fitD), np.log([met[D][mname] for D in fitD]), 1)
                pred = {D: float(np.exp(c0) * D**bexp) for D in (600000, 1000000) if D in met}
                out["fit"][e][mname] = dict(
                    exponent=float(-bexp),
                    pred={str(D): v for D, v in pred.items()},
                    rel_error={str(D): float((met[D][mname] - pred[D]) / pred[D]) for D in pred},
                )
        res[st] = out
        f = out["fit"]["genomic"]
        print(
            f"[extrap] {st} ({len(idx)} architectures), genomic test, measured vs power-law prediction (relative error) at 600k / 1M: "
            + "; ".join(
                f"{m}: "
                + " / ".join(
                    f"{f[m]['rel_error'][D]:+.1%}"
                    for D in ("600000", "1000000")
                    if D in f[m]["rel_error"]
                )
                + f" (exponent {f[m]['exponent']:.3f})"
                for m in f
            ),
            flush=True,
        )
    DATA["extrap"] = res


# ------------------------------------------------------------------------------------------------------------------- reduced
def an_reduced(n_rand=40, seed=0):
    """Does retraining only K of the 33 architectures (chosen by the previous size's validation r) hold up on random and motif pools, as it did for genomic?"""
    rng = np.random.default_rng(seed)
    res = {}
    for st in RES:
        res[st] = {}
        for D, Dp in ((30000, 10000), (100000, 30000), (300000, 100000)):
            cur, prev = load(dirs_for(st, D), rounds=[0]), load(dirs_for(st, Dp), rounds=[0])
            if cur is None or prev is None:
                continue
            idx = sorted(set(cur["idx"]) & set(prev["idx"]))
            ci = {i: j for j, i in enumerate(cur["idx"])}
            pi = {i: j for j, i in enumerate(prev["idx"])}
            cols = np.array([ci[i] for i in idx])
            pv = np.array(
                [
                    prev["val"][pi[i]]
                    if "val" in prev
                    else a.mj.pear(prev["V"][:, pi[i]], prev["yv"])
                    for i in idx
                ]
            )
            n = len(idx)
            full = ens(cur, np.isin(np.arange(cur["V"].shape[1]), cols))
            row = {"n": n, "full": full["genomic"], "K": {}}
            for K in (5, 8, 12):
                sel = np.zeros(cur["V"].shape[1], bool)
                sel[cols[np.argsort(-pv)[:K]]] = True
                rr = np.mean(
                    [
                        ens(
                            cur,
                            np.isin(
                                np.arange(cur["V"].shape[1]), cols[rng.choice(n, K, replace=False)]
                            ),
                        )["genomic"]
                        for _ in range(n_rand)
                    ]
                )
                row["K"][str(K)] = {
                    "prev_val": ens(cur, sel)["genomic"] - full["genomic"],
                    "random": float(rr - full["genomic"]),
                }
            res[st][str(D)] = row
            print(
                f"[reduced] {st} {D // 1000}k ({n} architectures, chosen at {Dp // 1000}k): full {full['genomic']:.4f}; K=5/8/12 by previous-size val "
                + " / ".join(f"{row['K'][k]['prev_val']:+.4f}" for k in ("5", "8", "12"))
                + "; random K=8 "
                + f"{row['K']['8']['random']:+.4f}",
                flush=True,
            )
    DATA["reduced_all"] = res


# ---------------------------------------------------------------------------------------------------------------- ensmethods
def greedy_select(V, y, steps=25):
    """Forward selection with replacement on validation Pearson r of the running mean (Caruana)."""
    chosen, cur = [], np.zeros(len(y))
    for _ in range(steps):
        best, bi = -2, 0
        for j in range(V.shape[1]):
            r = a.mj.pear((cur * len(chosen) + V[:, j]) / (len(chosen) + 1), y)
            if r > best:
                best, bi = r, j
        chosen.append(bi)
        cur = V[:, chosen].mean(1)
    return chosen


def an_ensmethods():
    """Ensemble construction methods across reservoirs and sizes: mean of all, mean of the 8 best by validation, greedy selection, non-negative weights."""
    res = {}
    for st in RES:
        for D in (10000, 30000, 100000, 300000):
            c = load(dirs_for(st, D), rounds=[0])
            if c is None:
                continue
            n = c["V"].shape[1]
            val = np.array([a.mj.pear(c["V"][:, j], c["yv"]) for j in range(n)])
            row = {}
            for e in EV:
                T, y = c["T"][e], c["yt"][e]
                top8 = np.argsort(-val)[:8]
                g = greedy_select(c["V"], c["yv"])
                row[e] = {
                    "mean_all": a.mj.pear(T.mean(1), y),
                    "mean_top8": a.mj.pear(T[:, top8].mean(1), y),
                    "greedy": a.mj.pear(T[:, g].mean(1), y),
                    "nnls": ens(c, np.ones(n, bool))[e],
                    "best_single": float(max(a.mj.pear(T[:, j], y) for j in range(n))),
                    "best_single_by_val": a.mj.pear(T[:, int(np.argmax(val))], y),
                }
            res[f"{st}_{D // 1000}k"] = {"n": n, **row}
            r = row["genomic"]
            print(
                f"[ensmethods] {st} {D // 1000}k ({n} models), genomic test: mean all {r['mean_all']:.4f}, mean top-8 {r['mean_top8']:.4f}, greedy {r['greedy']:.4f}, non-negative {r['nnls']:.4f}; best single by val {r['best_single_by_val']:.4f}",
                flush=True,
            )
    DATA["ensmethods"] = res


STEPS = {"coord": an_coord, "extrap": an_extrap, "reduced": an_reduced, "ensmethods": an_ensmethods}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    ap.add_argument("--only", default=",".join(STEPS))
    args = ap.parse_args()
    path = f"{args.out}/data.json"
    for step in args.only.split(","):
        try:
            STEPS[step]()
        except Exception as ex:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"[{step}] FAILED: {ex}", flush=True)
        prev = json.load(open(path)) if os.path.exists(path) else {}
        prev.update(DATA)
        json.dump(prev, open(path, "w"), indent=1, default=float)
    print("wrote", path)


if __name__ == "__main__":
    main()
