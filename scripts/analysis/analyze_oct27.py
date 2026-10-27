#!/usr/bin/env python
"""Oct-27 analyses: consistency of the coordinator effect, the extrapolation under other metrics, reduced menus and ensemble methods across reservoirs and sizes,
and the overnight experiments (training-recipe sensitivity, validation labels and HepG2 across sizes, loop-element controls, 600k retune).

    python scripts/analysis/analyze_oct27.py --only coord,extrap,reduced,ensmethods[,recipe,valdep,hgdep,loopctl,hptune] [--out outputs/justification_oct22]
"""

from __future__ import annotations

import argparse
import glob
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
    DATA["extrap_metrics"] = res


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


# ---------------------------------------------------------------------------------------------------------------------- recipe
def meta_stats(dirs, rounds=(0,)):
    rows = []
    for d in dirs:
        d = d if d.startswith("outputs") else f"{MR_}/{d}"
        for f in glob.glob(f"{d}/r*_menu_*_meta.json"):
            m = json.load(open(f))
            if int(m["round"]) in rounds:
                rows.append(m)
    if not rows:
        return None
    ep = [m.get("epochs_trained") or 0 for m in rows]
    plan = [(m.get("train_config") or {}).get("epochs") or 0 for m in rows]
    return dict(
        n=len(rows),
        epochs=float(np.mean(ep)),
        planned=float(np.mean(plan)),
        frac_epoch_limit=float(np.mean([e >= p_ for e, p_ in zip(ep, plan)])),
        mean_val=float(np.mean([m["val_pearson"] for m in rows])),
        mean_test=float(np.mean([m["test_oracle_pearson"] for m in rows])),
        minutes=float(np.mean([m.get("train_time_sec", 0) for m in rows]) / 60),
    )


def an_recipe():
    """Training-recipe sensitivity of the retrained menu: patience 4 / 16 (default 8) and epoch budget x0.5 / x2, per reservoir and size, against the default-recipe baseline of the same pool and seed."""
    cells = [(st, 30000) for st in RES] + [("genomic", 10000), ("genomic", 100000)]
    res = {}
    for st, D in cells:
        k = D // 1000
        base = load(dirs_for(st, D), rounds=[0])
        if base is None:
            continue
        arms = {
            arm: load([f"tr{k}k_{arm}_{st}_r1"], rounds=[0])
            for arm in ("pat4", "pat16", "ep05", "ep2")
        }
        arms = {m: c for m, c in arms.items() if c is not None and c["V"].shape[1] >= 20}
        if not arms:
            continue
        idx = common([base, *arms.values()])
        cell = {
            "n_arch": len(idx),
            "default": dict(
                r=ens(base, np.isin(base["idx"], list(idx))), **(meta_stats(dirs_for(st, D)) or {})
            ),
        }
        for m, c in arms.items():
            cell[m] = dict(
                r=ens(c, np.isin(c["idx"], list(idx))),
                **(meta_stats([f"tr{k}k_{m}_{st}_r1"]) or {}),
            )
        res[f"{st}_{k}k"] = cell
        print(
            f"[recipe] {st} {k}k ({len(idx)} arch): default {cell['default']['r']['genomic']:.4f} ({cell['default'].get('epochs', 0):.0f} epochs)"
            + "; ".join(
                f" {m} {cell[m]['r']['genomic'] - cell['default']['r']['genomic']:+.4f} ({cell[m].get('epochs', 0):.0f} ep, {cell[m].get('minutes', 0):.0f} min)"
                for m in arms
            ),
            flush=True,
        )
    for m in ("pat4", "pat16", "ep05", "ep2"):
        d = [
            (v[m]["r"][e] - v["default"]["r"][e])
            for v in res.values()
            if m in v
            for e in ("genomic",)
        ]
        if d:
            res.setdefault("summary", {})[m] = dict(
                mean=float(np.mean(d)), min=float(np.min(d)), max=float(np.max(d)), n=len(d)
            )
    DATA["recipe"] = res


# ---------------------------------------------------------------------------------------------------------------------- valdep
def an_valdep():
    """Oracle-relabelled validation set (V0c) against the default one across sizes and reservoirs (single seed, ensemble weights and early stopping both on that set)."""
    res = {}
    for st in RES:
        for D in (10000, 30000, 100000):
            k = D // 1000
            base = load(dirs_for(st, D), rounds=[0])
            v0c = load([f"vv{k}k_V0c_{st}_r1"], rounds=[0])
            if base is None or v0c is None or v0c["V"].shape[1] < 20:
                continue
            idx = list(common([base, v0c]))
            rb, rv = ens(base, np.isin(base["idx"], idx)), ens(v0c, np.isin(v0c["idx"], idx))
            res[f"{st}_{k}k"] = dict(
                n_arch=len(idx),
                default=rb,
                v0c=rv,
                delta={e: rv[e] - rb[e] for e in EV},
                base_stats=meta_stats(dirs_for(st, D)),
                v0c_stats=meta_stats([f"vv{k}k_V0c_{st}_r1"]),
            )
            print(
                f"[valdep] {st} {k}k ({len(idx)} arch): default {rb['genomic']:.4f}, V0c {rv['genomic']:.4f} (delta {rv['genomic'] - rb['genomic']:+.4f}; ood {rv['ood'] - rb['ood']:+.4f}; random {rv['random_32k'] - rb['random_32k']:+.4f})",
                flush=True,
            )
    cells_ = {k: v for k, v in res.items()}
    if cells_:
        for e in EV:
            d = [v["delta"][e] for v in cells_.values()]
            res.setdefault("summary", {})[e] = dict(
                mean=float(np.mean(d)),
                se=float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None,
                n=len(d),
                positive=int(sum(x > 0 for x in d)),
            )
        # reservoir ordering preserved?
        order = {}
        for k in ("10k", "30k", "100k"):
            if all(f"{st}_{k}" in res for st in RES):
                order[k] = {
                    arm: sorted(RES, key=lambda s_: -res[f"{s_}_{k}"][arm]["genomic"])
                    for arm in ("default", "v0c")
                }
        res["order"] = order
    DATA["valdep"] = res


# ---------------------------------------------------------------------------------------------------------------------- hgdep
def an_hgdep():
    """Direct differential student against the difference of two students at 10k and 100k (the 30k result has nine runs)."""
    side = {
        e: np.load(
            f"outputs/oracle_v3_hepg2/relabel/eval_sets/{e}_oracle.multicell.npz", allow_pickle=True
        )
        for e in EV
    }
    lab = {
        e: {t: side[e][f"{t}_label"].astype(float) for t in ("k562", "hepg2", "diff")} for e in EV
    }
    res = {}
    for D in (10000, 100000):
        k = D // 1000
        for st in RES:
            arms = {
                "k562": load([f"vv{k}k_V0c_{st}_r1"], rounds=[0]),
                "hepg2": load([f"hg{k}k_hepg2_{st}_r1"], rounds=[0]),
                "diff": load([f"hg{k}k_diff_{st}_r1"], rounds=[0]),
            }
            if any(v is None or v["V"].shape[1] < 20 for v in arms.values()):
                continue
            idx = sorted(common(list(arms.values())))
            P = {
                t: b26.ens_pred(c, np.flatnonzero(np.isin(c["idx"], idx))) for t, c in arms.items()
            }
            out = {"n_arch": len(idx), "direct": {}, "two_students": {}}
            for e in EV:
                out["direct"][e] = {
                    t: a.mj.pear(P[t][e], lab[e][t]) for t in ("k562", "hepg2", "diff")
                }
                out["two_students"][e] = a.mj.pear(P["k562"][e] - P["hepg2"][e], lab[e]["diff"])
            res[f"{st}_{k}k"] = out
            print(
                f"[hgdep] {st} {k}k ({len(idx)} arch), genomic test: K562 {out['direct']['genomic']['k562']:.4f}, HepG2 {out['direct']['genomic']['hepg2']:.4f}, direct diff {out['direct']['genomic']['diff']:.4f} vs two students {out['two_students']['genomic']:.4f}",
                flush=True,
            )
    runs = [v for v in res.values()]
    if runs:
        for e in EV:
            d = [v["direct"][e]["diff"] - v["two_students"][e] for v in runs]
            res.setdefault("summary", {})[e] = dict(
                mean=float(np.mean(d)),
                min=float(np.min(d)),
                max=float(np.max(d)),
                n=len(d),
                positive=int(sum(x > 0 for x in d)),
            )
    DATA["hgdep"] = res


# ---------------------------------------------------------------------------------------------------------------------- loopctl
def an_loopctl():
    """Loop-element controls on the six genomic 30k pools, and the prompt-content (guidance) arms, paired with the default single-call runs."""
    import analyze_oct25 as b

    def runs(key, n=6, d=None):
        return [b.run_stats(f"{O22}/{key}/rep{r}") for r in range(1, n + 1)]

    arms = {
        "default single call": "genomic_30k",
        "separate coder (no coordinator, no reflector)": "genomic_30k_agent_plain",
        "6 ideas, 2 picked at random": "genomic_30k_agent_randpick",
        "6 ideas, 2 picked by the coordinator": "genomic_30k_agent_coord_noref",
        "coordinator + reflector": "genomic_30k_agent_coord",
        "explore-style proposer in the agent loop": "genomic_30k_agent_explore",
    }
    st = {k: runs(v) for k, v in arms.items()}
    res = {"arms": {}, "pairs": {}}
    for k, v in st.items():
        ok = [x for x in v if x]
        res["arms"][k] = dict(
            n=len(ok),
            mean=[float(np.mean([x["ens"][i] for x in ok])) for i in range(3)] if ok else None,
            mean_val=float(np.mean([x["mean_val"] for x in ok])) if ok else None,
        )

    def pair(x, y, label):
        ok = [i for i in range(6) if st[x][i] and st[y][i]]
        if len(ok) >= 3:
            res["pairs"][label] = {
                e: b.paired([st[x][i]["ens"][j] for i in ok], [st[y][i]["ens"][j] for i in ok])
                for j, e in enumerate(EV)
            }
            g = res["pairs"][label]["genomic"]
            print(
                f"[loopctl] {label}: {g['n']} pairs, genomic {g['mean']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}], {g['wins']} ahead",
                flush=True,
            )

    D0 = "default single call"
    for k in list(arms)[1:]:
        pair(D0, k, f"{k} minus default")
    pair(
        "6 ideas, 2 picked at random",
        "6 ideas, 2 picked by the coordinator",
        "coordinator minus random pick (same over-generation)",
    )
    pair(
        "separate coder (no coordinator, no reflector)",
        "6 ideas, 2 picked by the coordinator",
        "coordinator minus separate coder alone",
    )
    pair(
        "separate coder (no coordinator, no reflector)",
        "6 ideas, 2 picked at random",
        "random pick minus separate coder alone",
    )
    res["guidance"] = {}
    for tag, key_g, key_d, n in (
        ("random 30k", "random_30k_noguid", "random_30k", 6),
        ("motif 30k", "motif_ct_enriched_30k_noguid", "motif_ct_enriched_30k", 6),
        ("genomic 10k", "genomic_10k_noguid", "genomic_10k", 6),
    ):
        g_, d_ = runs(key_g, n), runs(key_d, n)
        ok = [i for i in range(n) if g_[i] and d_[i]]
        if len(ok) >= 3:
            res["guidance"][tag] = {
                e: b.paired([d_[i]["ens"][j] for i in ok], [g_[i]["ens"][j] for i in ok])
                for j, e in enumerate(EV)
            }
            g = res["guidance"][tag]["genomic"]
            print(
                f"[loopctl] guidance none minus known_good, {tag}: {g['n']} pairs, genomic {g['mean']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}], {g['wins']} ahead",
                flush=True,
            )
    DATA["loopctl"] = res


# ---------------------------------------------------------------------------------------------------------------------- hptune
def an_hptune():
    """Does retuning the recipe at 600k close the gap to the 10k-300k extrapolation? Three architectures of the random pool, one recipe change at a time, against their default-recipe 600k run."""
    base_dir = f"{MR_}/anchor_random_r1_d600k"
    k3 = json.load(open("outputs/acq_oct26/menu_k3_random600k.json"))
    ids = [e["id"] for e in k3]
    bm = {}
    for f in glob.glob(f"{base_dir}/r00_menu_*_meta.json"):
        m = json.load(open(f))
        if m["menu_entry"] in ids:
            bm[m["menu_entry"]] = (m, np.load(f.replace("_meta.json", ".npz")))
    lab = np.load(f"{base_dir}/labels.npz")
    yt = lab["oracle_genomic"].astype(float)
    res = {"arms": {}, "base": {}}
    for i in ids:
        if i in bm:
            res["base"][i] = dict(
                val=bm[i][0]["val_pearson"],
                test=a.mj.pear(bm[i][1]["test_pred_genomic"], yt),
                epochs=bm[i][0]["epochs_trained"],
                minutes=bm[i][0]["train_time_sec"] / 60,
            )
    for arm in ("lr05", "lr2", "bs4", "bs4lr2"):
        d = f"{MR_}/hptune600k_{arm}_random_r1"
        out = {}
        for f in glob.glob(f"{d}/r00_menu_*_meta.json"):
            m = json.load(open(f))
            if m["menu_entry"] in bm:
                z = np.load(f.replace("_meta.json", ".npz"))
                out[m["menu_entry"]] = dict(
                    val=m["val_pearson"],
                    test=a.mj.pear(z["test_pred_genomic"], yt),
                    epochs=m["epochs_trained"],
                    minutes=m["train_time_sec"] / 60,
                    d_val=m["val_pearson"] - bm[m["menu_entry"]][0]["val_pearson"],
                    d_test=a.mj.pear(z["test_pred_genomic"], yt)
                    - res["base"][m["menu_entry"]]["test"],
                )
        if out:
            res["arms"][arm] = out
            print(
                f"[hptune] {arm}: "
                + ", ".join(
                    f"{k[-14:]} dval {v['d_val']:+.4f} dtest {v['d_test']:+.4f} ({v['epochs']} ep, {v['minutes']:.0f} min)"
                    for k, v in out.items()
                ),
                flush=True,
            )
    DATA["hptune"] = res


def an_unseen_menu():
    """Retrained-menu ensemble r for every reservoir family at 10k, 30k and 100k (mean of the three pools); the Oct-22 unseen analysis skips 100k because it also needs a family-specific search there."""
    res = {}
    for D in (10000, 30000, 100000):
        for st in ["genomic", *a.UNSEEN, "random", "motif_ct_enriched"]:
            m = a.menu_data(st, D)
            if m is None:
                continue
            r, _ = a.full_enet(m)
            res[f"{st}|{D}"] = dict(
                menu={e: float(r[e]) for e in EV}, n_models=int(m["V"].shape[1])
            )
            print(
                f"[unseen_menu] {D // 1000}k {st}: " + " / ".join(f"{r[e]:.4f}" for e in EV),
                flush=True,
            )
    for st in ("genomic", "random", "motif_ct_enriched", "evoaug_paper2023", "motif_shared_core"):
        c = load(dirs_for(st, 300000), rounds=[0])
        if c is None:
            continue
        n = c["V"].shape[1]
        r = ens(c, np.ones(n, bool))
        res[f"{st}|300000"] = dict(
            menu={e: float(r[e]) for e in EV}, n_models=int(n), single_seed=True
        )
        print(
            f"[unseen_menu] 300k {st} (nested pool 1, {n} models): "
            + " / ".join(f"{r[e]:.4f}" for e in EV),
            flush=True,
        )
    DATA["unseen_menu"] = res


def kcurve_orders(menu, avail, seed=0, n_random=60):
    """Orderings of the menu entries that were successfully retrained (`avail`: menu indices). Rules:
    val      all entries by the validation r they had in their own search;
    first1 / first2   the best 1 (2) entries of every search run first (by search validation r), the rest by search validation r;
    random   n_random random permutations (reported as their mean)."""
    avail = list(avail)

    def v(i):
        x = menu[i].get("orig_val")
        return -1.0 if x is None else float(x)

    by_val = sorted(avail, key=lambda i: -v(i))
    out = {"val": by_val}
    runs: dict = {}
    for i in by_val:
        runs.setdefault(menu[i]["cell"], []).append(i)
    for nfirst in (1, 2):
        head = []
        for rank in range(nfirst):
            head += sorted((r[rank] for r in runs.values() if len(r) > rank), key=lambda i: -v(i))
        out[f"first{nfirst}"] = head + [i for i in by_val if i not in set(head)]
    rg = np.random.default_rng(seed)
    out["random"] = [list(rg.permutation(avail)) for _ in range(n_random)]
    return out


def kcurve_cell(menu, c, allowed=None):
    """For one retrained pool: ensemble r against the number K of configurations trained, per ordering rule, for the final 8-member ensemble
    (the 8 best of the K by retrained validation r, equal weights) and for non-negative weights over all K."""
    if c is None or c["V"].shape[1] < 12:
        return None
    col = {int(i): j for j, i in enumerate(c["idx"]) if allowed is None or int(i) in allowed}
    val = np.array([a.mj.pear(c["V"][:, j], c["yv"]) for j in range(c["V"].shape[1])])
    orders = kcurve_orders(menu, list(col))
    n = len(col)
    every = sorted(col.values())

    def top8(cols):
        cols = list(cols)
        sel = sorted(cols, key=lambda j: -val[j])[:8]
        return {e: a.mj.pear(c["T"][e][:, sel].mean(1), c["yt"][e]) for e in EV}

    res = {"n": n}
    for rule, od in orders.items():
        if rule == "random":
            acc = {e: np.zeros(n) for e in EV}
            for perm in od:
                for k in range(1, n + 1):
                    r = top8([col[i] for i in perm[:k]])
                    for e in EV:
                        acc[e][k - 1] += r[e] / len(od)
            res[rule] = {"top8": {e: acc[e].tolist() for e in EV}}
            continue
        t8 = {e: [] for e in EV}
        nn = {e: [] for e in EV}
        for k in range(1, n + 1):
            cols = [col[i] for i in od[:k]]
            r = top8(cols)
            for e in EV:
                t8[e].append(r[e])
            if k >= 2:
                q = a.nnls_ens(c["V"], c["yv"], c["T"], c["yt"], cols)
                for e in EV:
                    nn[e].append(q[e])
            else:
                for e in EV:
                    nn[e].append(r[e])
        res[rule] = {"top8": t8, "nnls": nn, "order": [int(i) for i in od]}
    res["full_nnls"] = a.nnls_ens(c["V"], c["yv"], c["T"], c["yt"], every)
    return res


def an_recipe2():
    """Patience 16 against the default (8) on replicate pools r1-r3, paired by pool: does the gain of the first seed hold on other pools, reservoirs and sizes?"""
    cells = [(s_, 30000) for s_ in RES] + [
        ("genomic", 10000),
        ("random", 10000),
        ("motif_ct_enriched", 10000),
        ("genomic", 100000),
    ]
    res = {}
    for st, D in cells:
        k = D // 1000
        rows = []
        for r in (1, 2, 3):
            base_dir = f"ind30k_{st}_r{r}" if D == 30000 else f"un{k}k_{st}_r{r}"
            base, arm = load([base_dir], rounds=[0]), load([f"tr{k}k_pat16_{st}_r{r}"], rounds=[0])
            if base is None or arm is None or arm["V"].shape[1] < 20 or base["V"].shape[1] < 20:
                continue
            idx = common([base, arm])
            rb, ra = (
                ens(base, np.isin(base["idx"], list(idx))),
                ens(arm, np.isin(arm["idx"], list(idx))),
            )
            ms_b, ms_a = meta_stats([base_dir]), meta_stats([f"tr{k}k_pat16_{st}_r{r}"])
            rows.append(
                {
                    "pool": r,
                    "n_arch": len(idx),
                    "default": rb,
                    "pat16": ra,
                    "delta": {e: ra[e] - rb[e] for e in EV},
                    "minutes_ratio": (ms_a or {}).get("minutes", 0)
                    / max((ms_b or {}).get("minutes", 1e-9), 1e-9),
                }
            )
        if not rows:
            continue
        d = [x["delta"]["genomic"] for x in rows]
        res[f"{st}_{k}k"] = {
            "pools": rows,
            "mean": float(np.mean(d)),
            "n": len(d),
            "positive": int(sum(x > 0 for x in d)),
            "mean_ood": float(np.mean([x["delta"]["ood"] for x in rows])),
            "mean_random": float(np.mean([x["delta"]["random_32k"] for x in rows])),
        }
        print(
            f"[recipe2] {st} {k}k: pat16 minus default per pool "
            + ", ".join(f"r{x['pool']} {x['delta']['genomic']:+.4f}" for x in rows)
            + f"; mean {np.mean(d):+.4f}; designed set {res[f'{st}_{k}k']['mean_ood']:+.4f}; random sequences {res[f'{st}_{k}k']['mean_random']:+.4f}; time x{np.mean([x['minutes_ratio'] for x in rows]):.2f}",
            flush=True,
        )
    allp = [x["delta"]["genomic"] for v in res.values() for x in v["pools"]]
    if allp:
        res["summary"] = {
            "mean": float(np.mean(allp)),
            "se": float(np.std(allp, ddof=1) / np.sqrt(len(allp))) if len(allp) > 1 else 0.0,
            "n": len(allp),
            "positive": int(sum(x > 0 for x in allp)),
            "by_size": {},
        }
        for k in (10, 30, 100):
            v = [
                x["delta"]["genomic"]
                for kk, c in res.items()
                if kk.endswith(f"_{k}k")
                for x in c["pools"]
            ]
            if v:
                res["summary"]["by_size"][str(k)] = {
                    "mean": float(np.mean(v)),
                    "n": len(v),
                    "positive": int(sum(x > 0 for x in v)),
                }
        print(
            f"[recipe2] all cells and pools: pat16 minus default {res['summary']['mean']:+.4f} (SE {res['summary']['se']:.4f}), {res['summary']['positive']} of {res['summary']['n']} positive; "
            + "; ".join(
                f"{k}k {v['mean']:+.4f} ({v['positive']}/{v['n']})"
                for k, v in res["summary"]["by_size"].items()
            ),
            flush=True,
        )
    DATA["recipe2"] = res


def an_plain_grid():
    """Separate-coder arm (agent loop without coordinator and reflector) against the default single-call loop, and the coordinator arm against it, paired by replicate pool, across reservoirs and sizes."""
    import analyze_oct25 as b

    cells = [
        ("genomic", 30000, 6),
        ("random", 30000, 6),
        ("motif_ct_enriched", 30000, 6),
        ("genomic", 10000, 6),
    ]
    res = {"cells": {}}
    px, py = [], []
    for st, D, n in cells:
        k = D // 1000
        xs, ys, zs = [], [], []
        for r in range(1, n + 1):
            d0, d1 = (
                b.run_stats(f"{O22}/{st}_{k}k/rep{r}"),
                b.run_stats(f"{O22}/{st}_{k}k_agent_plain/rep{r}"),
            )
            if not (d0 and d1):
                continue
            d2 = b.run_stats(f"{O22}/{st}_{k}k_agent_coord_noref/rep{r}")
            xs.append(d0["ens"])
            ys.append(d1["ens"])
            zs.append(d2["ens"] if d2 else [np.nan] * len(EV))
        if not xs:
            print(f"[plain_grid] {st} {k}k: no complete pairs yet", flush=True)
            continue
        xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
        cell = {e: b.paired(xs[:, i], ys[:, i]) for i, e in enumerate(EV)} | {
            "default_mean": xs.mean(0).tolist(),
            "plain_mean": ys.mean(0).tolist(),
        }
        ok = np.isfinite(zs[:, 0])
        if ok.sum() >= 3:
            cell["coordinator_minus_plain"] = {
                e: b.paired(ys[ok, i], zs[ok, i]) for i, e in enumerate(EV)
            }
        res["cells"][f"{st}_{k}k"] = cell
        px.append(xs)
        py.append(ys)
        g = cell["genomic"]
        cm = cell.get("coordinator_minus_plain", {}).get("genomic")
        print(
            f"[plain_grid] {st:17s} {k:2d}k: {g['n']} pairs, default {xs[:, 0].mean():.4f} separate coder {ys[:, 0].mean():.4f} diff {g['mean']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}] wins {g['wins']}/{g['n']}"
            + (
                f"; coordinator minus separate coder {cm['mean']:+.4f} ({cm['n']} pairs)"
                if cm
                else ""
            ),
            flush=True,
        )
    if px:
        X, Y = np.vstack(px), np.vstack(py)
        res["pooled"] = {e: b.paired(X[:, i], Y[:, i]) for i, e in enumerate(EV)}
        g = res["pooled"]["genomic"]
        print(
            f"[plain_grid] POOLED over {g['n']} paired runs: separate coder - default genomic {g['mean']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}] p={g['p']:.3f} wins {g['wins']}/{g['n']}",
            flush=True,
        )
    DATA["plain_grid"] = res


def kcurve_seeds_cell(menu, c, seeds=(0, 1, 2), allowed=None):
    """K configurations x several seeds on one pool (best-2-of-every-run-first order): the 8 best of the K x S trained models by validation r,
    against the 8 best of all configurations x S seeds."""
    if c is None:
        return None
    keep = np.isin(c["rnd"], list(seeds))
    cols_all = np.flatnonzero(keep)
    cols_all = np.array([j for j in cols_all if allowed is None or int(c["idx"][j]) in allowed])
    idxs = sorted({int(i) for i in c["idx"][cols_all]})
    if len(idxs) < 12:
        return None
    val = np.array([a.mj.pear(c["V"][:, j], c["yv"]) for j in range(c["V"].shape[1])])
    order = kcurve_orders(menu, idxs)["first2"]

    def top8(cols):
        sel = sorted(cols, key=lambda j: -val[j])[:8]
        return {e: a.mj.pear(c["T"][e][:, sel].mean(1), c["yt"][e]) for e in EV}

    top = {e: [] for e in EV}
    for k in range(1, len(order) + 1):
        cs = [j for j in cols_all if int(c["idx"][j]) in set(order[:k])]
        r = top8(cs)
        for e in EV:
            top[e].append(r[e])
    return {
        "n_models_full": int(len(cols_all)),
        "seeds": list(seeds),
        "top8": top,
        "full_top8": top8(list(cols_all)),
        "full_nnls": a.nnls_ens(c["V"], c["yv"], c["T"], c["yt"], list(cols_all)),
    }


def an_kcurve():
    """How many configurations need to be trained on a dataset for an approximately optimal ensemble? Retrained menu_v0 (33 configurations from four search runs
    and the reference) on every reservoir x size where three pools exist, plus the 45-candidate menus from 15 runs (MG, MF) at 30k."""
    menu = json.load(open("outputs/menu_retrain_v0/menu_v0.json"))
    # three configurations are left out of the shipped recipe: two exceed the CPU forward-time check on many nodes (so were retrained on only some pools)
    # and one produces exploding predictions on distribution-shifted sets (see menus/per_size.json "skip")
    skip_ids = {"son55_rep1_r00_arch_00", "son55_rep2_r15_arch_31", "son45_rep1_r11_arch_15"}
    usable = {i for i, e in enumerate(menu) if e["id"] not in skip_ids}

    def pools(st, D):
        k = D // 1000
        if D == 30000:
            return [f"ind30k_{st}_r{r}" for r in (1, 2, 3)]
        tag = f"un{k}k_{st}_r"
        if all(os.path.exists(f"{MR_}/{tag}{r}/COMPLETE_2of3") for r in (1, 2, 3)):
            return [f"{tag}{r}" for r in (1, 2, 3)]
        return [f"lad_{st}_r1_d{k}k"] if st != "genomic" else [f"nest_genomic_r1_d{k}k"]

    res = {}
    for st in RES:
        for D in (10000, 30000, 100000):
            loaded = [
                c
                for c in (load([d], rounds=[0]) for d in pools(st, D))
                if c is not None and c["V"].shape[1] >= 12
            ]
            if not loaded:
                continue
            common_idx = set.intersection(*[{int(i) for i in c["idx"]} for c in loaded]) & usable
            cells = [kcurve_cell(menu, c, common_idx) for c in loaded]
            res[f"menu33|{st}|{D}"] = {
                "pools": cells,
                "n_pools": len(cells),
                "pool_dirs": pools(st, D),
            }
            f = np.mean([c["full_nnls"]["genomic"] for c in cells])
            f8 = np.mean([c["val"]["top8"]["genomic"][-1] for c in cells])
            k90 = {}
            for rule in ("val", "first1", "first2", "random"):
                m = np.mean([c[rule]["top8"]["genomic"] for c in cells], 0)
                k90[rule] = {
                    tol: int(next((k + 1 for k in range(len(m)) if m[k] >= f8 - tol), len(m)))
                    for tol in (0.002, 0.003, 0.005)
                }
            res[f"menu33|{st}|{D}"]["k_needed_top8"] = {
                r: {str(t): v for t, v in kk.items()} for r, kk in k90.items()
            }
            res[f"menu33|{st}|{D}"]["full_top8"], res[f"menu33|{st}|{D}"]["full_nnls"] = (
                float(f8),
                float(f),
            )
            print(
                f"[kcurve] {st} {D // 1000}k ({len(cells)} pools, {len(common_idx)} configs): top-8 of all {f8:.4f}, non-negative weights on all {f:.4f}; K needed for the 8-member ensemble within 0.003 of the top-8 of all: "
                + ", ".join(f"{r} {v[0.003]}" for r, v in k90.items()),
                flush=True,
            )
    seedsrc = {
        ("genomic", 10000): ["nest_genomic_r1_d10k", "lad_genomic_r1_d10k_seeds12"],
        ("genomic", 30000): ["ind30k_genomic_r1", "ind30k_genomic_r1_seeds3to8"],
        ("genomic", 100000): ["nest_genomic_r1_d100k", "lad_genomic_r1_d100k_seeds12"],
        ("random", 30000): ["ind30k_random_r1", "ind30k_random_r1_seeds3to8"],
        ("motif_ct_enriched", 30000): [
            "ind30k_motif_ct_enriched_r1",
            "ind30k_motif_ct_enriched_r1_seeds3to8",
        ],
    }
    for (st, D), dirs in seedsrc.items():
        r = kcurve_seeds_cell(menu, load(dirs), allowed=usable)
        if r:
            res[f"seeds3|{st}|{D}"] = {"seeds": r}
            m = r["top8"]["genomic"]
            print(
                f"[kcurve] {st} {D // 1000}k x 3 seeds: top-8 of all {r['full_top8']['genomic']:.4f}; K=4 {m[3]:.4f}, K=8 {m[7]:.4f}, K=12 {m[11]:.4f}, K=16 {m[15]:.4f}",
                flush=True,
            )
    for src in ("MF", "MG"):
        m45 = json.load(open(f"outputs/menus_oct22/{src}.json"))
        for t in ("genomic", "random", "motif"):
            r = kcurve_cell(m45, load([f"as30k_{src}_{t}"], rounds=[0]))
            if r:
                res[f"menu45_{src}|{t}|30000"] = {
                    "pools": [r],
                    "n_pools": 1,
                    "pool_dirs": [f"as30k_{src}_{t}"],
                }
                m = r["first1"]["top8"]["genomic"]
                print(
                    f"[kcurve] {src} 45 candidates on {t} pool 30k: full {r['full_nnls']['genomic']:.4f}; top-8 first1 K=8 {m[7]:.4f}, K=16 {m[15]:.4f}, K=24 {m[23]:.4f}, K={len(m)} {m[-1]:.4f}",
                    flush=True,
                )
    DATA["kcurve"] = res


STEPS = {
    "coord": an_coord,
    "extrap": an_extrap,
    "reduced": an_reduced,
    "ensmethods": an_ensmethods,
    "recipe": an_recipe,
    "valdep": an_valdep,
    "hgdep": an_hgdep,
    "loopctl": an_loopctl,
    "hptune": an_hptune,
    "unseen_menu": an_unseen_menu,
    "kcurve": an_kcurve,
    "recipe2": an_recipe2,
    "plain_grid": an_plain_grid,
}


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
