#!/usr/bin/env python
"""Analysis of the Oct-22 experiment program on the finished cells.

  runs      ensemble accuracy against the number of independent AutoResearch runs (elbow), genomic 10k/30k/100k, random and motif 30k
  archsrc   does the reservoir(s) the architectures were searched on matter? menus built from 15 runs on genomic / random / a mix /
            the existing 5-reservoir family, each retrained on genomic, random and motif pools (equal candidate counts: top 3 of 15 runs)
  unseen    bespoke search (3 runs on own pools) vs the genomic-derived menu (3 independent pools), per reservoir and size, and the
            difference of differences: gap(genomic - X) under separate searches minus the same gap under the fixed menu
  mix       the 21-corpus mixture sweep at 30k, with single-reservoir controls and an ensemble across the best mixtures

Every ensemble is a non-negative combination of z-scored validation predictions (the repo's enet_all_pos for headline numbers, a fast
NNLS version for resampling), evaluated on held-out test sets against oracle labels. Run from the repo root on the HPC:

    python scripts/analysis/analyze_oct22.py --out outputs/justification_oct22 [--only runs,archsrc,unseen,mix]
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import make_justification as mj  # noqa: E402  (chdirs to the repo root, sets the plot style)

import matplotlib.pyplot as plt  # noqa: E402

EV = mj.EV
O22 = "outputs/archsearch_reps_oct22"
MR_ = "outputs/menu_retrain_v0"
DATA: dict = {}
BLUE, ORANGE, GREEN, RED, GREY, PURPLE, TEAL = mj.BLUE, mj.ORANGE, mj.GREEN, mj.RED, mj.GREY, mj.PURPLE, mj.TEAL


# ----------------------------------------------------------------------------------------------------------------- ensembles
def nnls_ens(V, yv, T, yt, cols):
    """Fast stand-in for enet_all_pos used inside resampling loops: NNLS on z-scored validation predictions (no shrinkage)."""
    cols = list(cols)
    sub = mj.subidx(len(yv))
    A = V[sub][:, cols]
    mu, sd = A.mean(0), A.std(0) + 1e-12
    w, _ = nnls((A - mu) / sd, yv[sub] - yv[sub].mean())
    return {e: mj.pear(((T[e][:, cols] - mu) / sd) @ w, yt[e]) for e in yt}


def run_cands(c, k=3, first_n=20):
    """Top-k models by validation among the first `first_n` of a run (the rule used to build the menus)."""
    val = c["val"][:first_n]
    return [int(i) for i in np.argsort(-val)[:k]]


def stack(cells_cols):
    """[(cell, [cols])...] -> pooled dict(V,T,yv,yt) with only the chosen columns."""
    V = np.column_stack([c["V"][:, cols] for c, cols in cells_cols])
    T = {e: np.column_stack([c["T"][e][:, cols] for c, cols in cells_cols]) for e in cells_cols[0][0]["T"]}
    return dict(V=V, T=T, yv=cells_cols[0][0]["yv"], yt=cells_cols[0][0]["yt"])


def menu_matrix(dirs):
    """Menu retrain dirs -> dict(V,T,yv,yt, runs=[origin cell per column]). One column per retrained model."""
    yv = yt = None
    cols_v, cols_t, origin, unit = [], {e: [] for e in EV}, [], []
    for di, d in enumerate(dirs):
        if not os.path.exists(f"{d}/labels.npz"):
            continue
        lab = np.load(f"{d}/labels.npz")
        yv, yt = lab["val_labels"].astype(float), {e: lab[f"oracle_{e}"].astype(float) for e in EV}
        for f in sorted(glob.glob(f"{d}/r*_menu_*_meta.json")):
            m = json.load(open(f))
            z = np.load(f.replace("_meta.json", ".npz"))
            vp = z["val_pred"].astype(float)
            tp = {e: z[f"test_pred_{e}"].astype(float) for e in EV}
            if not np.isfinite(vp).all() or np.abs(vp).max() > 1e3 or any((not np.isfinite(t).all()) or np.abs(t).max() > 1e3 for t in tp.values()):
                continue
            cols_v.append(vp)
            for e in EV:
                cols_t[e].append(tp[e])
            origin.append(m.get("origin_cell", m.get("menu_entry", "")))
            unit.append(di)
    if not cols_v:
        return None
    return dict(V=np.column_stack(cols_v), T={e: np.column_stack(cols_t[e]) for e in EV}, yv=yv, yt=yt, origin=np.array(origin), unit=np.array(unit))


def full_enet(d):
    """enet_all_pos on every column -> {set: r}, n_kept."""
    return mj.enet(d["V"], d["yv"], d["T"], d["yt"], range(d["V"].shape[1]))


def boot_units(d, unit_of, n_boot, rng, n_cols=None):
    """Bootstrap over resampling units (runs / pools): returns array (n_boot, n_sets) of NNLS ensemble r."""
    units = sorted(set(unit_of))
    idx = {u: np.where(np.asarray(unit_of) == u)[0] for u in units}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(len(units), len(units), replace=True)
        cols = np.concatenate([idx[units[i]] for i in pick])
        if n_cols and len(cols) > n_cols:  # equal model count: subsample the (larger) menu to the size of the competing ensemble
            cols = rng.choice(cols, n_cols, replace=False)
        r = nnls_ens(d["V"], d["yv"], d["T"], d["yt"], cols)
        out.append([r[e] for e in EV])
    return np.array(out)


def ci(a):
    return [float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def savefig(fig, name, out):
    os.makedirs(f"{out}/figs", exist_ok=True)
    fig.savefig(f"{out}/figs/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------ 1. run-count curves
def complete(d):
    return os.path.exists(f"{d}/COMPLETE") or len(glob.glob(f"{d}/r*_meta.json")) >= 20


def run_dirs(label):
    if label == "genomic 100k":
        ds = [f"outputs/archsearch_reps100k_oct21/genomic/rep{r}" for r in (1, 2, 3)] + [f"{O22}/genomic_100k/rep{r}" for r in range(4, 11)]
    else:
        key = {"genomic 10k": "genomic_10k", "genomic 30k": "genomic_30k", "random 30k": "random_30k", "motif 30k": "motif_ct_enriched_30k"}[label]
        ds = [f"{O22}/{key}/rep{r}" for r in range(1, 16)]
    return [d for d in ds if os.path.isdir(d) and complete(d)]


def an_runs(out, n_sub=60, seed=0):
    rng = np.random.default_rng(seed)
    res = {}
    for label in ("genomic 10k", "genomic 30k", "genomic 100k", "random 30k", "motif 30k"):
        cells = []
        for d in run_dirs(label):
            c = mj.load_cell(d)
            if c is not None and c["V"].shape[1] >= 6:
                cells.append(c)
        n = len(cells)
        if n < 3:
            continue
        cand = [run_cands(c) for c in cells]
        curve = {}
        for k in range(1, n + 1):
            combos = list(itertools.combinations(range(n), k)) if k in (1, n - 1, n) or n <= 8 else None
            if combos is None or len(combos) > n_sub:
                combos = [tuple(sorted(rng.choice(n, k, replace=False))) for _ in range(n_sub)]
            vals = []
            for cb in combos:
                d = stack([(cells[i], cand[i]) for i in cb])
                r = nnls_ens(d["V"], d["yv"], d["T"], d["yt"], range(d["V"].shape[1]))
                vals.append([r[e] for e in EV])
            vals = np.array(vals)
            curve[k] = dict(mean=vals.mean(0).tolist(), lo=vals.min(0).tolist(), hi=vals.max(0).tolist(), sd=vals.std(0).tolist(), n=len(vals))
        full = stack([(cells[i], cand[i]) for i in range(n)])
        e_r, kept = mj.enet(full["V"], full["yv"], full["T"], full["yt"], range(full["V"].shape[1]))
        m = np.array([curve[k]["mean"][0] for k in range(1, n + 1)])
        gain = m[-1] - m[0]
        k99 = int(next(k for k in range(1, n + 1) if m[k - 1] - m[0] >= 0.9 * gain))  # runs needed for 90% of the total gain
        marg = [float(m[k] - m[k - 1]) for k in range(1, n)]
        k_marg = int(next((k for k in range(1, n) if marg[k - 1] < 0.001), n))        # first k whose next run adds < 0.001
        res[label] = dict(n_runs=n, curve=curve, enet_all_runs=e_r, k90=k99, k_marginal_lt_0p001=k_marg, gain_1_to_n=float(gain), single_run_sd=curve[1]["sd"][0])
        print(f"[runs] {label}: n={n} k1={m[0]:.4f} kN={m[-1]:.4f} (enet {e_r['genomic']:.4f}) k90={k99} marginal<0.001 at k={k_marg}", flush=True)
    DATA["runs"] = res
    fig, axs = plt.subplots(1, len(res), figsize=(3.2 * len(res), 3.3), sharey=False)
    axs = np.atleast_1d(axs)
    for ax, (label, r) in zip(axs, res.items()):
        ks = sorted(r["curve"], key=int)
        mean = np.array([r["curve"][k]["mean"][0] for k in ks]); lo = np.array([r["curve"][k]["lo"][0] for k in ks]); hi = np.array([r["curve"][k]["hi"][0] for k in ks])
        ax.fill_between([int(k) for k in ks], lo, hi, color=BLUE, alpha=0.18, lw=0)
        ax.plot([int(k) for k in ks], mean, "o-", color=BLUE, ms=3.5)
        ax.axvline(r["k90"], color=ORANGE, ls="--", lw=1)
        ax.set_title(f"{label} ({r['n_runs']} runs)"); ax.set_xlabel("independent runs pooled")
        ax.text(0.97, 0.05, f"90% of gain at k={r['k90']}", transform=ax.transAxes, ha="right", fontsize=8.5, color=ORANGE)
    axs[0].set_ylabel("ensemble Pearson r (genomic test)")
    fig.suptitle("Ensemble accuracy against number of independent runs (top 3 per run; mean and min-max over subsets)", fontsize=10.5, y=1.02)
    savefig(fig, "run_count_curves", out)


# --------------------------------------------------------------------------------------------------- 2. architecture-source test
def an_archsrc(out, n_boot=150, seed=1):
    rng = np.random.default_rng(seed)
    conds = {"MG": "genomic only", "MR": "random only", "MGR": "genomic + random", "MF": "5 reservoirs", "MX": "searched on mixed pools"}
    targets = {"genomic": "genomic pool", "random": "random pool", "motif": "motif pool", "grm": "mixed pool (G+R+M)"}
    res = {}
    for t in targets:
        for c in conds:
            d = menu_matrix([f"{MR_}/as30k_{c}_{t}"])
            if d is None or d["V"].shape[1] < 30:
                continue
            e_r, kept = full_enet(d)
            b = boot_units(d, d["origin"], n_boot, rng)
            res[f"{c}|{t}"] = dict(n_models=int(d["V"].shape[1]), enet=e_r, kept=kept, boot={e: ci(b[:, i]) for i, e in enumerate(EV)}, boot_sd={e: float(b[:, i].std()) for i, e in enumerate(EV)}, boot_raw=b[:, 0].tolist())
            print(f"[archsrc] {c:3s} on {t:7s}: n={d['V'].shape[1]} genomic={e_r['genomic']:.4f} ood={e_r['ood']:.4f} random={e_r['random_32k']:.4f}", flush=True)
    # paired differences vs genomic-only menu, per target (bootstrap draws paired by index)
    diffs = {}
    for t in targets:
        if f"MG|{t}" not in res:
            continue
        for c in ("MR", "MGR", "MF", "MX"):
            if f"{c}|{t}" in res:
                diffs[f"{c}-MG|{t}"] = {"genomic_eval_boot": ci(np.array(res[f"{c}|{t}"]["boot_raw"]) - np.array(res[f"MG|{t}"]["boot_raw"])),
                                         "genomic_eval_enet": res[f"{c}|{t}"]["enet"]["genomic"] - res[f"MG|{t}"]["enet"]["genomic"],
                                         "ood_enet": res[f"{c}|{t}"]["enet"]["ood"] - res[f"MG|{t}"]["enet"]["ood"],
                                         "random_enet": res[f"{c}|{t}"]["enet"]["random_32k"] - res[f"MG|{t}"]["enet"]["random_32k"]}
    DATA["archsrc"] = dict(results=res, diffs=diffs)
    fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.4), sharey=False)
    cols = {"MG": BLUE, "MR": ORANGE, "MGR": TEAL, "MF": GREY, "MX": PURPLE}
    for ax, (t, tl) in zip(axs, targets.items()):
        for i, c in enumerate(conds):
            r = res.get(f"{c}|{t}")
            if not r:
                continue
            ax.errorbar(i, r["enet"]["genomic"], yerr=1.96 * r["boot_sd"]["genomic"], fmt="o", color=cols[c], capsize=3)
        ax.set_xticks(range(len(conds))); ax.set_xticklabels([f"{c}\n{conds[c]}" for c in conds], fontsize=7.5); ax.set_title(f"retrained on the {tl}", fontsize=9.5)
    axs[0].set_ylabel("ensemble r on genomic test (bars: +-1.96 SD over resampled runs)")
    fig.suptitle("Where the architectures were searched barely matters (same 45 candidates per source, retrained on each pool)", fontsize=10.5, y=1.03)
    savefig(fig, "arch_source", out)


# ------------------------------------------------------------------------------------------- 3. unseen reservoirs: search vs menu
UNSEEN = ["motif_shared_core", "motif_syntax_core", "zoonomia_orthologs", "mutagenesis", "gc_matched", "dinuc_shuffle"]
SEEN30 = ["random", "motif_ct_enriched"]


def bespoke_dirs(st, D):
    if st == "genomic" and D == 100000:
        return [f"outputs/archsearch_reps100k_oct21/genomic/rep{r}" for r in (1, 2, 3)]
    key = f"{st}_{D // 1000}k"
    return [f"{O22}/{key}/rep{r}" for r in (1, 2, 3)]


def menu_dirs(st, D):
    tag = {10000: "un10k", 30000: "un30k", 100000: "un100k"}[D]
    if D == 30000 and st in ("genomic", "random", "motif_ct_enriched"):
        tag = "ind30k"
    return [f"{MR_}/{tag}_{st}_r{r}" for r in (1, 2, 3)]


def bespoke_data(st, D):
    cells = [mj.load_cell(d) for d in bespoke_dirs(st, D) if os.path.isdir(d) and complete(d)]
    cells = [c for c in cells if c is not None]
    if len(cells) < 3:
        return None
    all_ = stack([(c, list(range(c["V"].shape[1]))) for c in cells])
    val = np.concatenate([c["val"] for c in cells])
    top8 = np.argsort(-val)[:8]
    return dict(all=all_, val=val, top8=top8, unit=np.concatenate([[i] * c["V"].shape[1] for i, c in enumerate(cells)]))


def menu_data(st, D):
    ds = [d for d in menu_dirs(st, D) if os.path.exists(f"{d}/COMPLETE_2of3")]
    if len(ds) < 3:
        return None
    return menu_matrix(ds)


def an_unseen(out, n_boot=200, seed=2):
    rng = np.random.default_rng(seed)
    res = {}
    for D in (10000, 30000, 100000):
        sts = ["genomic"] + UNSEEN + (SEEN30 if D == 30000 else [])
        for st in sts:
            b, m = bespoke_data(st, D), menu_data(st, D)
            if b is None or m is None:
                continue
            row = {}
            row["bespoke_all"], _ = mj.enet(b["all"]["V"], b["all"]["yv"], b["all"]["T"], b["all"]["yt"], range(b["all"]["V"].shape[1]))
            row["bespoke_top8"], _ = mj.enet(b["all"]["V"], b["all"]["yv"], b["all"]["T"], b["all"]["yt"], b["top8"])
            row["menu"], _ = full_enet(m)
            row["n_bespoke_models"], row["n_menu_models"] = int(b["all"]["V"].shape[1]), int(m["V"].shape[1])
            # equal-count control: the menu has more models (3 pools x ~33) than 3 bespoke runs (~60); average NNLS ensembles of random menu subsets of the bespoke size
            nbm = row["n_bespoke_models"]
            ba = nnls_ens(b["all"]["V"], b["all"]["yv"], b["all"]["T"], b["all"]["yt"], range(nbm))
            row["bespoke_all_nnls"] = ba
            if m["V"].shape[1] > nbm:
                dr = [nnls_ens(m["V"], m["yv"], m["T"], m["yt"], rng.choice(m["V"].shape[1], nbm, replace=False)) for _ in range(30)]
                row["menu_equal_count"] = {e: float(np.mean([d_[e] for d_ in dr])) for e in EV}
            else:
                row["menu_equal_count"] = nnls_ens(m["V"], m["yv"], m["T"], m["yt"], range(m["V"].shape[1]))
            bb = boot_units(b["all"], b["unit"], n_boot, rng); bm = boot_units(m, m["unit"], n_boot, rng, n_cols=nbm)
            row["boot_bespoke_all"] = [ci(bb[:, i]) for i in range(len(EV))]; row["boot_menu"] = [ci(bm[:, i]) for i in range(len(EV))]
            row["_bb"], row["_bm"] = bb[:, 0], bm[:, 0]
            res[f"{st}|{D}"] = row
            print(f"[unseen] {D // 1000:>3d}k {st:20s} bespoke_all={row['bespoke_all']['genomic']:.4f} top8={row['bespoke_top8']['genomic']:.4f} menu={row['menu']['genomic']:.4f}", flush=True)
    # difference of differences: gap(genomic - X) under separate searches minus the gap under the menu (positive = genomic gains from its own search more than X)
    dod = {}
    for key, row in res.items():
        st, D = key.split("|")
        if st == "genomic" or f"genomic|{D}" not in res:
            continue
        g = res[f"genomic|{D}"]
        for kind in ("bespoke_all", "bespoke_top8"):
            pt = {e: (g[kind][e] - row[kind][e]) - (g["menu"][e] - row["menu"][e]) for e in EV}
            dod[f"{st}|{D}|{kind}"] = dict(point=pt)
        pe = {e: (g["bespoke_all_nnls"][e] - row["bespoke_all_nnls"][e]) - (g["menu_equal_count"][e] - row["menu_equal_count"][e]) for e in EV}
        nb = min(len(g["_bb"]), len(row["_bb"]))
        d_ = (g["_bb"][:nb] - row["_bb"][:nb]) - (rng.permutation(g["_bm"])[:nb] - rng.permutation(row["_bm"])[:nb])  # draws are unpaired, so shuffle the menu draws
        dod[f"{st}|{D}|equal_count"] = dict(point=pe, boot_genomic_eval=ci(d_))
    DATA["unseen"] = dict(results={k: {a: b for a, b in v.items() if not a.startswith("_")} for k, v in res.items()}, dod=dod)
    # figure: for each size, own-search minus menu for each reservoir (genomic eval), with bootstrap bars
    sizes = [D for D in (10000, 30000, 100000) if any(k.endswith(f"|{D}") for k in res)]
    fig, axs = plt.subplots(1, len(sizes), figsize=(5.6 * len(sizes), 4.0), sharey=False, gridspec_kw=dict(wspace=0.75))
    axs = np.atleast_1d(axs)
    for ax, D in zip(axs, sizes):
        names = [k.split("|")[0] for k in res if k.endswith(f"|{D}")]
        for i, st in enumerate(names):
            r = res[f"{st}|{D}"]
            diff = r["_bb"] - rng.permutation(r["_bm"])[: len(r["_bb"])]
            m_, lo, hi = ci(diff)
            ax.errorbar(m_, i, xerr=[[m_ - lo], [hi - m_]], fmt="o", color=BLUE if st != "genomic" else GREEN, capsize=3)
        ax.axvline(0, color=GREY, lw=1); ax.axvspan(-0.005, 0.005, color=GREY, alpha=0.15, lw=0)
        ax.set_yticks(range(len(names))); ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=8.5); ax.invert_yaxis()
        ax.set_title(f"{D // 1000}k"); ax.set_xlabel("own search minus menu (genomic-test r)")
    fig.suptitle("A search on the strategy itself vs the genomic-derived menu (3 runs vs 3 pools; shaded = noise floor)", fontsize=10.5, y=1.03)
    savefig(fig, "unseen_search_vs_menu", out)


# ---------------------------------------------------------------------------------------------------------------- 4. mixture sweep
def mix_defs():
    sys.path.insert(0, "scripts")
    import importlib
    return importlib.import_module("build_mixture_pools").MIXES


def an_mix(out):
    mixes = mix_defs()
    ctr = {}
    for s, res_ in (("G", "genomic"), ("R", "random"), ("M", "motif_ct_enriched"), ("E", "evoaug_paper2023")):
        d = menu_matrix([f"{MR_}/ind30k_{res_}_r1"])
        if d is not None:
            ctr[s] = d
    rows, mats = {}, {}
    for name, spec in mixes.items():
        d = menu_matrix([f"{MR_}/mix30k_{name}"])
        if d is None or d["V"].shape[1] < 25:
            continue
        r, kept = full_enet(d)
        mats[name] = d
        rows[name] = dict(spec=spec, n_models=int(d["V"].shape[1]), enet=r, val_r=float(mj.pear(d["V"][:, :].mean(1), d["yv"])))
    controls = {}
    for s, d in ctr.items():
        r, _ = full_enet(d)
        controls[s] = r
    controls3 = {}  # three independent pools of one reservoir pooled (about 100+ models): the model-count match for ensembles over several mixtures
    for s, res_ in (("G", "genomic"), ("R", "random"), ("M", "motif_ct_enriched"), ("E", "evoaug_paper2023")):
        d3 = menu_matrix([f"{MR_}/ind30k_{res_}_r{r}" for r in (1, 2, 3)])
        if d3 is not None:
            r3, _ = full_enet(d3)
            controls3[s] = dict(enet=r3, n_models=int(d3["V"].shape[1]))
    for name, row in rows.items():
        comp = [k for k in row["spec"] if k in controls]
        if len(comp) == len(row["spec"]):  # mixture made only of G/R/M/E components: compare with the weight-averaged single-reservoir results
            row["single_mean"] = {e: float(sum(row["spec"][k] * controls[k][e] for k in comp)) for e in EV}
            row["gain_vs_single_mean"] = {e: row["enet"][e] - row["single_mean"][e] for e in EV}
    # ensemble across the best mixtures chosen on VALIDATION (mean val r of each mixture's ensemble), then evaluated on test
    rank = sorted(rows, key=lambda n: -rows[n]["val_r"])
    best3 = rank[:3]
    pool = {"V": np.column_stack([mats[n]["V"] for n in best3]), "T": {e: np.column_stack([mats[n]["T"][e] for n in best3]) for e in EV}, "yv": mats[best3[0]]["yv"], "yt": mats[best3[0]]["yt"]}
    ens_best, _ = mj.enet(pool["V"], pool["yv"], pool["T"], pool["yt"], range(pool["V"].shape[1]))
    allm = {"V": np.column_stack([mats[n]["V"] for n in mats]), "T": {e: np.column_stack([mats[n]["T"][e] for n in mats]) for e in EV}, "yv": mats[best3[0]]["yv"], "yt": mats[best3[0]]["yt"]}
    ens_all, _ = mj.enet(allm["V"], allm["yv"], allm["T"], allm["yt"], range(allm["V"].shape[1]))
    DATA["mix"] = dict(rows=rows, controls=controls, controls_3pools=controls3, n_models_best3=int(pool["V"].shape[1]), n_models_all=int(allm["V"].shape[1]), best3_by_val=best3, ensemble_best3=ens_best, ensemble_all_mixtures=ens_all)
    for n in sorted(rows, key=lambda n: -rows[n]["enet"]["genomic"]):
        r = rows[n]["enet"]
        print(f"[mix] {n:10s} genomic={r['genomic']:.4f} ood={r['ood']:.4f} random={r['random_32k']:.4f}", flush=True)
    print("[mix] controls", {k: round(v["genomic"], 4) for k, v in controls.items()}, "| best3 by val", best3, "ensemble", {e: round(v, 4) for e, v in ens_best.items()}, flush=True)
    names = sorted(rows, key=lambda n: rows[n]["enet"]["genomic"])
    fig, axs = plt.subplots(1, 3, figsize=(11, 5.4), sharey=True)
    for ax, e, tl in zip(axs, EV, ("genomic test", "designed (OOD)", "random")):
        ax.barh(range(len(names)), [rows[n]["enet"][e] for n in names], color=BLUE, alpha=0.85, height=0.7)
        for s, c in zip(("G", "R", "M", "E"), (GREEN, ORANGE, PURPLE, GREY)):
            if s in controls:
                ax.axvline(controls[s][e], color=c, ls="--", lw=1, label={"G": "genomic", "R": "random", "M": "motif", "E": "evoaug"}[s])
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8); ax.set_title(tl)
        lo_ = min(rows[n]["enet"][e] for n in names); ax.set_xlim(lo_ - 0.03, max(rows[n]["enet"][e] for n in names) + 0.01)
        ax.set_xlabel("ensemble r")
    axs[0].legend(loc="lower right", fontsize=7.5)
    fig.suptitle("Mixture sweep at 30k (menu retrained on each mixed corpus); dashed = single-reservoir controls", fontsize=10.5, y=1.0)
    savefig(fig, "mixture_sweep", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    ap.add_argument("--only", default="runs,archsrc,unseen,mix")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prev = {}
    if os.path.exists(f"{a.out}/data.json"):
        prev = json.load(open(f"{a.out}/data.json"))
    for step, fn in (("runs", an_runs), ("archsrc", an_archsrc), ("unseen", an_unseen), ("mix", an_mix)):
        if step in a.only.split(","):
            try:
                fn(a.out)
            except Exception as ex:  # keep the other analyses; report and continue
                import traceback
                traceback.print_exc()
                print(f"[{step}] FAILED: {ex}", flush=True)
            prev.update(DATA)
            json.dump(prev, open(f"{a.out}/data.json", "w"), indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
