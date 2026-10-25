#!/usr/bin/env python
"""Analyses of the Oct-26 experiment program (all from menu-retrain directories; ensemble = non-negative least squares on z-scored validation predictions).

noise    assay-noise test: ensemble test r on clean oracle labels when the training/validation labels carry Gosai-sized measurement error
seeds    nine seeds on one pool: ensemble r against the number of seeds, between-seed SD against between-pool SD
ladder   nested three-seed ladder at 10k/30k/100k: scaling exponent per seed and for the 3-seed ensemble
zoo      Zoonomia rebuilt after the validation-chromosome fix against the old pools
anchor   measured 600k / 1M points against the power-law extrapolation from 10k-300k (random and motif_ct_enriched)

  python scripts/analysis/analyze_oct26.py --only noise,seeds [--out outputs/justification_oct22]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_oct22 as a  # noqa: E402

EV, MR_ = a.EV, a.MR_
RES = ["genomic", "random", "motif_ct_enriched"]
DATA: dict = {}


def load(dirs, rounds=None):
    """Menu-retrain dirs -> dict(V, T, yv, yt, idx=menu index, rnd=round) over every finite model (optionally only some rounds)."""
    cols, T, idx, rnd, yv, yt = [], {e: [] for e in EV}, [], [], None, None
    for d in dirs:
        d = d if d.startswith("/") or d.startswith("outputs") else f"{MR_}/{d}"
        if not os.path.exists(f"{d}/labels.npz"):
            continue
        lab = np.load(f"{d}/labels.npz")
        if yv is None:
            yv, yt = (
                lab["val_labels"].astype(float),
                {e: lab[f"oracle_{e}"].astype(float) for e in EV},
            )
        for f in sorted(glob.glob(f"{d}/r*_menu_*_meta.json")):
            m = json.load(open(f))
            if rounds is not None and int(m["round"]) not in rounds:
                continue
            z = np.load(f.replace("_meta.json", ".npz"))
            vp = z["val_pred"].astype(float)
            tp = {e: z[f"test_pred_{e}"].astype(float) for e in EV}
            if (
                not np.isfinite(vp).all()
                or np.abs(vp).max() > 1e3
                or any((not np.isfinite(t).all()) or np.abs(t).max() > 1e3 for t in tp.values())
            ):
                continue
            cols.append(vp)
            for e in EV:
                T[e].append(tp[e])
            idx.append(int(m["menu_index"]))
            rnd.append(int(m["round"]))
    if not cols:
        return None
    return dict(
        V=np.column_stack(cols),
        T={e: np.column_stack(T[e]) for e in EV},
        yv=yv,
        yt=yt,
        idx=np.array(idx),
        rnd=np.array(rnd),
    )


def ens(c, mask):
    cols = np.flatnonzero(mask)
    return a.nnls_ens(c["V"], c["yv"], c["T"], c["yt"], cols) if len(cols) >= 2 else None


def common(cs):
    s = set.intersection(*[set(c["idx"]) for c in cs])
    return s


def exponent(Ds, rs):
    """b in (1 - r) ~ D^-b from the points (D, r)."""
    return float(-np.polyfit(np.log(Ds), np.log(1 - np.asarray(rs)), 1)[0])


def fmt(r):
    return " ".join(f"{e}={r[e]:.4f}" for e in EV) if r else "-"


# ---------------------------------------------------------------------------------------------------------------------- noise
def clean_dirs(st, D):
    if D == 30000:
        return [f"ind30k_{st}_r1"]
    k = D // 1000
    return [f"nest_genomic_r1_d{k}k"] if st == "genomic" else [f"lad_{st}_r1_d{k}k"]


def an_noise():
    res = {}
    for st in RES:
        res[st] = {}
        for D in (10000, 30000, 100000):
            k = D // 1000
            arms = {
                "0": load(clean_dirs(st, D), rounds=[0]),
                "1": load([f"nz1_{st}_r1_d{k}k"], rounds=[0]),
            }
            if D == 30000:
                arms["2"] = load([f"nz2_{st}_r1_d30k"], rounds=[0])
            arms = {s: c for s, c in arms.items() if c is not None}
            if len(arms) < 2:
                continue
            idx = common(list(arms.values()))
            res[st][str(D)] = {
                s: dict(r=ens(c, np.isin(c["idx"], list(idx))), n=len(idx)) for s, c in arms.items()
            }
            print(
                f"[noise] {st} {k}k ({len(idx)} common architectures): "
                + "; ".join(f"scale {s}: {fmt(v['r'])}" for s, v in res[st][str(D)].items()),
                flush=True,
            )
        for e in EV:
            for s in ("0", "1"):
                Ds = [
                    D for D in (10000, 30000, 100000) if str(D) in res[st] and s in res[st][str(D)]
                ]
                if len(Ds) == 3:
                    res[st].setdefault("exponent", {}).setdefault(e, {})[s] = exponent(
                        Ds, [res[st][str(D)][s]["r"][e] for D in Ds]
                    )
        if "exponent" in res[st]:
            print(
                f"[noise] {st} exponent clean vs scale-1 noise:",
                {e: {s: round(b, 3) for s, b in v.items()} for e, v in res[st]["exponent"].items()},
                flush=True,
            )
    DATA["noise"] = res


# ---------------------------------------------------------------------------------------------------------------------- seeds
def an_seeds(n_draw=60, seed=0):
    rng = np.random.default_rng(seed)
    res = {}
    for st in RES:
        c = load([f"ind30k_{st}_r1", f"ind30k_{st}_r1_seeds3to8"])
        if c is None:
            continue
        seeds = sorted(set(c["rnd"]))
        single = {int(s): ens(c, c["rnd"] == s) for s in seeds}
        trip = {}
        for g in ((0, 1, 2), (3, 4, 5), (6, 7, 8)):
            if all(s in seeds for s in g):
                trip[str(g)] = ens(c, np.isin(c["rnd"], g))
        curve = {}
        for kk in (1, 2, 3, 6, 9):
            if kk > len(seeds):
                continue
            vals = [
                ens(c, np.isin(c["rnd"], rng.choice(seeds, kk, replace=False)))
                for _ in range(1 if kk == len(seeds) else n_draw)
            ]
            curve[str(kk)] = {
                e: dict(
                    mean=float(np.mean([v[e] for v in vals])),
                    sd=float(np.std([v[e] for v in vals])),
                )
                for e in EV
            }
        pools = [load([f"ind30k_{st}_r{r}"], rounds=[0]) for r in (1, 2, 3)]
        pool_r = [ens(p, np.ones(p["V"].shape[1], bool)) for p in pools if p is not None]
        res[st] = dict(
            n_seeds=len(seeds),
            single=single,
            triplets=trip,
            curve=curve,
            between_seed_sd={e: float(np.std([v[e] for v in single.values()], ddof=1)) for e in EV},
            between_pool_sd={e: float(np.std([v[e] for v in pool_r], ddof=1)) for e in EV}
            if len(pool_r) >= 3
            else None,
            pool_r=pool_r,
        )
        print(
            f"[seeds] {st}: {len(seeds)} seeds; ensemble r by number of seeds "
            + "; ".join(
                f"{k}: {v['genomic']['mean']:.4f}±{v['genomic']['sd']:.4f}"
                for k, v in curve.items()
            )
            + f" | genomic SD between single-seed ensembles {res[st]['between_seed_sd']['genomic']:.4f}, between pools {res[st]['between_pool_sd']['genomic'] if res[st]['between_pool_sd'] else float('nan'):.4f}",
            flush=True,
        )
    DATA["seeds"] = res


# ---------------------------------------------------------------------------------------------------------------------- ladder
def ladder_dirs(st, D):
    k = D // 1000
    if D == 30000:
        return [f"ind30k_{st}_r1"]
    if st == "genomic":
        return [f"nest_genomic_r1_d{k}k", f"lad_genomic_r1_d{k}k_seeds12"]
    return [f"lad_{st}_r1_d{k}k"]


def an_ladder():
    res = {}
    for st in RES:
        cs = {D: load(ladder_dirs(st, D), rounds=[0, 1, 2]) for D in (10000, 30000, 100000)}
        if any(c is None for c in cs.values()):
            continue
        idx = common(list(cs.values()))
        out = {"n_arch": len(idx), "single": {}, "all3": {}}
        for D, c in cs.items():
            for s in (0, 1, 2):
                if (c["rnd"] == s).any():
                    out["single"].setdefault(str(s), {})[str(D)] = ens(
                        c, (c["rnd"] == s) & np.isin(c["idx"], list(idx))
                    )
            out["all3"][str(D)] = ens(c, np.isin(c["idx"], list(idx)))
        out["exponent"] = {}
        for e in EV:
            bs = []
            for s in ("0", "1", "2"):
                if s in out["single"] and all(
                    str(D) in out["single"][s] for D in (10000, 30000, 100000)
                ):
                    bs.append(
                        exponent(
                            [10000, 30000, 100000],
                            [out["single"][s][str(D)][e] for D in (10000, 30000, 100000)],
                        )
                    )
            out["exponent"][e] = dict(
                per_seed=bs,
                sd=float(np.std(bs, ddof=1)) if len(bs) > 1 else None,
                ensemble3=exponent(
                    [10000, 30000, 100000], [out["all3"][str(D)][e] for D in (10000, 30000, 100000)]
                ),
            )
        res[st] = out
        print(
            f"[ladder] {st}: exponent (genomic eval) per seed {np.round(out['exponent']['genomic']['per_seed'], 3)} (SD {out['exponent']['genomic']['sd']}), 3-seed ensembles {out['exponent']['genomic']['ensemble3']:.3f}",
            flush=True,
        )
    DATA["ladder"] = res


# ---------------------------------------------------------------------------------------------------------------------- zoo
def an_zoo():
    res = {}
    for D in (10000, 30000):
        k = D // 1000
        for r in (1, 2, 3):
            old, new = (
                load([f"un{k}k_zoonomia_orthologs_r{r}"], rounds=[0]),
                load([f"un{k}k_zoonomia_v2_r{r}"], rounds=[0]),
            )
            if old is None or new is None:
                continue
            idx = list(common([old, new]))
            res[f"{k}k_r{r}"] = dict(
                old=ens(old, np.isin(old["idx"], idx)),
                new=ens(new, np.isin(new["idx"], idx)),
                n=len(idx),
            )
            print(
                f"[zoo] {k}k r{r}: old {fmt(res[f'{k}k_r{r}']['old'])} | rebuilt {fmt(res[f'{k}k_r{r}']['new'])}",
                flush=True,
            )
    z = {}
    for r in (1, 2, 3):
        po, pn = (
            f"outputs/grid_pools/k562_reps/zoonomia_orthologs_r{r}_d30000.npz",
            f"outputs/grid_pools/k562_reps/zoonomia_orthologs_v2_r{r}_d30000.npz",
        )
        if os.path.exists(po) and os.path.exists(pn):
            so, sn = (
                set(np.load(po, allow_pickle=True)["sequences"]),
                set(np.load(pn, allow_pickle=True)["sequences"]),
            )
            z[f"r{r}"] = dict(n_old=len(so), n_new=len(sn), shared=len(so & sn))
    res["pool_overlap"] = z
    DATA["zoo"] = res


# -------------------------------------------------------------------------------------------------------------------- anchor
def an_anchor():
    res = {}
    for st in ("random", "motif_ct_enriched"):
        pts = {}
        for D, dirs in (
            (10000, ladder_dirs(st, 10000)),
            (30000, ladder_dirs(st, 30000)),
            (100000, ladder_dirs(st, 100000)),
            (300000, [f"nest_{st}_r1_d300k"]),
            (600000, [f"anchor_{st}_r1_d600k"]),
            (1000000, [f"anchor_{st}_r1_d1000k"]),
        ):
            c = load(dirs, rounds=[0])
            if c is not None:
                pts[D] = c
        if len(pts) < 5:
            print(f"[anchor] {st}: only {sorted(pts)} so far", flush=True)
        if not pts:
            continue
        idx = list(common(list(pts.values())))
        out = {
            "n_arch_common": len(idx),
            "points": {},
            "n_models": {str(D): int(c["V"].shape[1]) for D, c in pts.items()},
        }
        for D, c in pts.items():
            out["points"][str(D)] = ens(c, np.isin(c["idx"], idx))
        fitD = [D for D in (10000, 30000, 100000, 300000) if D in pts]
        out["pred"] = {}
        if len(fitD) >= 3:
            for e in EV:
                b, c0 = np.polyfit(
                    np.log(fitD), np.log([1 - out["points"][str(D)][e] for D in fitD]), 1
                )
                out["pred"][e] = {
                    str(D): float(1 - np.exp(c0) * D**b) for D in (600000, 1000000)
                } | {"exponent": float(-b)}
        res[st] = out
        print(
            f"[anchor] {st} ({len(idx)} common architectures): "
            + "; ".join(f"{D // 1000}k {out['points'][str(D)]['genomic']:.4f}" for D in pts)
            + (
                " | extrapolated from 10k-300k: "
                + ", ".join(
                    f"{D // 1000}k {out['pred']['genomic'][str(D)]:.4f}" for D in (600000, 1000000)
                )
                if out["pred"]
                else ""
            ),
            flush=True,
        )
    DATA["anchor"] = res


# -------------------------------------------------------------------------------------------------------------------------- hg
def ens_pred(c, cols):
    """Test-set predictions of the NNLS ensemble (same weights recipe as a.nnls_ens) for every evaluation set."""
    from scipy.optimize import nnls

    cols = list(cols)
    sub = a.mj.subidx(len(c["yv"]))
    A = c["V"][sub][:, cols]
    mu, sd = A.mean(0), A.std(0) + 1e-12
    w, _ = nnls((A - mu) / sd, c["yv"][sub] - c["yv"][sub].mean())
    return {e: ((c["T"][e][:, cols] - mu) / sd) @ w for e in EV}


def an_hg():
    """K562-optimised menu retrained on K562, HepG2 and differential (K562 - HepG2) labels: does the menu transfer, and is a difference of two students as good as a direct one?"""
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
    for st in RES:
        arms = {
            "k562": load([f"vv30k_V0c_{st}_r1"], rounds=[0]),
            "hepg2": load([f"hg30k_hepg2_{st}_r1"], rounds=[0]),
            "diff": load([f"hg30k_diff_{st}_r1"], rounds=[0]),
        }
        if any(v is None for v in arms.values()):
            print(
                f"[hg] {st}: missing arms {[k for k, v in arms.items() if v is None]}", flush=True
            )
            continue
        idx = sorted(common(list(arms.values())))
        P = {k: ens_pred(c, np.flatnonzero(np.isin(c["idx"], idx))) for k, c in arms.items()}
        out = {
            "n_arch": len(idx),
            "direct": {},
            "difference_of_students": {},
            "k562_student_as": {},
            "arch_rank_spearman": {},
        }
        for e in EV:
            out["direct"][e] = {t: a.mj.pear(P[t][e], lab[e][t]) for t in ("k562", "hepg2", "diff")}
            out["difference_of_students"][e] = a.mj.pear(
                P["k562"][e] - P["hepg2"][e], lab[e]["diff"]
            )
            out["k562_student_as"][e] = {
                t: a.mj.pear(P["k562"][e], lab[e][t]) for t in ("hepg2", "diff")
            }
        # single-architecture ranking agreement between targets (test r on each target's own labels), common architectures only
        from scipy.stats import spearmanr

        single = {}
        for k, c in arms.items():
            tgt = k
            single[k] = {
                e: np.array(
                    [
                        a.mj.pear(c["T"][e][:, int(np.flatnonzero(c["idx"] == i)[0])], lab[e][tgt])
                        for i in idx
                    ]
                )
                for e in EV
            }
        for e in EV:
            out["arch_rank_spearman"][e] = {
                "k562_vs_hepg2": float(spearmanr(single["k562"][e], single["hepg2"][e])[0]),
                "k562_vs_diff": float(spearmanr(single["k562"][e], single["diff"][e])[0]),
            }
        res[st] = out
        print(
            f"[hg] {st} ({len(idx)} architectures), genomic eval: direct k562 {out['direct']['genomic']['k562']:.4f}, hepg2 {out['direct']['genomic']['hepg2']:.4f}, diff {out['direct']['genomic']['diff']:.4f}"
            f" | diff from two students {out['difference_of_students']['genomic']:.4f} | k562 student on hepg2 labels {out['k562_student_as']['genomic']['hepg2']:.4f}"
            f" | architecture rank Spearman k562 vs hepg2 {out['arch_rank_spearman']['genomic']['k562_vs_hepg2']:.2f}",
            flush=True,
        )
    DATA["hg"] = res


STEPS = {
    "noise": an_noise,
    "seeds": an_seeds,
    "ladder": an_ladder,
    "zoo": an_zoo,
    "anchor": an_anchor,
    "hg": an_hg,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    ap.add_argument("--only", default=",".join(STEPS))
    args = ap.parse_args()
    path = f"{args.out}/data.json"
    prev = json.load(open(path)) if os.path.exists(path) else {}
    for step in args.only.split(","):
        try:
            STEPS[step]()
        except Exception as ex:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"[{step}] FAILED: {ex}", flush=True)
        prev.update(DATA)
        json.dump(prev, open(path, "w"), indent=1, default=float)
    print("wrote", path)


if __name__ == "__main__":
    main()
