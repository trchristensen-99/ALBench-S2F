#!/usr/bin/env python
"""Planning analyses for the per-D procedure (search once, retrain a fixed menu at every D, ensemble):

  cost       GPU-time of a 33-architecture menu at 10k..300k and its extrapolation to 400k..1M (what a higher-D run costs)
  reduced    ensemble r when only K of the 33 menu architectures are retrained at the next D, chosen by the previous D's validation r, by the same D's
             validation r (needs the full retrain) or at random
  converge   how models end training at each D: time cap, patience, or the epoch limit (is a large D under-trained?)
  archexp    per-architecture scaling exponents against the ensemble's, and whether reservoir orderings hold architecture by architecture

    python scripts/analysis/analyze_oct25c.py --out outputs/justification_oct22
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
import make_justification as mj  # noqa: E402

EV = a.EV
MR_ = a.MR_
DATA: dict = {}
DIRS = {10000: ["g10k", "g10k_seeds12"], 30000: ["g30k", "g30k_seeds12"], 100000: ["g100k", "g100k_seeds12"], 300000: ["g300k"]}


def metas(D):
    rows = []
    for d in DIRS[D]:
        for f in glob.glob(f"{MR_}/{d}/r*_menu_*_meta.json"):
            m = json.load(open(f))
            rows.append(m)
    return rows


def menu_cell(D, seed=0):
    """Round-`seed` menu retrain at D as a load_cell dict with the menu index of every column."""
    for d in DIRS[D]:
        pat = f"r{seed:02d}_menu_*_meta.json"
        if glob.glob(f"{MR_}/{d}/{pat}"):
            c = mj.load_cell(f"{MR_}/{d}", pattern=pat)
            if c is not None:
                c["idx"] = np.array([m["menu_index"] for m in c["meta"]])
                return c
    return None


# ---------------------------------------------------------------------------------------------------------------------- cost
def an_cost():
    res = {}
    for D in DIRS:
        rows = metas(D)
        per = {}
        for m in rows:
            t = m.get("train_time_sec")
            if not t:
                continue
            g = m.get("gpu", "")
            t_h100 = t if "H100" in g else t / 2.4  # V100 about 0.42 of an H100 (measured throughput)
            per.setdefault(m["menu_entry"], []).append(t_h100)
        ent = {k: float(np.mean(v)) for k, v in per.items()}
        res[str(D)] = dict(n_entries=len(ent), mean_model_min=float(np.mean(list(ent.values())) / 60), median_model_min=float(np.median(list(ent.values())) / 60),
                           menu_gpu_h_one_seed=float(sum(ent.values()) / 3600), frac_h100=float(np.mean(["H100" in m.get("gpu", "") for m in rows])))
        print(f"[cost] {D // 1000}k: {len(ent)} entries, mean {res[str(D)]['mean_model_min']:.1f} min/model (H100-equivalent), menu of {len(ent)} = {res[str(D)]['menu_gpu_h_one_seed']:.1f} GPU-h per seed, H100 share {res[str(D)]['frac_h100']:.2f}", flush=True)
    Ds = np.array(sorted(int(k) for k in res), float)
    ys = np.array([res[str(int(D))]["menu_gpu_h_one_seed"] for D in Ds])
    b, c = np.polyfit(np.log(Ds), np.log(ys), 1)
    pred = {str(int(D)): float(np.exp(c) * D ** b) for D in (400_000, 500_000, 600_000, 700_000, 1_000_000)}
    res["fit_exponent"] = float(b)
    res["pred_gpu_h_one_seed"] = pred
    print(f"[cost] time ~ D^{b:.2f}; extrapolated menu GPU-h per seed:", {k: round(v, 1) for k, v in pred.items()}, flush=True)
    DATA["cost"] = res


# ------------------------------------------------------------------------------------------------------------------- reduced
def an_reduced(n_rand=40, seed=0):
    rng = np.random.default_rng(seed)
    res = {}
    for D, Dprev in ((30000, 10000), (100000, 30000), (300000, 100000)):
        cur, prev = menu_cell(D), menu_cell(Dprev)
        if cur is None or prev is None:
            continue
        common = sorted(set(cur["idx"]) & set(prev["idx"]))
        ci = {i: j for j, i in enumerate(cur["idx"])}
        pi = {i: j for j, i in enumerate(prev["idx"])}
        cols = np.array([ci[i] for i in common])
        prev_val = np.array([prev["val"][pi[i]] for i in common])
        own_val = np.array([cur["val"][ci[i]] for i in common])
        n = len(common)

        def ens(sel):
            r = a.nnls_ens(cur["V"], cur["yv"], cur["T"], cur["yt"], cols[sel])
            return [r[e] for e in EV]

        full = ens(np.arange(n))
        out = {"n": n, "full": full, "K": {}}
        for K in (3, 5, 8, 12, 16, 24):
            if K >= n:
                continue
            row = {"prev_val": ens(np.argsort(-prev_val)[:K]), "own_val": ens(np.argsort(-own_val)[:K]),
                   "random": np.mean([ens(rng.choice(n, K, replace=False)) for _ in range(n_rand)], axis=0).tolist()}
            out["K"][str(K)] = row
        res[str(D)] = out
        print(f"[reduced] {D // 1000}k (selected from {Dprev // 1000}k val): full {n} = {full[0]:.4f}; " + "; ".join(
            f"K={K}: prev-val {v['prev_val'][0]:.4f}, own-val {v['own_val'][0]:.4f}, random {v['random'][0]:.4f}" for K, v in out["K"].items()), flush=True)
    DATA["reduced"] = res


# ------------------------------------------------------------------------------------------------------------------ converge
def an_converge():
    res = {}
    for D in DIRS:
        rows = metas(D)
        plan = [(m.get("train_config") or {}).get("epochs") for m in rows]
        ep = [m.get("epochs_trained") for m in rows]
        cap = [bool(m.get("cap_hit")) for m in rows]
        full_ep = [e is not None and p is not None and e >= p for e, p in zip(ep, plan)]
        res[str(D)] = dict(n=len(rows), cap_hit=float(np.mean(cap)), reached_epoch_limit=float(np.mean(full_ep)),
                           stopped_by_patience=float(np.mean([not c and not f for c, f in zip(cap, full_ep)])),
                           median_epochs=float(np.median([e for e in ep if e])), median_planned=float(np.median([p for p in plan if p])))
        r = res[str(D)]
        print(f"[converge] {D // 1000}k: {r['n']} models, cap hit {r['cap_hit']:.0%}, reached the epoch limit {r['reached_epoch_limit']:.0%}, stopped by patience {r['stopped_by_patience']:.0%}, median epochs {r['median_epochs']:.0f} of {r['median_planned']:.0f} planned", flush=True)
    DATA["converge"] = res


# -------------------------------------------------------------------------------------------------------------------- archexp
def per_entry_r(d_dir, pattern="r00_menu_*_meta.json", eval_set="genomic"):
    lab = np.load(f"{d_dir}/labels.npz")
    yt = lab[f"oracle_{eval_set}"].astype(float)
    out = {}
    for f in glob.glob(f"{d_dir}/{pattern}"):
        m = json.load(open(f))
        z = np.load(f.replace("_meta.json", ".npz"))
        tp = z[f"test_pred_{eval_set}"].astype(float)
        if np.isfinite(tp).all() and np.abs(tp).max() < 1e3:
            out[m["menu_index"]] = mj.pear(tp, yt)
    return out


def an_archexp():
    res = {}
    # (a) per-architecture exponents, genomic reservoir 10k..300k, mean over the available retrainings
    r_by = {}
    for D in DIRS:
        acc = {}
        for d in DIRS[D]:
            lab = np.load(f"{MR_}/{d}/labels.npz")
            yt = lab["oracle_genomic"].astype(float)
            for f in glob.glob(f"{MR_}/{d}/r*_menu_*_meta.json"):
                m = json.load(open(f))
                tp = np.load(f.replace("_meta.json", ".npz"))["test_pred_genomic"].astype(float)
                if np.isfinite(tp).all() and np.abs(tp).max() < 1e3:
                    acc.setdefault(m["menu_index"], []).append(mj.pear(tp, yt))
        r_by[D] = {k: float(np.mean(v)) for k, v in acc.items()}
    idx = sorted(set.intersection(*[set(v) for v in r_by.values()]))
    Ds = np.array(sorted(r_by), float)
    exps = []
    for i in idx:
        y = np.array([1 - r_by[int(D)][i] for D in Ds])
        exps.append(-np.polyfit(np.log(Ds), np.log(y), 1)[0])
    ens_r = [menu_cell(int(D)) for D in Ds]
    ens_y = np.array([1 - a.nnls_ens(c["V"], c["yv"], c["T"], c["yt"], range(c["V"].shape[1]))["genomic"] for c in ens_r])
    b_ens = -np.polyfit(np.log(Ds), np.log(ens_y), 1)[0]
    res["exponents"] = dict(n=len(idx), mean=float(np.mean(exps)), sd=float(np.std(exps)), lo=float(np.percentile(exps, 5)), hi=float(np.percentile(exps, 95)), ensemble=float(b_ens), exps=[float(x) for x in exps])
    print(f"[archexp] per-architecture exponent over {len(idx)} architectures (10k to 300k, genomic): mean {np.mean(exps):.3f}, SD {np.std(exps):.3f}, 5-95% {np.percentile(exps, 5):.3f} to {np.percentile(exps, 95):.3f}; ensemble exponent {b_ens:.3f}", flush=True)
    # (b) reservoir orderings by architecture at 30k (round 0 of the independent-pool menu retrains, pool r1)
    pools = {"genomic": "ind30k_genomic_r1", "random": "ind30k_random_r1", "motif": "ind30k_motif_ct_enriched_r1", "evoaug": "ind30k_evoaug_paper2023_r1"}
    out = {}
    for es in ("genomic", "ood", "random_32k"):
        per = {k: per_entry_r(f"{MR_}/{v}", eval_set=es) for k, v in pools.items()}
        common = sorted(set.intersection(*[set(v) for v in per.values()]))
        pairs = {}
        for x, y in (("motif", "genomic"), ("random", "genomic"), ("evoaug", "genomic"), ("random", "motif")):
            d_ = np.array([per[x][i] - per[y][i] for i in common])
            pairs[f"{x} > {y}"] = dict(frac=float((d_ > 0).mean()), mean_diff=float(d_.mean()), n=len(common))
        out[es] = pairs
        print(f"[archexp] {es}: " + "; ".join(f"{k} in {v['frac']:.0%} of {v['n']} architectures (mean {v['mean_diff']:+.3f})" for k, v in pairs.items()), flush=True)
    res["reservoir_order"] = out
    DATA["archexp"] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    ap.add_argument("--only", default="cost,converge,reduced,archexp")
    args = ap.parse_args()
    prev = json.load(open(f"{args.out}/data.json")) if os.path.exists(f"{args.out}/data.json") else {}
    for step, fn in (("cost", an_cost), ("converge", an_converge), ("reduced", an_reduced), ("archexp", an_archexp)):
        if step in args.only.split(","):
            try:
                fn()
            except Exception as ex:
                import traceback
                traceback.print_exc()
                print(f"[{step}] FAILED: {ex}", flush=True)
            prev.update(DATA)
            json.dump(prev, open(f"{args.out}/data.json", "w"), indent=1, default=float)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
