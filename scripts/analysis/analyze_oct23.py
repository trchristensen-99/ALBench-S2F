#!/usr/bin/env python
"""Follow-up analyses for the Oct-23 review of the Oct-22 program.

  runs2     ensemble against number of runs for several top-k per run, and for runs drawn from several reservoirs (how the final ensemble will be built)
  oodwhy    why a random-trained student does well on the designed (high-activity) set: label tails, composition support, calibration
  extrap    power-law fits of the menu scaling curves and extrapolation to 1M and 4M with resampling uncertainty
  legref    the tuned reference LegNet inside the 33-architecture menu, ranked on every pool we have retrained it on

    python scripts/analysis/analyze_oct23.py --out outputs/justification_oct22 [--only runs2,oodwhy,extrap,legref]
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_oct22 as a  # noqa: E402  (loaders, NNLS ensemble; chdirs to the repo root)
import make_justification as mj  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.linear_model import ElasticNetCV  # noqa: E402

EV = a.EV
O22, MR_ = a.O22, a.MR_
DATA: dict = {}
BLUE, ORANGE, GREEN, RED, GREY, PURPLE, TEAL = a.BLUE, a.ORANGE, a.GREEN, a.RED, a.GREY, a.PURPLE, a.TEAL


# ------------------------------------------------------------------------------------------------------- 1. run-count curves v2
def load_runs(dirs):
    cs = [mj.load_cell(d) for d in dirs if os.path.isdir(d)]
    return [c for c in cs if c is not None and c["V"].shape[1] >= 6]


def ens_runs(cells, idx, k):
    d = a.stack([(cells[i], a.run_cands(cells[i], k=k, first_n=20)) for i in idx])
    r = a.nnls_ens(d["V"], d["yv"], d["T"], d["yt"], range(d["V"].shape[1]))
    return [r[e] for e in EV]


def an_runs2(out, n_sub=24, seed=0):
    rng = np.random.default_rng(seed)
    src = {"genomic": [f"{O22}/genomic_30k/rep{r}" for r in range(1, 16)], "random": [f"{O22}/random_30k/rep{r}" for r in range(1, 16)],
           "motif": [f"{O22}/motif_ct_enriched_30k/rep{r}" for r in range(1, 16)]}
    cells = {s: load_runs([d for d in ds if os.path.exists(d + "/COMPLETE")]) for s, ds in src.items()}
    res = {"topk": {}, "mixed": {}, "five": {}}
    # (a) top-k per run, single source
    for s, cs in cells.items():
        n = len(cs)
        res["topk"][s] = {}
        for k in (1, 2, 3, 5, 8, 20):
            curve = []
            for m in range(1, n + 1):
                nsub = n_sub if m not in (n,) else 1
                combos = [tuple(sorted(rng.choice(n, m, replace=False))) for _ in range(nsub)] if m < n else [tuple(range(n))]
                curve.append(np.mean([ens_runs(cs, cb, k) for cb in combos], axis=0).tolist())
            res["topk"][s][str(k)] = curve
        print(f"[runs2] top-k {s}: n={n} k=3 -> {res['topk'][s]['3'][0][0]:.4f} (1 run) .. {res['topk'][s]['3'][-1][0]:.4f} ({n} runs); k=20 -> {res['topk'][s]['20'][-1][0]:.4f}", flush=True)
    # (b) runs drawn from several reservoirs, k=3: balanced genomic+random+motif, and single sources at the same total number of runs
    G, R, M = cells["genomic"], cells["random"], cells["motif"]
    for tot in range(3, 3 * min(len(G), len(R), len(M)) + 1, 3):
        m = tot // 3
        vals = []
        nsub = n_sub if m < min(len(G), len(R), len(M)) else 1
        for _ in range(nsub):
            sel = [(G, rng.choice(len(G), m, replace=False)), (R, rng.choice(len(R), m, replace=False)), (M, rng.choice(len(M), m, replace=False))]
            parts = [(c[i], a.run_cands(c[i], k=3, first_n=20)) for c, ids in sel for i in ids]
            d = a.stack(parts)
            r = a.nnls_ens(d["V"], d["yv"], d["T"], d["yt"], range(d["V"].shape[1]))
            vals.append([r[e] for e in EV])
        res["mixed"][str(tot)] = np.mean(vals, axis=0).tolist()
    print("[runs2] mixed G+R+M (k=3):", {t: round(v[0], 4) for t, v in res["mixed"].items()}, flush=True)
    # (c) the planned structure: 3 runs from each of 5 reservoirs (15 runs), balanced draws at 5, 10 and 15 runs
    fam = {"genomic": [f"outputs/archsearch_final_oct20/son55_wide/rep{r}" for r in (1, 2, 3)]}
    for f in ("random", "motif_ct_enriched", "evo2", "evoaug_paper2023"):
        fam[f] = [f"outputs/archsearch_family_oct20/{f}/rep{r}" for r in (1, 2, 3)]
    fc = {f: load_runs(ds) for f, ds in fam.items()}
    if all(len(v) == 3 for v in fc.values()):
        for m in (1, 2, 3):
            vals = []
            for _ in range(n_sub if m < 3 else 1):
                parts = []
                for f, cs in fc.items():
                    ids = rng.choice(3, m, replace=False) if m < 3 else range(3)
                    parts += [(cs[i], a.run_cands(cs[i], k=3, first_n=20)) for i in ids]
                d = a.stack(parts)
                r = a.nnls_ens(d["V"], d["yv"], d["T"], d["yt"], range(d["V"].shape[1]))
                vals.append([r[e] for e in EV])
            res["five"][str(5 * m)] = np.mean(vals, axis=0).tolist()
        print("[runs2] five-reservoir plan (k=3):", {t: [round(x, 4) for x in v] for t, v in res["five"].items()}, flush=True)
    DATA["runs2"] = res
    # figure 1: top-k
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.5), sharey=False)
    cols = {"1": GREY, "2": ORANGE, "3": BLUE, "5": TEAL, "8": PURPLE, "20": GREEN}
    for ax, s in zip(axs, res["topk"]):
        for k, curve in res["topk"][s].items():
            ax.plot(range(1, len(curve) + 1), [c[0] for c in curve], "o-", ms=2.8, lw=1.3, color=cols[k], label=f"top {k}" if k != "20" else "all 20")
        ax.set_title(f"{s} 30k"); ax.set_xlabel("independent runs pooled")
    axs[0].set_ylabel("ensemble r (genomic test)"); axs[0].legend(ncols=2, fontsize=7.5, loc="lower right")
    fig.suptitle("How many models to keep from each run: ensemble against runs for different top-k", fontsize=10.5, y=1.02)
    a.savefig(fig, "runcount_topk", out)
    # figure 2: mixed sources against single sources at equal total runs
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.5))
    for ax, (ei, e) in zip(axs, enumerate(EV)):
        for s, col in zip(("genomic", "random", "motif"), (GREEN, ORANGE, PURPLE)):
            curve = res["topk"][s]["3"]
            ax.plot(range(1, len(curve) + 1), [c[ei] for c in curve], "-", color=col, lw=1.3, label=f"{s} runs only")
        xs = sorted(int(t) for t in res["mixed"])
        ax.plot(xs, [res["mixed"][str(t)][ei] for t in xs], "o-", color=BLUE, lw=1.8, ms=4, label="genomic + random + motif (balanced)")
        if res["five"]:
            xf = sorted(int(t) for t in res["five"])
            ax.plot(xf, [res["five"][str(t)][ei] for t in xf], "s", color=RED, ms=6, label="5 reservoirs, balanced")
        ax.set_title({"genomic": "genomic test", "ood": "designed set (OOD)", "random_32k": "random sequences"}[e]); ax.set_xlabel("total independent runs pooled (top 3 each)")
    axs[0].set_ylabel("ensemble r"); axs[0].legend(fontsize=7.2, loc="lower right")
    fig.suptitle("Runs drawn from several reservoirs, as the final ensemble will be built, against runs from one reservoir", fontsize=10.5, y=1.02)
    a.savefig(fig, "runcount_mixed_sources", out)


# --------------------------------------------------------------------------------------------------------- 2. why random does well on OOD
def seq_stats(seqs):
    gc = np.array([(s.count("G") + s.count("C")) / len(s) for s in seqs])
    cpg = np.array([s.count("CG") / max(1, len(s) - 1) for s in seqs])
    return gc, cpg


def enet_pred(d, target):
    sub = mj.subidx(len(d["yv"]))
    mu, sd = d["V"][sub].mean(0), d["V"][sub].std(0) + 1e-12
    en = ElasticNetCV(cv=5, n_alphas=40, max_iter=5000, l1_ratio=[.1, .5, .9], positive=True).fit((d["V"][sub] - mu) / sd, d["yv"][sub])
    return en.predict((d["T"][target] - mu) / sd)


def an_oodwhy(out):
    reps = "outputs/grid_pools/k562_reps"
    ood = np.load("outputs/eval_sets_v3/labelled/ood.npz", allow_pickle=True)
    oseq, olab = list(ood["sequences"]), ood["oracle_labels"].astype(float)
    ogc, ocpg = seq_stats(oseq)
    res = {"designed": dict(n=len(oseq), gc=[float(ogc.mean()), float(ogc.std())], cpg=[float(ocpg.mean()), float(ocpg.std())], label_mean=float(olab.mean()),
                            label_q=[float(x) for x in np.percentile(olab, [5, 25, 50, 75, 95])])}
    pools = {"genomic": "genomic", "random": "random", "motif": "motif_ct_enriched", "evoaug": "evoaug_paper2023", "gc_matched": "gc_matched", "dinuc_shuffle": "dinuc_shuffle"}
    gcbins, cpgbins = np.arange(0, 1.0001, 0.05), np.arange(0, 0.12, 0.01)
    og = np.digitize(ogc, gcbins), np.digitize(ocpg, cpgbins)
    for name, st in pools.items():
        f = f"{reps}/{st}_r1_d30000.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        gc, cpg = seq_stats(list(z["sequences"]))
        lab = z["oracle_labels"].astype(float)
        H, _, _ = np.histogram2d(gc, cpg, bins=[gcbins, cpgbins])
        occ = H[np.clip(og[0] - 1, 0, H.shape[0] - 1), np.clip(og[1] - 1, 0, H.shape[1] - 1)]
        res[name] = dict(gc=[float(gc.mean()), float(gc.std())], cpg=[float(cpg.mean()), float(cpg.std())], label_mean=float(lab.mean()), label_sd=float(lab.std()),
                         frac_label_above={str(t): float((lab > t).mean()) for t in (1.0, 2.0, 3.0, 4.0)}, label_max=float(lab.max()),
                         designed_gc_cpg_support=float((occ >= 10).mean()))
        print(f"[oodwhy] pool {name:13s} gc {gc.mean():.3f} cpg {cpg.mean():.4f} label mean {lab.mean():.2f} frac>2 {(lab > 2).mean():.4f} frac>3 {(lab > 3).mean():.4f} max {lab.max():.1f} | designed (gc,cpg) cells covered {res[name]['designed_gc_cpg_support']:.2f}", flush=True)
    print(f"[oodwhy] designed: gc {ogc.mean():.3f} cpg {ocpg.mean():.4f} label mean {olab.mean():.2f} q5-95 {np.percentile(olab, [5, 95]).round(2).tolist()}", flush=True)
    # student behaviour on the designed set: ensembles of 3 independent-pool menus per training reservoir
    for name, st in (("genomic", "genomic"), ("random", "random"), ("motif", "motif_ct_enriched"), ("evoaug", "evoaug_paper2023")):
        d = a.menu_matrix([f"{MR_}/ind30k_{st}_r{r}" for r in (1, 2, 3)])
        if d is None:
            continue
        yt = d["yt"]["ood"]
        p = enet_pred(d, "ood")
        q75 = yt >= np.percentile(yt, 75)
        slope = float(np.polyfit(p, yt, 1)[0])
        res.setdefault("student", {})[name] = dict(r=mj.pear(p, yt), r_top_quartile=mj.pear(p[q75], yt[q75]), r_bottom_three_quarters=mj.pear(p[~q75], yt[~q75]), mean_pred=float(p.mean()), mean_label=float(yt.mean()),
                                                  slope_label_on_pred=slope, frac_pred_above_2=float((p > 2).mean()), frac_label_above_2=float((yt > 2).mean()), n_models=int(d["V"].shape[1]))
        s = res["student"][name]
        print(f"[oodwhy] student trained on {name:7s}: r {s['r']:.3f} (top quartile {s['r_top_quartile']:.3f}, rest {s['r_bottom_three_quarters']:.3f}); mean pred {s['mean_pred']:.2f} vs label {s['mean_label']:.2f}; slope {s['slope_label_on_pred']:.2f}; frac pred>2 {s['frac_pred_above_2']:.2f} (label {s['frac_label_above_2']:.2f})", flush=True)
    DATA["oodwhy"] = res
    # figure: label distributions of the pools against the designed set, and composition
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4))
    bins = np.linspace(-1.5, 8, 60)
    ax = axs[0]
    for name, col in (("genomic", GREEN), ("random", ORANGE), ("motif", PURPLE)):
        z = np.load(f"{reps}/{pools[name]}_r1_d30000.npz", allow_pickle=True)
        ax.hist(z["oracle_labels"], bins=bins, histtype="step", color=col, lw=1.4, density=True, label=f"{name} pool (30k)")
    ax.hist(olab, bins=bins, histtype="step", color=RED, lw=1.8, density=True, label="designed set")
    ax.set_yscale("log"); ax.set_xlabel("oracle activity"); ax.set_title("Activity: pools against the designed set"); ax.legend(fontsize=7.5)
    ax = axs[1]
    for name, col in (("genomic", GREEN), ("random", ORANGE), ("motif", PURPLE)):
        z = np.load(f"{reps}/{pools[name]}_r1_d30000.npz", allow_pickle=True)
        gc, cpg = seq_stats(list(z["sequences"])[:5000])
        ax.scatter(gc, cpg, s=3, alpha=0.25, color=col, label=name)
    ax.scatter(ogc[:5000], ocpg[:5000], s=3, alpha=0.25, color=RED, label="designed")
    ax.set_xlabel("GC fraction"); ax.set_ylabel("CpG per dinucleotide"); ax.set_title("Composition"); ax.legend(fontsize=7.5, markerscale=3)
    ax = axs[2]
    stu = res.get("student", {})
    names = list(stu)
    ax.bar(np.arange(len(names)) - 0.2, [stu[n]["r"] for n in names], width=0.4, color=BLUE, label="all designed")
    ax.bar(np.arange(len(names)) + 0.2, [stu[n]["r_top_quartile"] for n in names], width=0.4, color=ORANGE, label="top quartile only")
    ax.set_xticks(range(len(names))); ax.set_xticklabels([f"trained on\n{n}" for n in names], fontsize=8); ax.set_ylabel("r on the designed set"); ax.set_title("Student accuracy on designed sequences"); ax.legend(fontsize=7.5)
    a.savefig(fig, "ood_why_random", out)


# ------------------------------------------------------------------------------------------------------------------ 3. extrapolation
def fit_power(D, err):
    x, y = np.log(D), np.log(err)
    A = np.vstack([x, np.ones_like(x)]).T
    (m, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    return -m, c  # err = exp(c) * D^-b


def an_extrap(out, n_boot=2000, noise=0.004, seed=0):
    rng = np.random.default_rng(seed)
    d = json.load(open("outputs/justification/data.json"))["menu_scaling"]
    res = {}
    targets = (1_000_000, 4_000_000)
    for rname, curve in d.items():
        Ds = sorted(int(k) * 1000 for k in curve)
        if len(Ds) < 3:
            continue
        res[rname] = {}
        for e in EV:
            r = np.array([curve[str(D // 1000)][e] for D in Ds])
            err = 1 - r
            b, c = fit_power(np.array(Ds, float), err)
            draws = []
            for _ in range(n_boot):
                rr = r + rng.normal(0, noise, len(r))
                bb, cc = fit_power(np.array(Ds, float), 1 - rr)
                draws.append([1 - np.exp(cc) * t ** (-bb) for t in targets] + [bb])
            draws = np.array(draws)
            # sensitivity: a floor on the error (the student cannot beat the oracle's own noise)
            floors = {}
            for fl in (0.01, 0.03):
                ok = err - fl > 1e-4
                if ok.sum() >= 3:
                    bf, cf = fit_power(np.array(Ds, float)[ok], err[ok] - fl)
                    floors[str(fl)] = [float(1 - (fl + np.exp(cf) * t ** (-bf))) for t in targets]
            res[rname][e] = dict(sizes=Ds, r=r.tolist(), exponent=float(b), pred={str(t): ci(draws[:, i]) for i, t in enumerate(targets)}, exponent_ci=ci(draws[:, 2]), floor_sensitivity=floors,
                                 gain_per_doubling_now=float((1 - np.exp(c) * (Ds[-1] * 2) ** (-b)) - r[-1]))
            p = res[rname][e]["pred"]
            print(f"[extrap] {rname:18s} {e:10s} b={b:.3f} r@{Ds[-1] // 1000}k={r[-1]:.3f} -> 1M {p[str(targets[0])][0]:.3f} [{p[str(targets[0])][1]:.3f},{p[str(targets[0])][2]:.3f}]  4M {p[str(targets[1])][0]:.3f} [{p[str(targets[1])][1]:.3f},{p[str(targets[1])][2]:.3f}]  floors {floors}", flush=True)
    DATA["extrap"] = res
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6))
    cmap = {"genomic": GREEN, "random": ORANGE, "motif_ct_enriched": PURPLE, "evo2": TEAL, "evoaug_paper2023": GREY}
    for ax, e in zip(axs, EV):
        for rname, row in res.items():
            if e not in row:
                continue
            Ds = np.array(row[e]["sizes"], float)
            err = 1 - np.array(row[e]["r"])
            b, c = fit_power(Ds, err)
            xs = np.logspace(np.log10(Ds[0]), np.log10(4e6), 40)
            ax.plot(Ds, err, "o", color=cmap.get(rname, GREY), ms=4)
            ax.plot(xs, np.exp(c) * xs ** (-b), "-", color=cmap.get(rname, GREY), lw=1.1, alpha=0.8, label=rname.replace("_", " "))
        ax.set_xscale("log"); ax.set_yscale("log"); ax.axvline(1e6, color=GREY, ls=":", lw=1); ax.axvline(3e5, color=GREY, ls=":", lw=1)
        ax.set_title({"genomic": "genomic test", "ood": "designed set (OOD)", "random_32k": "random sequences"}[e]); ax.set_xlabel("training sequences"); ax.set_ylabel("1 - r")
    axs[0].legend(fontsize=7.2)
    fig.suptitle("Power-law fits of the menu ensembles (10k to 300k) extended to 4M; dotted lines at 300k and 1M", fontsize=10.5, y=1.02)
    a.savefig(fig, "extrapolation", out)


def ci(v):
    v = np.asarray(v)
    return [float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


# ------------------------------------------------------------------------------------------- 4. reference LegNet against the searched entries
def per_entry_test(d):
    """menu retrain dir -> {entry: [genomic test r per retraining]}"""
    lab = np.load(f"{d}/labels.npz")
    yt = lab["oracle_genomic"].astype(float)
    out = {}
    for f in glob.glob(f"{d}/r*_menu_*_meta.json"):
        m = json.load(open(f))
        z = np.load(f.replace("_meta.json", ".npz"))
        tp = z["test_pred_genomic"].astype(float)
        if not np.isfinite(tp).all() or np.abs(tp).max() > 1e3:
            continue
        out.setdefault(m["menu_entry"], []).append(mj.pear(tp, yt))
    return out


def an_legref(out):
    rows = []
    for d in sorted(glob.glob(f"{MR_}/*")):
        b = os.path.basename(d)
        if not re.match(r"^(g\d+k|f\d+k_|ind30k_|un\d+k_|mix30k_|as30k_|nest_)", b) or "seeds12" in b or not os.path.exists(f"{d}/labels.npz"):
            continue
        pe = per_entry_test(d)
        if "reference_legnet_eff" not in pe or len(pe) < 25:
            continue
        m = {k: float(np.mean(v)) for k, v in pe.items()}
        ref = m.pop("reference_legnet_eff")
        vals = np.array(sorted(m.values(), reverse=True))
        rank = int(1 + (vals > ref).sum())
        size = re.search(r"(\d+)k", b)
        rows.append(dict(dir=b, size_k=int(size.group(1)) if size else -1, n_entries=len(vals) + 1, ref=ref, rank=rank, median=float(np.median(vals)), top5=float(vals[:5].mean()), best=float(vals[0]), n_retrainings=len(pe["reference_legnet_eff"])))
    DATA["legref"] = rows
    by = {}
    for r in rows:
        by.setdefault(r["size_k"], []).append(r)
    summ = {}
    for sk, rs in sorted(by.items()):
        summ[str(sk)] = dict(n_pools=len(rs), mean_rank=float(np.mean([r["rank"] for r in rs])), mean_n=float(np.mean([r["n_entries"] for r in rs])), ref_minus_median=float(np.mean([r["ref"] - r["median"] for r in rs])),
                             best_minus_ref=float(np.mean([r["best"] - r["ref"] for r in rs])), top5_minus_ref=float(np.mean([r["top5"] - r["ref"] for r in rs])), frac_ref_top_quartile=float(np.mean([r["rank"] <= r["n_entries"] / 4 for r in rs])))
        s = summ[str(sk)]
        print(f"[legref] {sk:>4d}k: {s['n_pools']} pools, reference rank {s['mean_rank']:.1f} of {s['mean_n']:.0f}; reference minus median {s['ref_minus_median']:+.4f}; top-5 mean minus reference {s['top5_minus_ref']:+.4f}; best minus reference {s['best_minus_ref']:+.4f}; in top quartile on {s['frac_ref_top_quartile']:.0%}", flush=True)
    DATA["legref_summary"] = summ
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for r in rows:
        ax.scatter(r["size_k"] * (1 + np.random.default_rng(len(r["dir"])).uniform(-0.04, 0.04)), r["best"] - r["ref"], s=14, color=BLUE, alpha=0.6)
        ax.scatter(r["size_k"] * (1 + np.random.default_rng(len(r["dir"]) + 1).uniform(-0.04, 0.04)), r["median"] - r["ref"], s=14, color=ORANGE, alpha=0.6)
    ax.scatter([], [], color=BLUE, label="best searched entry minus reference"); ax.scatter([], [], color=ORANGE, label="median searched entry minus reference")
    ax.axhline(0, color=GREY, lw=1); ax.axhspan(-0.005, 0.005, color=GREY, alpha=0.15, lw=0)
    ax.set_xscale("log"); ax.set_xlabel("training sequences (k)"); ax.set_ylabel("test r difference (genomic test)"); ax.legend(fontsize=7.8)
    ax.set_title("Tuned LegNet against the searched architectures (every retrained pool)")
    a.savefig(fig, "legnet_reference", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    ap.add_argument("--only", default="runs2,oodwhy,extrap,legref")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    prev = json.load(open(f"{args.out}/data.json")) if os.path.exists(f"{args.out}/data.json") else {}
    for step, fn in (("legref", an_legref), ("extrap", an_extrap), ("oodwhy", an_oodwhy), ("runs2", an_runs2)):
        if step in args.only.split(","):
            try:
                fn(args.out)
            except Exception as ex:
                import traceback
                traceback.print_exc()
                print(f"[{step}] FAILED: {ex}", flush=True)
            prev.update(DATA)
            json.dump(prev, open(f"{args.out}/data.json", "w"), indent=1, default=float)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
