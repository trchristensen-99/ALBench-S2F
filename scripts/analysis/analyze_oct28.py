# ruff: noqa
#!/usr/bin/env python
"""Oct-28 analyses on existing data: does model size matter at larger dataset sizes (params), and what the configuration sets and trained ensembles look like (overview).

    python scripts/analysis/analyze_oct28.py --only params,overview [--out outputs/justification_oct22]

Run on a compute node (cpuq), not on the login node.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

DATA: dict = {}
MR = "outputs/menu_retrain_v0"
O22 = "outputs/archsearch_reps_oct22"

FAMILIES = [  # (family, regex on the proposer's one-line idea; first match wins)
    ("U-Net / multi-resolution", r"u-?net|multi-?resolution|hrnet|pyramid|multi-?scale"),
    (
        "motif-count / pooled-readout",
        r"bag-of-motifs|motif-count|position-?binned|soft-?count|pairwise|second-order|spacing|co-?occurrence|readout",
    ),
    ("dilated / WaveNet", r"dilat|wavenet"),
    ("DenseNet / ResNeXt / Inception", r"densenet|resnext|inception|selective-?kernel|grouped"),
    ("recurrent / looped / weight-tied", r"recurrent|looped|weight-?tied|lstm|gru"),
    ("attention / transformer", r"transformer|attention"),
    ("long-conv / state-space (Hyena, S4)", r"hyena|s4d|long-?conv|state-?space"),
    ("LegNet-style SE tower", r"legnet|squeeze|\bse\b|inverted-?residual|efficient"),
]


def family(idea: str) -> str:
    s = (idea or "").lower()
    for name, pat in FAMILIES:
        if re.search(pat, s):
            return name
    return "other conv tower"


def source_features(src: str) -> dict:
    s = src or ""
    ks = [int(k) for k in re.findall(r"Conv1d\([^)]*?kernel_size\s*=\s*(\d+)", s)] + [
        int(k) for k in re.findall(r"Conv1d\(\s*\w+\s*,\s*\w+\s*,\s*(\d+)", s)
    ]
    return {
        "attention": bool(
            re.search(r"MultiheadAttention|scaled_dot_product|TransformerEncoder", s)
        ),
        "recurrent": bool(re.search(r"nn\.(LSTM|GRU|RNN)", s)),
        "dilated": bool(re.search(r"dilation\s*=\s*[2-9]", s)),
        "depthwise": bool(re.search(r"groups\s*=", s)),
        "upsample": bool(re.search(r"Upsample|ConvTranspose|interpolate", s)),
        "n_conv_layers": len(re.findall(r"Conv1d\(", s)),
        "max_kernel": max(ks) if ks else None,
        "median_kernel": float(np.median(ks)) if ks else None,
        "source_chars": len(s),
    }


def metas(dirs, rounds=(0,)):
    out = []
    for d in dirs:
        for f in glob.glob(f"{d}/r*_*_meta.json"):
            try:
                m = json.load(open(f))
            except Exception:
                continue
            if (
                "n_params" in m
                and m.get("val_pearson") is not None
                and int(m.get("round", 0)) in rounds
                and m.get("n_params")
            ):
                out.append(m)
    return out


def per_doubling_slope(logp, y):
    if len(y) < 6 or np.ptp(logp) == 0:
        return None
    return float(np.polyfit(logp, y, 1)[0])


def an_params():
    """Does a larger model do better, and does that change with the dataset size? Two views: the retrained menu (the same 33 configurations at every size) and the search runs (models proposed at 30k, 100k and 300k)."""
    res = {"menu": {}, "search": {}}
    dirs = {
        "genomic": {
            10000: "nest_genomic_r1_d10k",
            30000: "ind30k_genomic_r1",
            100000: "nest_genomic_r1_d100k",
            300000: "nest_genomic_r1_d300k",
        },
        "random": {
            10000: "lad_random_r1_d10k",
            30000: "ind30k_random_r1",
            100000: "lad_random_r1_d100k",
            300000: "nest_random_r1_d300k",
            600000: "anchor_random_r1_d600k",
            1000000: "anchor_random_r1_d1000k",
        },
        "motif_ct_enriched": {
            10000: "lad_motif_ct_enriched_r1_d10k",
            30000: "ind30k_motif_ct_enriched_r1",
            100000: "lad_motif_ct_enriched_r1_d100k",
            300000: "nest_motif_ct_enriched_r1_d300k",
            600000: "anchor_motif_ct_enriched_r1_d600k",
            1000000: "anchor_motif_ct_enriched_r1_d1000k",
        },
    }
    for st, by in dirs.items():
        for D, d in by.items():
            ms = metas([f"{MR}/{d}"])
            ms = [
                m
                for m in ms
                if m.get("test_oracle_pearson") is not None and m["test_oracle_pearson"] > 0.2
            ]
            if len(ms) < 12:
                continue
            lp = np.log2([m["n_params"] for m in ms])
            r = np.array([m["test_oracle_pearson"] for m in ms])
            rho, p = stats.spearmanr(lp, r)
            order = np.argsort(lp)
            t = len(ms) // 3
            res["menu"][f"{st}|{D}"] = {
                "n": len(ms),
                "params_min": int(2 ** lp.min()),
                "params_max": int(2 ** lp.max()),
                "params_median": int(2 ** np.median(lp)),
                "spearman": float(rho),
                "p": float(p),
                "slope_per_doubling": per_doubling_slope(lp, r),
                "mean_r_smallest_third": float(r[order[:t]].mean()),
                "mean_r_largest_third": float(r[order[-t:]].mean()),
            }
            print(
                f"[params] menu {st} {D // 1000}k (n={len(ms)}, {2 ** lp.min() / 1e6:.2f}M to {2 ** lp.max() / 1e6:.2f}M params): Spearman {rho:+.2f} (p {p:.2f}); per doubling {res['menu'][f'{st}|{D}']['slope_per_doubling']:+.4f}; smallest third {r[order[:t]].mean():.4f}, largest third {r[order[-t:]].mean():.4f}",
                flush=True,
            )
    sets = {
        "genomic 30k (15 runs)": [f"{O22}/genomic_30k/rep{r}" for r in range(1, 16)],
        "random 30k (15 runs)": [f"{O22}/random_30k/rep{r}" for r in range(1, 16)],
        "motif 30k (15 runs)": [f"{O22}/motif_ct_enriched_30k/rep{r}" for r in range(1, 16)],
        "genomic 100k (3 runs)": [
            f"outputs/archsearch_reps100k_oct21/genomic/rep{r}" for r in (1, 2, 3)
        ],
        "genomic 300k (3 runs)": [f"{O22}/genomic_300k/rep{r}" for r in (1, 2, 3)],
    }
    for label, ds in sets.items():
        ys, xs, runs = [], [], []
        for i, d in enumerate(ds):
            ms = [m for m in metas([d], rounds=range(0, 40)) if m["val_pearson"] > 0.3]
            if len(ms) < 6:
                continue
            v = np.array([m["val_pearson"] for m in ms])
            ys += list(
                v - v.mean()
            )  # within-run, so that run-to-run level differences do not masquerade as a size effect
            xs += list(np.log2([m["n_params"] for m in ms]))
            runs += [i] * len(ms)
        if len(ys) < 20:
            continue
        xs, ys = np.array(xs), np.array(ys)
        rho, p = stats.spearmanr(xs, ys)
        res["search"][label] = {
            "n_models": len(ys),
            "n_runs": len(set(runs)),
            "params_median": int(2 ** np.median(xs)),
            "params_p90": int(2 ** np.percentile(xs, 90)),
            "params_max": int(2 ** xs.max()),
            "spearman_within_run": float(rho),
            "p": float(p),
            "slope_per_doubling": per_doubling_slope(xs, ys),
        }
        print(
            f"[params] search {label}: {len(ys)} models; median {2 ** np.median(xs) / 1e6:.2f}M, 90th percentile {2 ** np.percentile(xs, 90) / 1e6:.2f}M, max {2 ** xs.max() / 1e6:.2f}M; within-run Spearman {rho:+.2f} (p {p:.3f}); validation r per doubling of parameters {per_doubling_slope(xs, ys):+.4f}",
            flush=True,
        )
    DATA["params"] = res


def an_overview():
    """Architectural and hyper-parameter features of the shipped configuration sets, and of the trained genomic ensembles found in the shared folder."""
    sys.path.insert(0, os.getcwd())
    from albench.ensemble.menu import load_menu

    menu = load_menu("v0")
    skipped = menu.skipped
    rows = {}
    for e in menu.entries:
        tc = e.train_config
        rows[e.id] = {
            "family": family(e.idea),
            "run": e.run,
            "search_val": e.search_val,
            "n_params": e.n_params,
            "skipped": e.id in skipped,
            "idea": e.idea[:120],
            **{
                k: tc.get(k)
                for k in ("optimizer", "lr", "weight_decay", "batch_size", "epochs", "lr_schedule")
            },
            **source_features(e.source),
        }
    usable = {k: v for k, v in rows.items() if not v["skipped"]}
    res = {"configs": rows}
    res["summary"] = {
        "n_usable": len(usable),
        "families": dict(Counter(v["family"] for v in usable.values())),
        "optimizer": dict(Counter(v["optimizer"] for v in usable.values())),
        "lr": dict(Counter(v["lr"] for v in usable.values())),
        "epochs": dict(Counter(v["epochs"] for v in usable.values())),
        "batch_size": dict(Counter(v["batch_size"] for v in usable.values())),
        "schedule": dict(Counter(v["lr_schedule"] for v in usable.values())),
        "params_quantiles": {
            q: int(np.nanpercentile([v["n_params"] or np.nan for v in usable.values()], q))
            for q in (0, 25, 50, 75, 100)
        },
        "attention": sum(v["attention"] for v in usable.values()),
        "recurrent": sum(v["recurrent"] for v in usable.values()),
        "dilated": sum(v["dilated"] for v in usable.values()),
        "upsample_unet": sum(v["upsample"] for v in usable.values()),
        "weight_decay": dict(Counter(v["weight_decay"] for v in usable.values())),
    }
    sizes = {}
    for size in (10000, 30000, 100000):
        ents = [e.id for e in menu.for_size(size)]
        sizes[str(size)] = {
            "k": len(ents),
            "families": dict(Counter(rows[i]["family"] for i in ents)),
            "median_params": int(np.median([rows[i]["n_params"] or 0 for i in ents])),
            "ids": ents,
        }
    res["per_size"] = sizes
    ens = {}
    for d in sorted(glob.glob("/grid/koo/home/shared/albench/ensembles/*/ensemble.json")):
        meta = json.load(open(d))
        name = os.path.basename(os.path.dirname(d))
        w = defaultdict(float)
        seeds_kept = defaultdict(int)
        for r in meta["members"]:
            w[r["entry"]] += r["weight"]
            if r["weight"] > 0:
                seeds_kept[r["entry"]] += 1
        tot = sum(w.values()) or 1.0
        fam_w = defaultdict(float)
        opt_w = defaultdict(float)
        par_w = []
        for e, x in w.items():
            fam_w[rows[e]["family"]] += x / tot
            opt_w[rows[e]["optimizer"]] += x / tot
            par_w.append((rows[e]["n_params"] or 0, x / tot))
        kept = [r for r in meta["members"] if r["weight"] > 0]
        pw = np.array(par_w)
        ens[name] = {
            "members": len(meta["members"]),
            "members_positive_weight": len(kept),
            "configs_with_weight": sum(1 for x in w.values() if x > 0),
            "weight_by_family": dict(sorted(fam_w.items(), key=lambda kv: -kv[1])),
            "weight_by_optimizer": dict(opt_w),
            "weight_median_params": float(
                pw[np.argsort(pw[:, 0])][
                    np.searchsorted(np.cumsum(pw[np.argsort(pw[:, 0])][:, 1]), 0.5), 0
                ]
            )
            if len(pw)
            else None,
            "top_configs": [
                (e, round(x / tot, 3), rows[e]["family"], rows[e]["n_params"], seeds_kept[e])
                for e, x in sorted(w.items(), key=lambda kv: -kv[1])[:8]
            ],
            "val_r_ensemble": meta.get("val_r_ensemble"),
            "mean_member_val_r": float(np.mean([r["val_r"] for r in meta["members"]])),
            "best_member_val_r": float(np.max([r["val_r"] for r in meta["members"]])),
            "patience": meta.get("patience"),
        }
        print(
            f"[overview] ensemble {name}: {len(meta['members'])} members, {len(kept)} with positive weight over {ens[name]['configs_with_weight']} configurations; weight by family "
            + ", ".join(f"{k} {v:.0%}" for k, v in list(ens[name]["weight_by_family"].items())[:5]),
            flush=True,
        )
    res["ensembles"] = ens
    print(
        f"[overview] {res['summary']['n_usable']} usable configurations: families {res['summary']['families']}; optimizers {res['summary']['optimizer']}; parameters {res['summary']['params_quantiles']}",
        flush=True,
    )
    DATA["overview"] = res


def an_runsrc(n_draw=40, seed=0):
    """Does it matter which reservoirs the architecture searches ran on, and how many runs are enough? Candidates: the top 3 models of every one of the 45 search runs of the 30k program
    (15 on genomic, 15 on random, 15 on motif CT-enriched pools; menus MG, MR, MM), retrained on each target pool (genomic, random, motif, GRM and the two reservoirs no search used:
    evoaug, evo2). For a target, compositions with the same total number of runs are compared by the non-negative stack's test r: all runs from one reservoir against a balanced draw
    across the three, and the run-count curve beyond 15."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import analyze_oct26 as b26

    rng = np.random.default_rng(seed)
    menus = {c: json.load(open(f"outputs/menus_oct22/{c}.json")) for c in ("MG", "MR", "MM")}
    resv = {"MG": "genomic", "MR": "random", "MM": "motif"}
    res = {}
    for t in ("genomic", "random", "motif", "grm", "evoaug", "evo2"):
        cols_V, cols_T, run_of, res_of = [], {e: [] for e in b26.EV}, [], []
        yv = yt = None
        for cond, rname in resv.items():
            c = b26.load([f"as30k_{cond}_{t}"], rounds=[0])
            if c is None:
                continue
            yv, yt = c["yv"], c["yt"]
            for j, idx in enumerate(c["idx"]):
                cols_V.append(c["V"][:, j])
                for e in b26.EV:
                    cols_T[e].append(c["T"][e][:, j])
                run_of.append((cond, menus[cond][int(idx)]["cell"]))
                res_of.append(rname)
        if len(run_of) < 60:
            print(
                f"[runsrc] {t}: only {len(run_of)} candidates retrained so far; skipped", flush=True
            )
            continue
        V, T = np.column_stack(cols_V), {e: np.column_stack(cols_T[e]) for e in b26.EV}
        runs = sorted(set(run_of))
        by_res = {r: [x for x in runs if resv[x[0]] == r] for r in ("genomic", "random", "motif")}

        def score(sel):
            cols = [j for j, r in enumerate(run_of) if r in sel]
            if len(cols) < 2:
                return None
            q = b26.a.nnls_ens(V, yv, T, yt, cols)
            return [q[e] for e in b26.EV]

        out = {"n_candidates": len(run_of), "n_runs": len(runs), "single": {}, "balanced": {}}
        for r, rs in by_res.items():
            for n in (3, 5, 8, 10, 15):
                if n <= len(rs):
                    v = np.array(
                        [
                            score({rs[i] for i in rng.choice(len(rs), n, replace=False)})
                            for _ in range(n_draw if n < len(rs) else 1)
                        ],
                        float,
                    )
                    out["single"][f"{r}|{n}"] = {
                        "mean": v.mean(0).tolist(),
                        "sd": v.std(0).tolist(),
                        "n_runs": n,
                    }
        for n in (3, 6, 9, 15, 30, 45):
            per = n // 3
            if n <= min(len(rs) for rs in by_res.values()) * 3:
                v = np.array(
                    [
                        score(
                            {
                                x
                                for rs in by_res.values()
                                for x in [rs[i] for i in rng.choice(len(rs), per, replace=False)]
                            }
                        )
                        for _ in range(n_draw if n < 45 else 1)
                    ],
                    float,
                )
                out["balanced"][str(n)] = {
                    "mean": v.mean(0).tolist(),
                    "sd": v.std(0).tolist(),
                    "n_runs": n,
                }
        res[t] = out
        gn = out["single"].get("genomic|15", {}).get("mean", [np.nan])[0]
        print(
            f"[runsrc] target {t} ({len(runs)} runs, {len(run_of)} candidates): genomic-test r with 15 runs from genomic {gn:.4f}, random {out['single'].get('random|15', {}).get('mean', [np.nan])[0]:.4f}, motif {out['single'].get('motif|15', {}).get('mean', [np.nan])[0]:.4f}, balanced 15 {out['balanced'].get('15', {}).get('mean', [np.nan])[0]:.4f}, all 45 {out['balanced'].get('45', {}).get('mean', [np.nan])[0]:.4f}",
            flush=True,
        )
    DATA["runsrc"] = res


def an_v1sim(seed=0):
    """Simulate the proposed unified recipe on existing retrained models (30k, one pool): 3 runs from each of 5 reservoir families (genomic, random, motif, evoaug, evo2) plus 3 runs on the
    mixed corpus = 18 runs; the top configuration by search validation r of each run, plus the 2 best remaining by search validation r = 20 configurations, all kept in a weighted ensemble
    (positive ElasticNetCV on z-scored validation predictions). Compared with the same weighting over: the 45 candidates of each source, the best 20 of the genomic-only 15-run menu, a random 20
    of the union, and the 30 usable configurations of the current menu."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import analyze_oct26 as b26

    rng = np.random.default_rng(seed)
    mf, mx, mg = (json.load(open(f"outputs/menus_oct22/{c}.json")) for c in ("MF", "MX", "MG"))
    fam_runs = sorted({e["cell"] for e in mf})
    grm_runs = [f"GRM_30k/rep{r}" for r in (1, 2, 3)]
    res = {}
    for t in ("genomic", "random", "motif", "grm"):
        cf, cx, cg = (b26.load([f"as30k_{c}_{t}"], rounds=[0]) for c in ("MF", "MX", "MG"))
        if cf is None or cx is None or cg is None:
            continue
        yv, yt = cf["yv"], cf["yt"]
        pool = []  # (source, idx, entry, column data)
        for src, c, menu in (("MF", cf, mf), ("MX", cx, mx)):
            for j, idx in enumerate(c["idx"]):
                e = menu[int(idx)]
                if src == "MX" and e["cell"] not in grm_runs:
                    continue
                pool.append((src, c, j, e))
        by_run: dict = {}
        for it in pool:
            by_run.setdefault(it[3]["cell"], []).append(it)
        chosen, rest = [], []
        for run, items in by_run.items():
            items.sort(key=lambda it: -(it[3]["orig_val"] or -1))
            chosen.append(items[0])
            rest += items[1:]
        rest.sort(key=lambda it: -(it[3]["orig_val"] or -1))
        v1 = chosen + rest[:2]

        def stack(items):
            V = np.column_stack([c["V"][:, j] for _, c, j, _ in items])
            T = {e: np.column_stack([c["T"][e][:, j] for _, c, j, _ in items]) for e in b26.EV}
            r, kept = b26.a.mj.enet(V, yv, T, yt, range(V.shape[1]))
            return {e: float(r[e]) for e in b26.EV}, int(kept)

        def stack_all(c):
            r, kept = b26.a.mj.enet(c["V"], yv, c["T"], yt, range(c["V"].shape[1]))
            return {e: float(r[e]) for e in b26.EV}, int(kept)

        out = {
            "n_runs": len(by_run),
            "v1": stack(v1),
            "v1_composition": dict(
                sorted(
                    {
                        k: sum(1 for it in v1 if it[3]["cell"].split("/")[0] == k)
                        for k in {it[3]["cell"].split("/")[0] for it in v1}
                    }.items()
                )
            ),
        }
        out["MF_all_45"], out["MX_all_45"], out["MG_all_45"] = (
            stack_all(cf),
            stack_all(cx),
            stack_all(cg),
        )
        mg_items = sorted(
            range(len(cg["idx"])), key=lambda j: -(mg[int(cg["idx"][j])]["orig_val"] or -1)
        )[:20]
        r, kept = b26.a.mj.enet(cg["V"], yv, cg["T"], yt, mg_items)
        out["MG_best20_by_search_val"] = ({e: float(r[e]) for e in b26.EV}, int(kept))
        draws = []
        for _ in range(10):
            sel = [pool[i] for i in rng.choice(len(pool), 20, replace=False)]
            draws.append(stack(sel)[0])
        out["random20_of_family_pool"] = {e: float(np.mean([d[e] for d in draws])) for e in b26.EV}
        res[t] = out
        f = lambda x: (
            f"{x[0]['genomic']:.4f}/{x[0]['ood']:.3f}/{x[0]['random_32k']:.4f} ({x[1]} kept)"
        )  # noqa: E731
        print(
            f"[v1sim] target {t}: v1 (20 configs, {out['v1_composition']}) {f(out['v1'])}; MF all 45 {f(out['MF_all_45'])}; MX all {f(out['MX_all_45'])}; MG all 45 {f(out['MG_all_45'])}; MG best 20 {f(out['MG_best20_by_search_val'])}; random 20 of the pool {out['random20_of_family_pool']['genomic']:.4f}/{out['random20_of_family_pool']['ood']:.3f}/{out['random20_of_family_pool']['random_32k']:.4f}",
            flush=True,
        )
    DATA["v1sim"] = res


def an_dupsim(seed=0):
    """Is it better to spend extra training runs on duplicate seeds of the best configurations or on more distinct configurations? Nine seeds of every configuration of the current menu exist at 30k
    (genomic, random, motif; pool r1). With the models ranked by their own validation r: 20 distinct configurations; 24 distinct; and 20 distinct plus 2 more seeds of the best configuration and
    1 more of the second and third (24 models); all with the same weighting."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import analyze_oct26 as b26
    from albench.ensemble.menu import load_menu

    rng = np.random.default_rng(seed)
    skipped = set(load_menu("v0").skipped)
    menu = load_menu("v0")
    skip_idx = {menu.index_of(i) for i in skipped}
    res = {}
    for st in ("genomic", "random", "motif_ct_enriched"):
        c = b26.load([f"ind30k_{st}_r1", f"ind30k_{st}_r1_seeds3to8"])
        if c is None:
            continue
        val = np.array([b26.a.mj.pear(c["V"][:, j], c["yv"]) for j in range(c["V"].shape[1])])
        seeds = sorted(set(int(x) for x in c["rnd"]))
        by_cfg: dict = {}
        for j in range(len(val)):
            if int(c["idx"][j]) not in skip_idx:
                by_cfg.setdefault(int(c["idx"][j]), []).append(j)
        base = {
            k: min(v, key=lambda j: c["rnd"][j])
            for k, v in by_cfg.items()
            if any(c["rnd"][j] == 0 for j in v)
        }  # seed 0 column of each configuration
        rank = sorted(base, key=lambda k: -val[base[k]])

        def stack(cols):
            r, kept = b26.a.mj.enet(c["V"], c["yv"], c["T"], c["yt"], cols)
            return {e: float(r[e]) for e in b26.EV}, int(kept), len(cols)

        distinct20 = [base[k] for k in rank[:20]]
        distinct24 = [base[k] for k in rank[:24]]
        out = {
            "distinct_20": stack(distinct20),
            "distinct_24": stack(distinct24),
            "distinct_30": stack([base[k] for k in rank]),
        }
        dup_runs = {}
        for variant, extra in (
            ("top1_x2_more_top2_top3_x1_more", {rank[0]: 2, rank[1]: 1, rank[2]: 1}),
            ("top3_x3_more_each", {rank[0]: 3, rank[1]: 3, rank[2]: 3}),
        ):
            cols = list(distinct20)
            draws = []
            for _ in range(8):
                add = []
                for k, n in extra.items():
                    others = [j for j in by_cfg[k] if j != base[k]]
                    add += list(rng.choice(others, min(n, len(others)), replace=False))
                draws.append(stack(cols + add))
            dup_runs[variant] = (
                {e: float(np.mean([d[0][e] for d in draws])) for e in b26.EV},
                float(np.mean([d[1] for d in draws])),
                draws[0][2],
            )
        out["duplicates"] = dup_runs
        res[st] = out
        f = lambda x: (
            f"{x[0]['genomic']:.4f}/{x[0]['ood']:.3f}/{x[0]['random_32k']:.4f} ({x[2]} models)"
        )  # noqa: E731
        print(
            f"[dupsim] {st} 30k: 20 distinct {f(out['distinct_20'])}; 24 distinct {f(out['distinct_24'])}; all 30 {f(out['distinct_30'])}; 20 + extra seeds of the best three (2,1,1) {f(dup_runs['top1_x2_more_top2_top3_x1_more'])}; (3,3,3) {f(dup_runs['top3_x3_more_each'])}",
            flush=True,
        )
    DATA["dupsim"] = res


STEPS = {
    "params": an_params,
    "overview": an_overview,
    "runsrc": an_runsrc,
    "v1sim": an_v1sim,
    "dupsim": an_dupsim,
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
