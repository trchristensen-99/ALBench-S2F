# ruff: noqa
#!/usr/bin/env python
"""Recipe, flank and search-size experiments of Oct 28 (cells hp34_, hp34s1_, fl34_, fl34s1_, lg34_, ds34_), all validated on the true V0c labels, all on the canonical genomic pool.

(1) Paired effect of each change on the 20 menu-v1 configurations at 30k, against hp34_base_30k (same configuration, same seed): mean difference of validation r and of test r on the three shared test sets,
    its standard error over configurations, the number of configurations that improve, and the SEED-NOISE FLOOR: the same paired difference between two training seeds of the unchanged recipe
    (hp34s1_base_30k minus hp34_base_30k). The flank arm has two seeds.
(2) Ensembles (positive ElasticNetCV stacks fitted on V0c): each arm alone; the base recipe over two seeds; all non-flank recipe arms pooled (what marginalising over the recipe buys); the best arm per architecture by
    validation r; the same with the flank arms; and at 10k and 100k the menu built from searches at that size against menu v1.

    python scripts/analysis/analyze_div34.py [--out outputs/justification_oct22]      (cpuq)
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

SH = "/grid/koo/home/shared/albench/data"
MR = "outputs/menu_retrain_v0"
SETS = ("genomic", "ood", "random_32k")
SKIP = {"son55_rep1_r00_arch_00", "son45_rep1_r11_arch_15", "son55_rep2_r15_arch_31"}


def pearson(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def load(d, seed):
    """{menu index: dict(val, test{set}, vr, tr{set}, tmeas)} for the models of one seed index (r00 / r01) of a retrain directory."""
    out = {}
    for mf in glob.glob(f"{MR}/{d}/r{seed:02d}_menu_*_meta.json"):
        i = int(os.path.basename(mf).split("_")[2])
        try:
            z = np.load(mf.replace("_meta.json", ".npz"))
            v = z["val_pred"].astype(np.float32)
            t = {s: z[f"test_pred_{s}"].astype(np.float32) for s in SETS}
        except Exception:
            continue
        if not np.isfinite(v).all() or any(
            not np.isfinite(x).all() or np.abs(x).max() > 1e3 for x in t.values()
        ):
            continue
        out[i] = {"val": v, "test": t}
    return out


def score(models):
    for m in models.values():
        m["vr"] = pearson(m["val"], VLAB)
        m["tr"] = {s: pearson(m["test"][s], CANON[s]) for s in SETS}
        m["tmeas"] = pearson(m["test"]["genomic"], MEAS)
    return models


def stack(models_list):
    ms = [m for m in models_list]
    V = np.stack([m["val"] for m in ms])
    st = fit_zscored_stack(V, VLAB)
    out = {"n": len(ms), "n_used": int((np.asarray(st.coef) > 1e-9).sum())}
    for s in SETS:
        out[s] = pearson(st.predict(np.stack([m["test"][s] for m in ms])), CANON[s])
    out["genomic_measured"] = pearson(
        st.predict(np.stack([m["test"]["genomic"] for m in ms])), MEAS
    )
    return out


def paired(a, b):
    c = sorted(set(a) & set(b))
    res = {"n": len(c)}
    for key, f in (
        ("val", lambda m: m["vr"]),
        *[(s, (lambda s: lambda m: m["tr"][s])(s)) for s in SETS],
    ):
        d = np.array([f(a[i]) - f(b[i]) for i in c])
        res[key] = {
            "mean": float(d.mean()),
            "se": float(d.std(ddof=1) / np.sqrt(len(d))),
            "sd": float(d.std(ddof=1)),
            "better": int((d > 0).sum()),
        }
    return res


def main():
    global VLAB, CANON, MEAS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    a = ap.parse_args()
    VLAB = np.load(f"{SH}/val_V0c.npz", allow_pickle=True)["oracle_labels"].astype(np.float64)
    CANON = {}
    for s in SETS:
        z = np.load(f"{SH}/test_sets/{s}_oracle.npz", allow_pickle=True)
        CANON[s] = z["oracle_mean"].astype(np.float64)
        if s == "genomic":
            MEAS = z["true_label"].astype(np.float64)
    arms = {
        "base": ("hp34_base_30k", 0),
        "base_s1": ("hp34s1_base_30k", 1),
        "lrx05": ("hp34_lrx05_30k", 0),
        "lrx2": ("hp34_lrx2_30k", 0),
        "bs128": ("hp34_bs128_30k", 0),
        "bs512": ("hp34_bs512_30k", 0),
        "wd1e3": ("hp34_wd1e3_30k", 0),
        "adamw": ("hp34_adamw_30k", 0),
        "lgrecipe": ("lg34_lgrecipe_30k", 0),
        "flank": ("fl34_flank_30k", 0),
        "flank_s1": ("fl34s1_flank_30k", 1),
        "lgrecipe_flank": ("lg34_lgrecipe_fl_30k", 0),
    }
    M = {k: score(load(d, s)) for k, (d, s) in arms.items()}
    print({k: len(v) for k, v in M.items()}, flush=True)
    out = {"n_models": {k: len(v) for k, v in M.items()}, "paired": {}, "ensembles": {}}
    out["mean_single"] = {
        k: {
            "val": float(np.mean([m["vr"] for m in v.values()])),
            **{s: float(np.mean([m["tr"][s] for m in v.values()])) for s in SETS},
        }
        for k, v in M.items()
    }
    for k in (
        "base_s1",
        "lrx05",
        "lrx2",
        "bs128",
        "bs512",
        "wd1e3",
        "adamw",
        "lgrecipe",
        "flank",
        "lgrecipe_flank",
    ):
        out["paired"][f"{k}_vs_base"] = paired(M[k], M["base"])
    out["paired"]["flank_s1_vs_base_s1"] = paired(M["flank_s1"], M["base_s1"])
    out["paired"]["lgrecipe_flank_vs_lgrecipe"] = paired(M["lgrecipe_flank"], M["lgrecipe"])
    # flank over both seeds: mean of the two paired differences per configuration
    c = sorted(set(M["flank"]) & set(M["base"]) & set(M["flank_s1"]) & set(M["base_s1"]))
    d = {
        s: np.array(
            [
                (
                    (M["flank"][i]["tr"][s] - M["base"][i]["tr"][s])
                    + (M["flank_s1"][i]["tr"][s] - M["base_s1"][i]["tr"][s])
                )
                / 2
                for i in c
            ]
        )
        for s in SETS
    }
    out["paired"]["flank_two_seeds"] = {
        "n": len(c),
        **{
            s: {
                "mean": float(d[s].mean()),
                "se": float(d[s].std(ddof=1) / np.sqrt(len(c))),
                "better": int((d[s] > 0).sum()),
            }
            for s in SETS
        },
    }
    for k, v in out["paired"].items():
        print(
            k,
            {
                kk: (round(vv["mean"], 4), round(vv.get("se", 0), 4), vv.get("better"))
                for kk, vv in v.items()
                if isinstance(vv, dict)
            },
            flush=True,
        )
    E = out["ensembles"]
    for k, v in M.items():
        if len(v) >= 10:
            E[k] = stack(list(v.values()))
    nonflank = ["base", "base_s1", "lrx05", "lrx2", "bs128", "bs512", "wd1e3", "adamw", "lgrecipe"]
    allm = [m for k in nonflank for m in M[k].values()]
    E["pooled_nonflank_recipes"] = stack(allm)
    E["base_two_seeds"] = stack(list(M["base"].values()) + list(M["base_s1"].values()))
    E["flank_two_seeds"] = stack(list(M["flank"].values()) + list(M["flank_s1"].values()))
    E["pooled_all_arms"] = stack([m for v in M.values() for m in v.values()])
    # best arm per architecture by V0c validation r (non-flank arms, seed 0 + recipe arms)
    best = []
    for i in sorted(M["base"]):
        cands = [M[k][i] for k in nonflank if i in M[k]]
        best.append(max(cands, key=lambda m: m["vr"]))
    E["best_recipe_per_architecture"] = stack(best)
    bestf = []
    for i in sorted(M["base"]):
        cands = [M[k][i] for k in ("flank", "flank_s1", "lgrecipe_flank") if i in M[k]]
        if cands:
            bestf.append(max(cands, key=lambda m: m["vr"]))
    E["best_flank_arm_per_architecture"] = stack(bestf)
    for name, ms in (
        ("top20_by_val_of_pooled_nonflank", sorted(allm, key=lambda m: -m["vr"])[:20]),
        ("top40_by_val_of_pooled_nonflank", sorted(allm, key=lambda m: -m["vr"])[:40]),
    ):
        E[name] = stack(ms)
    for k, v in E.items():
        print(
            "ensemble",
            k,
            {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()},
            flush=True,
        )
    # searches at the target size
    sz = {}
    for D, base, searched in (
        (10, "hp34_base_10k", "ds34_m10_10k"),
        (100, "hp34_base_100k", "ds34_m100_100k"),
    ):
        b, m = score(load(base, 0)), score(load(searched, 0))
        sz[f"{D}k"] = {
            "v1": stack(list(b.values())),
            "searched_at_size": stack(list(m.values())),
            "both": stack(list(b.values()) + list(m.values())),
            "mean_single_v1": {s: float(np.mean([x["tr"][s] for x in b.values()])) for s in SETS},
            "mean_single_searched": {
                s: float(np.mean([x["tr"][s] for x in m.values()])) for s in SETS
            },
        }
        print(
            D,
            "k",
            {
                k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)}
                for k, v in sz[f"{D}k"].items()
                if k in ("v1", "searched_at_size", "both")
            },
            flush=True,
        )
    out["size_search"] = sz
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/data_div34.json", "w"), indent=1, default=float)
    print("wrote", f"{a.out}/data_div34.json")


if __name__ == "__main__":
    main()
