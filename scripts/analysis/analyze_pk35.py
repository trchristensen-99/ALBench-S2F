# ruff: noqa
#!/usr/bin/env python
"""Oct-29 harness experiments on the canonical genomic 30k pool, V0c validation (menu v1, 20 configurations, one seed): what the student sees (real plasmid flank as input), weight averaging and BatchNorm
recalculation, the training recipe, and a LegNet-style reference under the same switches. Everything is paired with hp34_base_30k (same configuration, same seed index).

(1) Paired difference against the base for every arm: validation r and test r on the three shared test sets (oracle labels) and on the genomic test set against the experimental labels, its standard error over
    configurations, the number of configurations that improve. Context arms hold the configurations that can take wider input; the base is restricted to the same configurations. Seed-noise floor: hp34s1_base_30k minus hp34_base_30k.
(2) Every single model (arm, configuration, validation r, test r on each set) for the validation-against-test scatter.
(3) Ensembles (positive ElasticNetCV stacks fitted on V0c): each arm alone against the base on the same configurations, and everything pooled.
(4) LegNet-style reference (lg35_, lg35c_): per arm, the mean single-model test r over (3 recipes x 3 seeds), the paired difference against lg35_base, and the 9-model stack.
(5) Fairness (fr35_): menu v1 with 20 bases of context on ten other reservoirs against the same configurations without context (sc31_/sc32_ cells): ensemble r per reservoir and set, and the rank correlation
    of the reservoirs with and without context. Stacks there are fitted on each directory's own validation labels.

    python scripts/analysis/analyze_pk35.py [--out outputs/justification_oct22]      (cpuq)
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
MENU_ARMS = {  # key: (directory, group)
    "base": ("hp34_base_30k", "base"),
    "lrx05": ("hp34_lrx05_30k", "recipe"),
    "lrx2": ("hp34_lrx2_30k", "recipe"),
    "bs128": ("hp34_bs128_30k", "recipe"),
    "bs512": ("hp34_bs512_30k", "recipe"),
    "wd1e3": ("hp34_wd1e3_30k", "recipe"),
    "adamw": ("hp34_adamw_30k", "recipe"),
    "lgrecipe": ("lg34_lgrecipe_30k", "recipe"),
    "flank_shift": ("fl34_flank_30k", "flank"),
    "c0s30": ("fx35_c0s30_30k", "flank"),
    "c4s10": ("fx35_c4s10_30k", "context"),
    "c8s10": ("fx35_c8s10_30k", "context"),
    "c12s10": ("fx35_c12s10_30k", "context"),
    "c20s10": ("fx35_c20s10_30k", "context"),
    "c20s0": ("fx35_c20s0_30k", "context"),
    "c20s30": ("fx35_c20s30_30k", "context"),
    "c60s10": ("fx35_c60s10_30k", "context"),
    "c100s10": ("fx35_c100s10_30k", "context"),
    "c200s10": ("fx35_c200s10_30k", "context"),
    "ema": ("pk35_ema_30k", "averaging"),
    "bn": ("pk35_bn_30k", "averaging"),
    "emabn": ("pk35_emabn_30k", "averaging"),
    "ema999bn": ("pk35_ema999bn_30k", "averaging"),
    "c20s10_emabn": ("fe35_c20s10_emabn_30k", "combined"),
}
LG = {
    "base": "lg35_base_30k",
    "emabn": "lg35_emabn_30k",
    "flank": "lg35_flank_30k",
    "emabn_flank": "lg35_emabn_flank_30k",
    "c20": "lg35c_c20_30k",
    "c20_emabn": "lg35c_c20_emabn_30k",
    "c60_emabn": "lg35c_c60_emabn_30k",
}
FR = [
    "random",
    "motif_ct_enriched",
    "evoaug_paper2023",
    "zoonomia_orthologs",
    "motif_shared_core",
    "random_gc_sweep",
    "mutagenesis_mr10",
    "d3_steps64",
    "d3_steps256",
    "evo2",
]


def pearson(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def load(d, vlab=None):
    """{(seed index, menu index): model} for every finished model of a retrain directory; validation r against `vlab` (default V0c)."""
    vlab = VLAB if vlab is None else vlab
    out = {}
    for mf in glob.glob(f"{MR}/{d}/r*_menu_*_meta.json"):
        b = os.path.basename(mf).split("_")
        s, i = int(b[0][1:]), int(b[2])
        try:
            z = np.load(mf.replace("_meta.json", ".npz"))
            v = z["val_pred"].astype(np.float32)
            t = {k: z[f"test_pred_{k}"].astype(np.float32) for k in SETS}
            meta = json.load(open(mf))
        except Exception:
            continue
        if not np.isfinite(v).all() or any(
            not np.isfinite(x).all() or np.abs(x).max() > 1e3 for x in t.values()
        ):
            continue
        out[(s, i)] = {
            "val": v,
            "test": t,
            "epochs": meta.get("epochs_trained"),
            "sec": meta.get("train_time_sec"),
            "cap_hit": bool(meta.get("cap_hit")),
            "n_params": meta.get("n_params"),
        }
    for m in out.values():
        m["vr"] = pearson(m["val"], vlab)
        m["tr"] = {k: pearson(m["test"][k], CANON[k]) for k in SETS}
        m["tmeas"] = pearson(m["test"]["genomic"], MEAS)
    return out


def zero_seed(ms):
    return {(0, i): m for (s, i), m in ms.items()}


def stack(ms, vlab=None):
    vlab = VLAB if vlab is None else vlab
    V = np.stack([m["val"] for m in ms])
    st = fit_zscored_stack(V, vlab)
    o = {"n": len(ms), "n_used": int((np.asarray(st.coef) > 1e-9).sum())}
    for k in SETS:
        o[k] = pearson(st.predict(np.stack([m["test"][k] for m in ms])), CANON[k])
    o["genomic_measured"] = pearson(st.predict(np.stack([m["test"]["genomic"] for m in ms])), MEAS)
    return o


def paired(a, b, keys=None):
    c = sorted(set(a) & set(b)) if keys is None else sorted(set(keys) & set(a) & set(b))
    res = {"n": len(c)}
    for name, f in (
        ("val", lambda m: m["vr"]),
        ("genomic_measured", lambda m: m["tmeas"]),
        *[(k, (lambda k: lambda m: m["tr"][k])(k)) for k in SETS],
    ):
        d = np.array([f(a[i]) - f(b[i]) for i in c])
        res[name] = {
            "mean": float(d.mean()) if len(d) else float("nan"),
            "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
            "better": int((d > 0).sum()),
        }
    return res


def cost(ms):
    e = [m["epochs"] for m in ms.values() if m["epochs"]]
    s = [m["sec"] for m in ms.values() if m["sec"]]
    return {
        "n": len(ms),
        "mean_epochs": float(np.mean(e)) if e else None,
        "mean_sec": float(np.mean(s)) if s else None,
        "cap_hit": int(sum(m["cap_hit"] for m in ms.values())),
    }


def mean_single(ms):
    return {
        "val": float(np.mean([m["vr"] for m in ms.values()])),
        **{k: float(np.mean([m["tr"][k] for m in ms.values()])) for k in SETS},
        "genomic_measured": float(np.mean([m["tmeas"] for m in ms.values()])),
    }


def pr(label, d):
    print(
        label,
        {
            k: (round(v["mean"], 4), round(v["se"], 4), v["better"]) if isinstance(v, dict) else v
            for k, v in d.items()
        },
        flush=True,
    )


def main():
    global VLAB, CANON, MEAS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    a = ap.parse_args()
    VLAB = np.load(f"{SH}/val_V0c.npz", allow_pickle=True)["oracle_labels"].astype(np.float64)
    CANON = {}
    for k in SETS:
        z = np.load(f"{SH}/test_sets/{k}_oracle.npz", allow_pickle=True)
        CANON[k] = z["oracle_mean"].astype(np.float64)
        if k == "genomic":
            MEAS = z["true_label"].astype(np.float64)
    arms = {k: zero_seed(load(d)) for k, (d, _) in MENU_ARMS.items()}
    arms["base_s1"] = zero_seed(load("hp34s1_base_30k"))
    base = arms["base"]
    out = {
        "n_models": {k: len(v) for k, v in arms.items()},
        "paired": {},
        "ensembles": {},
        "cost": {},
        "mean_single": {},
        "group": {k: g for k, (_, g) in MENU_ARMS.items()},
    }
    out["seed_floor"] = paired(arms["base_s1"], base)
    pr("seed_floor", out["seed_floor"])
    for k, ms in arms.items():
        if k == "base" or not ms:
            continue
        out["paired"][k] = paired(ms, base)
        out["cost"][k] = cost(ms)
        out["mean_single"][k] = mean_single(ms)
        pr(k, out["paired"][k])
    out["cost"]["base"], out["mean_single"]["base"] = cost(base), mean_single(base)
    for k, ms in arms.items():
        if len(ms) >= 8:
            out["ensembles"][k] = stack(list(ms.values()))
            if k != "base":
                out["ensembles"][k + "__base_same_configs"] = stack(
                    [m for i, m in base.items() if i in ms]
                )
    out["ensembles"]["pool_all_menu_arms"] = stack(
        [m for k, ms in arms.items() if k != "base_s1" for m in ms.values()]
        + list(arms["base_s1"].values())
    )
    for k, v in out["ensembles"].items():
        print(
            "ensemble",
            k,
            {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()},
            flush=True,
        )
    out["models"] = [
        {
            "arm": k,
            "idx": i[1],
            "val": m["vr"],
            **{s: m["tr"][s] for s in SETS},
            "genomic_measured": m["tmeas"],
            "n_params": m["n_params"],
        }
        for k, ms in arms.items()
        for i, m in sorted(ms.items())
    ]
    # LegNet-style reference: 3 recipes x 3 seeds per arm
    lg = {arm: load(d) for arm, d in LG.items()}
    out["legnet"] = {"arms": {}, "paired_vs_base": {}, "recipes": {}}
    for arm, ms in lg.items():
        if not ms:
            continue
        out["legnet"]["arms"][arm] = {
            "n": len(ms),
            "mean_single": mean_single(ms),
            "stack": stack(list(ms.values())) if len(ms) >= 3 else None,
            "cost": cost(ms),
        }
        out["legnet"]["recipes"][arm] = {
            str(i): {
                k: float(np.mean([m["tr"][k] for (s, j), m in ms.items() if j == i])) for k in SETS
            }
            for i in sorted({i for _, i in ms})
        }
        if arm != "base" and lg["base"]:
            out["legnet"]["paired_vs_base"][arm] = paired(ms, lg["base"])
            pr("legnet " + arm, out["legnet"]["paired_vs_base"][arm])
    for arm, v in out["legnet"]["arms"].items():
        print(
            "legnet",
            arm,
            {k: round(x, 4) for k, x in v["mean_single"].items()},
            "stack",
            None if v["stack"] is None else round(v["stack"]["genomic"], 4),
            flush=True,
        )
    # fairness: does the ranking of reservoirs change when the student sees the flank?
    fair = {}
    for res in ["genomic"] + FR:
        if res == "genomic":
            with_d, without_d = "fx35_c20s10_30k", "hp34_base_30k"
        else:
            with_d = f"fr35_{res}_c20_30k"
            without_d = (
                f"sc31_30k_{res}" if os.path.isdir(f"{MR}/sc31_30k_{res}") else f"sc32_30k_{res}"
            )
        try:
            vl_w = np.load(f"{MR}/{with_d}/labels.npz")["val_labels"].astype(np.float64)
            vl_o = np.load(f"{MR}/{without_d}/labels.npz")["val_labels"].astype(np.float64)
        except Exception:
            continue
        w, o = zero_seed(load(with_d, vl_w)), zero_seed(load(without_d, vl_o))
        keys = sorted(set(w) & set(o))
        if len(keys) < 8:
            continue
        fair[res] = {
            "n_configs": len(keys),
            "with": stack([w[i] for i in keys], vl_w),
            "without": stack([o[i] for i in keys], vl_o),
            "mean_single_with": {k: float(np.mean([w[i]["tr"][k] for i in keys])) for k in SETS},
            "mean_single_without": {k: float(np.mean([o[i]["tr"][k] for i in keys])) for k in SETS},
            "complete": len(keys) >= 14,
        }
        print(
            "fairness",
            res,
            len(keys),
            {k: (round(fair[res]["without"][k], 4), round(fair[res]["with"][k], 4)) for k in SETS},
            flush=True,
        )
    out["fairness"] = fair
    full = [r for r, v in fair.items() if v["complete"]]
    if len(full) >= 4:
        from scipy.stats import spearmanr

        out["fairness_rank"] = {
            k: float(
                spearmanr(
                    [fair[r]["without"][k] for r in full], [fair[r]["with"][k] for r in full]
                )[0]
            )
            for k in SETS
        }
        out["fairness_rank"]["n_reservoirs"] = len(full)
        print(
            "rank correlation of reservoirs with vs without context",
            out["fairness_rank"],
            flush=True,
        )
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/data_pk35.json", "w"), indent=1, default=float)
    print("wrote", f"{a.out}/data_pk35.json")


if __name__ == "__main__":
    main()
