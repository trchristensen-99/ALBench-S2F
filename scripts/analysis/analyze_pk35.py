# ruff: noqa
#!/usr/bin/env python
"""Oct-29 harness experiments on the canonical genomic 30k pool, V0c validation: weight averaging + BN recalculation (pk35_), real flank context (fx35_), and a LegNet-style reference under the same switches (lg35_).

(1) Menu arms (pk35_, fx35_): paired difference against hp34_base_30k (same configuration, same seed index) of validation r and test r on the three shared test sets, its standard error over configurations,
    the number of configurations that improve, against the seed-noise floor (hp34s1_base_30k minus hp34_base_30k). Context arms hold the configurations that can take wider input; the base is restricted to the
    same configurations. Mean epochs and training seconds per arm (what the switch costs).
(2) Ensembles (positive ElasticNetCV stacks fitted on V0c): each arm alone against the base on the same configurations; the base plus all arms pooled.
(3) LegNet reference (lg35_): per arm, the mean single-model test r over (3 recipes x 3 seeds), the paired difference against lg35_base (same recipe and seed), and the 9-model stack.

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
PK = ["ema", "bn", "emabn", "ema999bn"]
FX = ["c0s30", "c20s10", "c60s10", "c100s10"]
LG = ["base", "emabn", "flank", "emabn_flank"]


def pearson(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def load(d):
    """{(seed index, menu index): model} for every finished model of a retrain directory."""
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
            "gpu": meta.get("gpu", ""),
            "cap_hit": bool(meta.get("cap_hit")),
            "n_params": meta.get("n_params"),
            "bn_gain": (meta.get("val_mse_before_bn_recalc"), meta.get("val_mse")),
        }
    for m in out.values():
        m["vr"] = pearson(m["val"], VLAB)
        m["tr"] = {k: pearson(m["test"][k], CANON[k]) for k in SETS}
    return out


def stack(ms):
    V = np.stack([m["val"] for m in ms])
    st = fit_zscored_stack(V, VLAB)
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
    base = load("hp34_base_30k")
    base_s1 = {(0, i): m for (s, i), m in load("hp34s1_base_30k").items()}
    base0 = {(0, i): m for (s, i), m in base.items()}
    out = {
        "menu": {},
        "ensembles": {},
        "cost": {"base": cost(base)},
        "mean_single": {"base": mean_single(base)},
    }
    pr("seed_floor", paired(base_s1, base0))
    out["seed_floor"] = paired(base_s1, base0)
    arms = {}
    for arm in PK:
        arms[f"pk35_{arm}"] = {(0, i): m for (s, i), m in load(f"pk35_{arm}_30k").items()}
    for arm in FX:
        arms[f"fx35_{arm}"] = {(0, i): m for (s, i), m in load(f"fx35_{arm}_30k").items()}
    arms["fl34_flank"] = {(0, i): m for (s, i), m in load("fl34_flank_30k").items()}
    for name, ms in arms.items():
        if not ms:
            continue
        keys = set(ms)
        out["menu"][name] = {"vs_base": paired(ms, base0), "n_models": len(ms)}
        out["cost"][name] = cost(ms)
        out["mean_single"][name] = mean_single(ms)
        pr(name, out["menu"][name]["vs_base"])
        if len(ms) >= 10:
            sub = [m for k, m in base0.items() if k in keys]
            out["ensembles"][name] = stack(list(ms.values()))
            out["ensembles"][name + "__base_same_configs"] = stack(sub)
    allm = list(base0.values()) + [m for ms in arms.values() for m in ms.values()]
    if arms:
        out["ensembles"]["pool_base_and_all_new_arms"] = stack(allm)
    for k, v in out["ensembles"].items():
        print(
            "ensemble",
            k,
            {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()},
            flush=True,
        )
    # LegNet-style reference: 3 recipes x 3 seeds per arm
    lg = {arm: load(f"lg35_{arm}_30k") for arm in LG}
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
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/data_pk35.json", "w"), indent=1, default=float)
    print("wrote", f"{a.out}/data_pk35.json")


if __name__ == "__main__":
    main()
