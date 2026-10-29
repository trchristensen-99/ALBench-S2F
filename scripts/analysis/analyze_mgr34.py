# ruff: noqa
#!/usr/bin/env python
"""Explore-then-exploit manager in the multi-agent AutoResearch loop (cells mgr34_): c15 (coordinator only, 15 rounds), s15 (manager "schedule"), e15 (manager "schedule_ens"), six replicate pools each, genomic 30k.

Per run, models in proposal order: stack r (fitted on V0c, test r on the three shared test sets and against the experimental labels) after the first 20 and after all 30 models; best single by validation; diversity of the run:
mean pairwise correlation of the models' test predictions, the number of distinct training recipes, the share of models that are not Muon / lr 2e-3, and counts of architecture classes in the sources (self-attention,
recurrent, U-Net-like, dilated, attention pooling). Arms are compared paired by replicate pool (same pool r1..r6). References: the Oct-25 10-round arms (default single call; coordinator only) at 20 models.

    python scripts/analysis/analyze_mgr34.py [--out outputs/justification_oct22]      (cpuq)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from albench.ensemble.stack import fit_zscored_stack  # noqa: E402

SH = "/grid/koo/home/shared/albench/data"
O22 = "outputs/archsearch_reps_oct22"
SETS = ("genomic", "ood", "random_32k")
CLS = {
    "self_attention": r"MultiheadAttention|TransformerEncoder|nn\.Transformer|scaled_dot_product_attention",
    "recurrent": r"nn\.(LSTM|GRU|RNN)\b",
    "updown": r"ConvTranspose|F\.interpolate|nn\.Upsample",
    "dilated": r"dilation\s*=\s*(?!1\b)\d+",
    "attn_pool": r"AttnPool|AttentionPool|[Aa]ttn[_A-Za-z]*\s*=",
}


def pearson(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def load_run(d):
    rows = []
    for mf in glob.glob(f"{d}/r*_arch_*_meta.json"):
        try:
            idx = int(re.search(r"_arch_(\d+)_meta", mf).group(1))
            z = np.load(mf.replace("_meta.json", ".npz"))
            v = z["val_pred"].astype(np.float32)
            t = {s: z[f"test_pred_{s}"].astype(np.float32) for s in SETS}
            if not np.isfinite(v).all() or any(
                not np.isfinite(x).all() or np.abs(x).max() > 1e3 for x in t.values()
            ):
                continue
            meta = json.load(open(mf))
        except Exception:
            continue
        rows.append(
            {
                "i": idx,
                "val": v,
                "test": t,
                "src": meta.get("source") or "",
                "tc": meta.get("train_config") or {},
            }
        )
    return sorted(rows, key=lambda r: r["i"])


def stack_r(rows):
    V = np.stack([r["val"] for r in rows])
    st = fit_zscored_stack(V, VLAB)
    out = {s: pearson(st.predict(np.stack([r["test"][s] for r in rows])), CANON[s]) for s in SETS}
    out["genomic_measured"] = pearson(
        st.predict(np.stack([r["test"]["genomic"] for r in rows])), MEAS
    )
    return out


def describe(rows):
    T = np.stack([r["test"]["genomic"] for r in rows])
    R = np.corrcoef(T)
    recipes = {
        (
            r["tc"].get("optimizer"),
            r["tc"].get("lr"),
            r["tc"].get("weight_decay"),
            r["tc"].get("batch_size"),
            r["tc"].get("lr_schedule"),
        )
        for r in rows
    }
    d = {
        "pairwise_corr": float(R[np.triu_indices(len(R), 1)].mean()),
        "n_recipes": len(recipes),
        "share_not_muon": float(np.mean([r["tc"].get("optimizer") != "muon" for r in rows])),
        "share_not_default_recipe": float(
            np.mean(
                [
                    not (
                        r["tc"].get("optimizer") == "muon"
                        and r["tc"].get("lr") == 0.002
                        and r["tc"].get("batch_size") == 256
                    )
                    for r in rows
                ]
            )
        ),
    }
    for k, pat in CLS.items():
        d[k] = int(sum(bool(re.search(pat, r["src"])) for r in rows))
    vr = [pearson(r["val"], VLAB) for r in rows]
    b = int(np.argmax(vr))
    d["best_single_val"] = float(vr[b])
    d["best_single_test"] = pearson(rows[b]["test"]["genomic"], CANON["genomic"])
    d["mean_single_test"] = float(
        np.mean([pearson(r["test"]["genomic"], CANON["genomic"]) for r in rows])
    )
    return d


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
        "c15": "genomic_30k_mgr_c15",
        "s15": "genomic_30k_mgr_s15",
        "e15": "genomic_30k_mgr_e15",
        "coord10": "genomic_30k_agent_coord_noref",
        "default10": "genomic_30k",
    }
    res = {}
    for arm, d in arms.items():
        res[arm] = {}
        for r in range(1, 7):
            rows = load_run(f"{O22}/{d}/rep{r}")
            if len(rows) < 20:
                continue
            e = {"n": len(rows), "stack20": stack_r(rows[:20]), "describe20": describe(rows[:20])}
            if len(rows) >= 30:
                e["stack30"] = stack_r(rows[:30])
                e["describe30"] = describe(rows[:30])
            res[arm][r] = e
            print(
                arm,
                r,
                e["n"],
                {k: round(v, 4) for k, v in e["stack20"].items()},
                {k: round(v, 4) for k, v in e.get("stack30", {}).items()},
                {
                    k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in e.get("describe30", e["describe20"]).items()
                    if k
                    in (
                        "pairwise_corr",
                        "n_recipes",
                        "share_not_muon",
                        "self_attention",
                        "recurrent",
                        "updown",
                    )
                },
                flush=True,
            )
    out = {"runs": res, "summary": {}, "paired": {}}
    for arm, rr in res.items():
        for key in ("stack20", "stack30"):
            v = [e[key] for e in rr.values() if key in e]
            if v:
                out["summary"].setdefault(arm, {})[key] = {
                    "n_runs": len(v),
                    **{
                        s: {
                            "mean": float(np.mean([x[s] for x in v])),
                            "sd": float(np.std([x[s] for x in v], ddof=1)) if len(v) > 1 else 0.0,
                        }
                        for s in (*SETS, "genomic_measured")
                    },
                }
        for key in ("describe20", "describe30"):
            v = [e[key] for e in rr.values() if key in e]
            if v:
                out["summary"].setdefault(arm, {})[key] = {
                    k: float(np.mean([x[k] for x in v])) for k in v[0]
                }
    for a_, b_ in (
        ("s15", "c15"),
        ("e15", "c15"),
        ("e15", "s15"),
        ("c15", "coord10"),
        ("c15", "default10"),
        ("s15", "coord10"),
        ("e15", "coord10"),
    ):
        for key in ("stack20", "stack30"):
            common = sorted(
                set(r for r, e in res[a_].items() if key in e)
                & set(r for r, e in res[b_].items() if key in e)
            )
            if len(common) < 3:
                continue
            o = {"n": len(common)}
            for s in (*SETS, "genomic_measured"):
                d = np.array([res[a_][r][key][s] - res[b_][r][key][s] for r in common])
                o[s] = {
                    "mean": float(d.mean()),
                    "se": float(d.std(ddof=1) / np.sqrt(len(d))),
                    "better": int((d > 0).sum()),
                }
            out["paired"][f"{a_}_minus_{b_}_{key}"] = o
            print(
                f"{a_} - {b_} {key} n={len(common)}",
                {
                    s: (round(o[s]["mean"], 4), round(o[s]["se"], 4), o[s]["better"])
                    for s in (*SETS, "genomic_measured")
                },
                flush=True,
            )
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/data_mgr34.json", "w"), indent=1, default=float)
    print("wrote", f"{a.out}/data_mgr34.json")


if __name__ == "__main__":
    main()
