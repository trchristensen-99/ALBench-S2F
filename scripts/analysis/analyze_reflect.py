# ruff: noqa
#!/usr/bin/env python
"""Three questions from the Oct-28 afternoon discussion, answered from data on disk.

(1) How different are the architectures? Every candidate of the default-loop 30k search runs (outputs/archsearch_reps_oct22) and every menu-v1 configuration is classified from its source code
    (convolution, self-attention, recurrent, state-space, up/down-sampling path, dilation, squeeze-excite, shared weights ...); validation r and share of the best 3 per run by class.
    And how different are the members' predictions? Pairwise correlation of the 20 members of the 30k genomic retrain on the genomic test set.
(2) What happens if the bottom few members by validation r are left out of the stack? For each retrained pool (sc31_{10,30}k_*, v1_100k_*), the stack is refit on the best n-m members by retrained validation r,
    by search validation r, and on random subsets of the same size; validation r of the stack and test r on the genomic / designed / random sets.

    python scripts/analysis/analyze_reflect.py [--out outputs/justification_oct22]      (cpuq)
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_oct32 as a32  # noqa: E402
from albench.ensemble.stack import fit_zscored_stack  # noqa: E402

SETS = ("genomic", "ood", "random_32k")
CLASSES = {
    "self_attention": r"MultiheadAttention|TransformerEncoder|nn\.Transformer|scaled_dot_product_attention|softmax\([^)]*transpose|torch\.bmm\([^)]*softmax",
    "attention_pooling": r"[Aa]ttn[_A-Za-z]*\s*=|AttnPool|AttentionPool|attention pool|softmax\(.*dim=-1\)",
    "recurrent": r"nn\.(LSTM|GRU|RNN)\b",
    "state_space": r"[Mm]amba|[Ss]tate.?[Ss]pace|selective.?scan|\bS4\b",
    "updown_path": r"ConvTranspose|F\.interpolate|nn\.Upsample|Upsample",
    "dilated": r"dilation\s*=\s*(?!1\b)\d+|dilations?\b",
    "squeeze_excite": r"\bSE\b|[Ss]queeze|SEBlock",
    "convolution": r"nn\.Conv1d|nn\.Conv2d|F\.conv1d|Conv1d|Conv2d",
}


def classify(src):
    return {k: bool(re.search(v, src or "")) for k, v in CLASSES.items()}


def an_arch(out):
    menu = json.load(open("albench/ensemble/menus/menu_v1.json"))
    mc = [classify(e.get("source")) for e in menu]
    out["menu_classes"] = {k: int(sum(c[k] for c in mc)) for k in CLASSES}
    out["menu_n"] = len(menu)
    fs = [
        f
        for f in sorted(glob.glob("outputs/archsearch_reps_oct22/*_30k/rep*/r*_meta.json"))
        if not any(x in f for x in ("agent", "explore", "noguid"))
    ]
    rows = []
    for f in fs:
        m = json.load(open(f))
        if m.get("val_pearson") is None:
            continue
        rows.append(
            {
                "run": f.rsplit("/", 1)[0],
                "v": float(m["val_pearson"]),
                "c": classify(m.get("source")),
                "n_params": m.get("n_params"),
            }
        )
    top = set()
    for run in {r["run"] for r in rows}:
        rs = sorted(
            (i for i, r in enumerate(rows) if r["run"] == run), key=lambda i: -rows[i]["v"]
        )[:3]
        top |= set(rs)
    res = {}
    for k in CLASSES:
        yes = [r["v"] for r in rows if r["c"][k]]
        no = [r["v"] for r in rows if not r["c"][k]]
        res[k] = {
            "n": len(yes),
            "share": len(yes) / len(rows),
            "mean_val": float(np.mean(yes)) if yes else None,
            "mean_val_without": float(np.mean(no)) if no else None,
            "share_of_top3": float(np.mean([rows[i]["c"][k] for i in top])),
        }
    nonconv = [r["v"] for r in rows if not r["c"]["convolution"]]
    res["no_convolution_at_all"] = {
        "n": len(nonconv),
        "mean_val": float(np.mean(nonconv)) if nonconv else None,
    }
    out["candidates"] = {"n": len(rows), "n_runs": len({r["run"] for r in rows}), "classes": res}
    print(
        "candidates",
        len(rows),
        {
            k: (v["n"], round(v["mean_val"], 3) if v["mean_val"] else None)
            for k, v in res.items()
            if k != "no_convolution_at_all"
        },
        flush=True,
    )


def an_diversity(out):
    c = a32.load_all("sc31_30k_genomic")
    st, V = a32.stack(c)
    coef = np.asarray(st.coef)
    T = np.stack([r["test"]["genomic"] for r in c["rows"]]).astype(float)
    R = np.corrcoef(T)
    iu = np.triu_indices(len(R), 1)
    yt = c["y"]["genomic"]
    single = [a32.pearson(t, yt) for t in T]
    out["diversity"] = {
        "n_members": len(R),
        "pairwise_r_mean": float(R[iu].mean()),
        "pairwise_r_min": float(R[iu].min()),
        "pairwise_r_max": float(R[iu].max()),
        "single_test_r_mean": float(np.mean(single)),
        "mean_prediction_ensemble_r": a32.pearson(T.mean(0), yt),
        "stack_r": a32.pearson(st.predict(np.stack([r["test"]["genomic"] for r in c["rows"]])), yt),
        "weights_gt0": int((coef > 1e-9).sum()),
        "top3_weight_share": float(np.sort(coef)[::-1][:3].sum() / max(coef.sum(), 1e-12)),
    }
    print(
        "diversity",
        {k: round(v, 4) if isinstance(v, float) else v for k, v in out["diversity"].items()},
        flush=True,
    )


def an_drop(out, n_rand=30, seed=0):
    rng = np.random.default_rng(seed)
    menu = {
        e["id"]: float(e["orig_val"])
        for e in json.load(open("albench/ensemble/menus/menu_v1.json"))
        if e.get("orig_val") is not None
    }
    dirs = [f"sc31_{d}k_{s}" for d in (10, 30) for s in a32.STRATS] + [
        f"v1_100k_{s}_r1" for s in ("genomic", "random", "motif_ct_enriched")
    ]
    res = {}
    for d in dirs:
        c = a32.load_all(d)
        if not c or len(c["rows"]) < 14:
            continue
        rows = c["rows"]
        n = len(rows)
        vy = c["val_y"]
        val_r = np.array([a32.pearson(r["val"], vy) for r in rows])
        search_r = np.array([menu.get(r["meta"].get("menu_entry"), np.nan) for r in rows])
        V = np.stack([r["val"] for r in rows])
        T = {s: np.stack([r["test"][s] for r in rows]) for s in SETS}

        def score(idx):
            idx = list(idx)
            st = fit_zscored_stack(V[idx], vy)
            return {
                "val": a32.pearson(st.predict(V[idx]), vy),
                **{s: a32.pearson(st.predict(T[s][idx]), c["y"][s]) for s in SETS},
            }

        cell = {"n": n, "by_val": {}, "by_search": {}, "random": {}}
        for m in range(0, min(10, n - 6) + 1):
            k = n - m
            cell["by_val"][m] = score(np.argsort(-val_r)[:k])
            if np.isfinite(search_r).all():
                cell["by_search"][m] = score(np.argsort(-search_r)[:k])
            rs = [score(rng.choice(n, k, replace=False)) for _ in range(n_rand if m else 1)]
            cell["random"][m] = {key: float(np.mean([x[key] for x in rs])) for key in rs[0]}
        res[d] = cell
        print(
            d,
            n,
            "m=0 genomic %.4f | drop 5 by val %.4f, by search %.4f, random %.4f"
            % (
                cell["by_val"][0]["genomic"],
                cell["by_val"][5]["genomic"],
                cell["by_search"].get(5, {}).get("genomic", float("nan")),
                cell["random"][5]["genomic"],
            ),
            flush=True,
        )
    out["drop"] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    a = ap.parse_args()
    out = {}
    for fn in (an_arch, an_diversity, an_drop):
        try:
            fn(out)
        except Exception:
            import traceback

            traceback.print_exc()
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/data_reflect.json", "w"), indent=1, default=float)
    print("wrote", f"{a.out}/data_reflect.json")


if __name__ == "__main__":
    main()
