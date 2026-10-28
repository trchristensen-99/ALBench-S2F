# ruff: noqa
#!/usr/bin/env python
"""The pooled 30k result (analyze_best30k.py) mixes models trained on DIFFERENT 30k draws of the genomic reservoir (replicate pools r1..r15 and the canonical nested pool), so the union of data the ensemble has
seen is larger than 30k. This script groups the same models by the exact training file they were trained on (from the dispatcher registry, else from the run name) and scores each training set on its own:
best single model by V0c r, and stacks of the best 20 / 100 / all models of that training set, fitted on V0c, test r on the three shared test sets.

    python scripts/analysis/analyze_best30k_pools.py [--out outputs/justification_oct22]      (cpuq)
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_best30k as B  # noqa: E402
from albench.ensemble.stack import fit_zscored_stack  # noqa: E402

REPS = "genomic_r{n}_d30000"


def pool_of(d, reg):
    out = f"{os.getcwd()}/{d}" if not d.startswith("/") else d
    for c in reg:
        o = c.get("out_dir", "")
        if o and (out.endswith(o) or d.endswith(o)):
            m = re.search(r"--reservoir_cache (\S+)", c["cmd"])
            if m:
                return os.path.basename(m.group(1)).replace(".npz", "")
    m = re.search(r"/rep(\d+)$", d)
    if m and "archsearch" in d:
        return REPS.format(n=m.group(1))
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    a = ap.parse_args()
    reg = json.load(open("scratchpad/dispatch_registry.json"))
    vlab = np.load(f"{B.SH}/val_V0c.npz", allow_pickle=True)["oracle_labels"].astype(np.float64)
    canon, meas = {}, None
    for s in B.SETS:
        z = np.load(f"{B.SH}/test_sets/{s}_oracle.npz", allow_pickle=True)
        canon[s] = z["oracle_mean"].astype(np.float64)
        if s == "genomic":
            meas = z["true_label"].astype(np.float64)
    models, _ = B.load_models(canon, vlab)
    by = collections.defaultdict(list)
    for i, m in enumerate(models):
        by[pool_of(m["dir"], reg)].append(i)
    VAL = np.stack([m["val"] for m in models])
    TEST = {s: np.stack([m["test"][s] for m in models]) for s in B.SETS}
    vr = np.array([B.pearson(v, vlab) for v in VAL])
    res = {}
    for pool, idx in sorted(by.items(), key=lambda kv: -len(kv[1])):
        idx = np.array([i for i in idx if vr[i] > 0.3])
        if len(idx) < 20:
            continue
        order = idx[np.argsort(-vr[idx])]
        b = order[0]
        r = {
            "n_models": int(len(idx)),
            "n_dirs": len({models[i]["dir"] for i in idx}),
            "single_best_by_val": {s: B.pearson(TEST[s][b], canon[s]) for s in B.SETS},
            "single_best_by_val_val_r": float(vr[b]),
            "ens": {},
        }
        r["single_best_by_val"]["genomic_measured"] = B.pearson(TEST["genomic"][b], meas)
        for K in (20, 100, len(order)):
            if K > len(order):
                continue
            st = fit_zscored_stack(VAL[order[:K]], vlab)
            e = {s: B.pearson(st.predict(TEST[s][order[:K]]), canon[s]) for s in B.SETS}
            e["genomic_measured"] = B.pearson(st.predict(TEST["genomic"][order[:K]]), meas)
            e["n_used"] = int((np.asarray(st.coef) > 1e-9).sum())
            r["ens"][str(K)] = e
        res[pool] = r
        print(
            pool,
            json.dumps({k: v for k, v in r.items() if k != "ens"}),
            {
                K: {s: round(x, 4) for s, x in e.items() if isinstance(x, float)}
                for K, e in r["ens"].items()
            },
            flush=True,
        )
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/data_best30k_pools.json", "w"), indent=1, default=float)
    print("wrote", f"{a.out}/data_best30k_pools.json")


if __name__ == "__main__":
    main()
