# ruff: noqa
#!/usr/bin/env python
"""Best numbers we can reach on the Gosai genomic set at 30k training sequences: every model we trained on a genomic 30k pool, pooled (Oct 28).

Pool: (1) every AutoResearch search run on a genomic 30k pool (outputs/archsearch_reps_oct22/genomic_30k*, outputs/archsearch_final_oct20/*; the models trained during the searches, all loop variants and
models, 10 to 30 rounds), (2) every menu retrain on a genomic 30k pool (menu v1 and v0, replicate pools r1-r3, extra seeds, patience, recipe and flank arms of today). Left out: label-noise,
HepG2, validation-variant and holdout-split runs. Models trained on different 30k draws of the genomic reservoir (replicate pools) are pooled on purpose: each is a legitimate 30k-trained model.

Scoring: everything against the SHARED test sets (oracle_v3 labels: genomic chr7/13 n=31,435, designed OOD n=22,962, random 32k) and the V0c validation labels
(/grid/koo/home/shared/albench/data). A directory whose stored test labels or validation sequences do not match is skipped and listed. Reported:
  single models   best by V0c validation r and its test r; the best test r of any model (an upper bound that uses the test set); distribution over models
  ensembles       positive ElasticNetCV stack (albench.ensemble.stack.fit_zscored_stack) fitted on V0c over the K models with the best V0c validation r, K = 20 ... 800; a 50/50 split of V0c gives a
                  validation-only estimate for each K (fit on one half, correlation on the other) that selects K without touching a test set
  one run         the stack of each single AutoResearch run alone (what one outside run of the loop gets), mean / max over runs
  against the experimental labels on the genomic test set, for the headline rows

    python scripts/analysis/analyze_best30k.py [--out outputs/justification_oct22]      (cpuq, 16 cores, ~1 h)
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())
from albench.ensemble.stack import fit_zscored_stack  # noqa: E402

SH = "/grid/koo/home/shared/albench/data"
MR = "outputs/menu_retrain_v0"
SETS = ("genomic", "ood", "random_32k")
KS = (20, 50, 100, 200, 400, 800)
RETRAIN_GLOBS = [
    "sc31_30k_genomic",
    "v1_30k_genomic_r1",
    "ind30k_genomic_r1",
    "ind30k_genomic_r1_seeds3to8",
    "ind30k_genomic_r2",
    "ind30k_genomic_r3",
    "nest_genomic_r1_d30k",
    "ens30k_genomic_p16",
    "ens30k_genomic_p8",
    "hp34_*_30k",
    "hp34s1_*_30k",
    "fl34_*_30k",
    "fl34s1_*_30k",
    "tr30k_*_genomic_r*",
    "as30k_M*_genomic",
    "acq30k_genomic_r1",
    "ar2m_genomic_30k",
    "prec30k_*_genomic_r1",
]
SEARCH_GLOBS = [
    "outputs/archsearch_reps_oct22/genomic_30k*/rep*",
    "outputs/archsearch_final_oct20/*/rep*",
]


def pearson(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def discover():
    dirs = []
    for g in RETRAIN_GLOBS:
        dirs += [
            ("retrain", d)
            for d in sorted(glob.glob(f"{MR}/{g}"))
            if os.path.exists(f"{d}/labels.npz")
        ]
    for g in SEARCH_GLOBS:
        dirs += [
            ("search", d)
            for d in sorted(glob.glob(g))
            if os.path.exists(f"{d}/labels.npz") and "smoke" not in d
        ]
    seen, out = set(), []
    for k, d in dirs:
        if d not in seen:
            seen.add(d)
            out.append((k, d))
    return out


def load_models(canon, vlab):
    models, skipped = [], []
    for kind, d in discover():
        try:
            lab = np.load(f"{d}/labels.npz")
            ok = len(lab["val_labels"]) == len(vlab) and pearson(lab["val_labels"], vlab) > 0.97
            for s in SETS:
                ok = (
                    ok
                    and f"oracle_{s}" in lab.files
                    and len(lab[f"oracle_{s}"]) == len(canon[s])
                    and np.abs(lab[f"oracle_{s}"] - canon[s]).max() < 0.05
                )
            if not ok:
                skipped.append((d, "labels differ"))
                continue
        except Exception as e:
            skipped.append((d, f"labels: {e}"))
            continue
        pat = "r*_menu_*_meta.json" if kind == "retrain" else "r*_arch_*_meta.json"
        n0 = len(models)
        for mf in sorted(glob.glob(f"{d}/{pat}")):
            try:
                z = np.load(mf.replace("_meta.json", ".npz"))
                if not all(f"test_pred_{s}" in z.files for s in SETS):
                    continue
                v = z["val_pred"].astype(np.float32)
                t = {s: z[f"test_pred_{s}"].astype(np.float32) for s in SETS}
                if (
                    not (np.isfinite(v).all() and all(np.isfinite(x).all() for x in t.values()))
                    or np.abs(v).max() > 1e3
                    or any(np.abs(x).max() > 1e3 for x in t.values())
                ):
                    continue
                meta = json.load(open(mf))
            except Exception:
                continue
            models.append(
                {
                    "dir": d,
                    "kind": kind,
                    "id": os.path.basename(mf).replace("_meta.json", ""),
                    "val": v,
                    "test": t,
                    "n_params": meta.get("n_params"),
                    "idea": (meta.get("idea") or meta.get("menu_entry") or "")[:80],
                }
            )
        print(f"{kind:7s} {d.split('outputs/')[-1]:62s} {len(models) - n0:4d} models", flush=True)
    return models, skipped


def stack_eval(args):
    """-> test r per set (and against the measured labels) of the stack over rows `idx`, fitted on V0c; plus a 50/50 validation estimate."""
    K, idx, cv = args
    V = VAL[idx]
    st = fit_zscored_stack(V, VLAB)
    res = {
        "K": K,
        "n_used": int((np.asarray(st.coef) > 1e-9).sum()),
        "val_fit_r": pearson(st.predict(V), VLAB),
    }
    for s in SETS:
        res[s] = pearson(st.predict(TEST[s][idx]), CANON[s])
    res["genomic_measured"] = pearson(st.predict(TEST["genomic"][idx]), MEAS)
    if cv:
        rng = np.random.default_rng(K)
        perm = rng.permutation(len(VLAB))
        a, b = perm[: len(perm) // 2], perm[len(perm) // 2 :]
        r = []
        for tr, te in ((a, b), (b, a)):
            st2 = fit_zscored_stack(V[:, tr], VLAB[tr])
            r.append(pearson(st2.predict(V[:, te]), VLAB[te]))
        res["val_cv_r"] = float(np.mean(r))
    print("K", K, {k: round(v, 4) for k, v in res.items() if isinstance(v, float)}, flush=True)
    return res


def main():
    global VAL, VLAB, TEST, CANON, MEAS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    t0 = time.time()
    VLAB = np.load(f"{SH}/val_V0c.npz", allow_pickle=True)["oracle_labels"].astype(np.float64)
    CANON, MEAS = {}, None
    for s in SETS:
        z = np.load(f"{SH}/test_sets/{s}_oracle.npz", allow_pickle=True)
        CANON[s] = z["oracle_mean"].astype(np.float64)
        if s == "genomic":
            MEAS = z["true_label"].astype(np.float64)
    models, skipped = load_models(CANON, VLAB)
    print(
        f"{len(models)} models from {len({m['dir'] for m in models})} directories ({sum(m['kind'] == 'search' for m in models)} search-time, {sum(m['kind'] == 'retrain' for m in models)} retrained); {len(skipped)} directories skipped",
        flush=True,
    )
    VAL = np.stack([m["val"] for m in models])
    TEST = {s: np.stack([m["test"][s] for m in models]) for s in SETS}
    val_r = np.array([pearson(v, VLAB) for v in VAL])
    test_r = {s: np.array([pearson(t, CANON[s]) for t in TEST[s]]) for s in SETS}
    meas_r = np.array([pearson(t, MEAS) for t in TEST["genomic"]])
    ok = np.isfinite(val_r) & (val_r > 0.3)
    keep = np.where(ok)[0]
    print(
        f"{len(keep)} models kept after the sanity filter (V0c r > 0.3); loaded in {time.time() - t0:.0f}s",
        flush=True,
    )
    order = keep[np.argsort(-val_r[keep])]
    out = {
        "n_models": int(len(keep)),
        "n_dirs": len({models[i]["dir"] for i in keep}),
        "n_search": int(sum(models[i]["kind"] == "search" for i in keep)),
        "n_retrain": int(sum(models[i]["kind"] == "retrain" for i in keep)),
        "skipped": skipped,
    }
    b = order[0]
    out["single"] = {
        "best_by_val": {
            "id": models[b]["id"],
            "dir": models[b]["dir"],
            "kind": models[b]["kind"],
            "idea": models[b]["idea"],
            "n_params": models[b]["n_params"],
            "val_r": float(val_r[b]),
            **{s: float(test_r[s][b]) for s in SETS},
            "genomic_measured": float(meas_r[b]),
        },
        "best_test_any": {s: float(test_r[s][keep].max()) for s in SETS},
        "distribution": {
            s: {
                "mean": float(test_r[s][keep].mean()),
                "sd": float(test_r[s][keep].std()),
                "p90": float(np.percentile(test_r[s][keep], 90)),
                "max": float(test_r[s][keep].max()),
            }
            for s in SETS
        },
        "top5_by_val": [
            {
                "id": models[i]["id"],
                "dir": models[i]["dir"].split("outputs/")[-1],
                "val_r": float(val_r[i]),
                **{s: float(test_r[s][i]) for s in SETS},
            }
            for i in order[:5]
        ],
    }
    print("best single by V0c val:", json.dumps(out["single"]["best_by_val"]), flush=True)
    jobs = [(K, order[:K], True) for K in KS if K <= len(order)] + [(len(order), order, False)]
    with mp.get_context("fork").Pool(min(a.procs, len(jobs))) as pool:
        out["ensembles"] = pool.map(stack_eval, jobs, chunksize=1)
    # one run alone
    per_run = {}
    for i in keep:
        if models[i]["kind"] == "search":
            per_run.setdefault(models[i]["dir"], []).append(i)
    runs = []
    for d, idx in per_run.items():
        if len(idx) >= 20:
            idx = np.array(idx)[:20] if False else np.array(idx)
            st = fit_zscored_stack(VAL[idx], VLAB)
            runs.append(
                {
                    "dir": d.split("outputs/")[-1],
                    "n": len(idx),
                    **{s: pearson(st.predict(TEST[s][idx]), CANON[s]) for s in SETS},
                }
            )
    out["per_run"] = runs
    if runs:
        out["per_run_summary"] = {
            s: {
                "mean": float(np.mean([r[s] for r in runs])),
                "sd": float(np.std([r[s] for r in runs])),
                "max": float(np.max([r[s] for r in runs])),
                "n_runs": len(runs),
            }
            for s in SETS
        }
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/data_best30k.json", "w"), indent=1, default=float)
    print("wrote", f"{a.out}/data_best30k.json", f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
