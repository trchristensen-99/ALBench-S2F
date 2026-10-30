#!/usr/bin/env python
"""Our AutoResearch runs on the 30k genomic pool (real plasmid flank as input) against the numbers of the collaborator's 200-model search.

His table (Oct 29): best single model validation 0.8540 / genomic test 0.8165; best three-seed average test 0.7969; top-5 / top-10 / top-20 ensemble test
0.8317 / 0.8304 / 0.8298. For every one of our runs (each directory is one independent search with its own training draw order and LLM randomness) we compute,
on the same kind of quantities:

  best_single   the model with the best validation r: its validation r and its test r (against the oracle labels and against the measured labels)
  mean_single   mean test r over all models of the run (no selection at all)
  top<K>_mean   mean prediction of the K models with the best validation r (K = 5, 10, 20), his way of building an ensemble
  top<K>_stack  the same K models with positive ElasticNet weights fitted on validation (our way)
  all_stack     positive ElasticNet over every model of the run

Only validation (V0c) selects models and weights; test sets are never used for a choice. Then mean, SD, min, max over runs and how many runs reach each of his numbers.
Which labels his test r refers to is not known (his files are not here yet), so both label columns are given.

    python scripts/analysis/compare_pi_30k.py [--out outputs/justification_oct22]      (cpuq)
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

AR = "outputs/archsearch_reps_oct22"
GROUPS = {
    "ctx35 (standard loop, 4 runs)": f"{AR}/genomic_30k_ctx35/rep*",
    "mgr35 coordinator-schedule (3)": f"{AR}/genomic_30k_mgr35_c15/rep*",
    "mgr35 explore-then-exploit (3)": f"{AR}/genomic_30k_mgr35_e15/rep*",
    "st38 a0 (2)": f"{AR}/st38_a0_genomic_30k/rep*",
    "st38 a4 gentle steering (2)": f"{AR}/st38_a4_genomic_30k/rep*",
    "lng35 long runs (2)": f"{AR}/genomic_30k_lng35/rep*",
}
GROUPS_10K = {
    f"st38 {a} (3)": f"{AR}/st38_{a}_genomic_10k/rep*" for a in ("a0", "a1", "a2", "a3", "a4")
}
GROUPS_10K["lng35 long runs 10k (2)"] = f"{AR}/genomic_10k_lng35/rep*"
# models of different searches that were trained on the SAME training pool (same replicate) can be combined into one ensemble
COMBINE_10K = {
    f"combined rep{r}": [
        f"{AR}/st38_{a}_genomic_10k/rep{r}" for a in ("a0", "a1", "a2", "a3", "a4")
    ]
    for r in (1, 2, 3)
}
PI = {
    "best_single_test": 0.8165,
    "best_single_val": 0.8540,
    "top5_test": 0.8317,
    "top10_test": 0.8304,
    "top20_test": 0.8298,
    "best_3seed_mean_test": 0.7969,
}


def pearson(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def load(dirs):
    """Models of one run, or of several runs that share the training pool and the evaluation labels (``dirs`` a path or a list of paths)."""
    dirs = [dirs] if isinstance(dirs, str) else list(dirs)
    lab, yv, ms = None, None, []
    for d in dirs:
        lb = np.load(f"{d}/labels.npz")
        y = lb["val_labels"].astype(np.float64)
        if lab is None:
            lab, yv = lb, y
        elif not (np.allclose(yv, y) and np.allclose(lab["oracle_genomic"], lb["oracle_genomic"])):
            raise SystemExit(f"{d}: validation or test labels differ from the first run")
        for mf in sorted(glob.glob(f"{d}/r*_arch_*_meta.json")):
            try:
                z = np.load(mf.replace("_meta.json", ".npz"))
                v, t = z["val_pred"].astype(np.float64), z["test_pred_genomic"].astype(np.float64)
            except Exception:
                continue
            if not (np.isfinite(v).all() and np.isfinite(t).all()) or np.abs(t).max() > 1e3:
                continue
            ms.append({"v": v, "t": t, "valr": pearson(v, yv)})
    return lab, yv, ms


def run_metrics(d) -> dict:
    lab, yv, ms = load(d)
    out: dict = {"n_models": len(ms)}
    if len(ms) < 6:
        return out
    yo = lab["oracle_genomic"].astype(np.float64)
    ym = lab["test_true"].astype(np.float64) if "test_true" in lab.files else None
    order = np.argsort([-m["valr"] for m in ms])
    V = np.stack([m["v"] for m in ms])
    T = np.stack([m["t"] for m in ms])
    b = int(order[0])
    out["best_single_val"] = ms[b]["valr"]
    out["best_single_test_oracle"] = pearson(T[b], yo)
    out["mean_single_test_oracle"] = float(np.mean([pearson(t, yo) for t in T]))
    if ym is not None:
        out["best_single_test_measured"] = pearson(T[b], ym)
    for K in (5, 10, 20):
        if len(ms) < K:
            continue
        sel = order[:K]
        out[f"top{K}_mean_oracle"] = pearson(T[sel].mean(0), yo)
        st = fit_zscored_stack(V[sel], yv)
        out[f"top{K}_stack_oracle"] = pearson(st.predict(T[sel]), yo)
        if ym is not None:
            out[f"top{K}_stack_measured"] = pearson(st.predict(T[sel]), ym)
    st = fit_zscored_stack(V, yv)
    out["all_stack_oracle"] = pearson(st.predict(T), yo)
    if ym is not None:
        out["all_stack_measured"] = pearson(st.predict(T), ym)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    ap.add_argument("--size", default="30k", choices=["30k", "10k"])
    a = ap.parse_args()
    groups = GROUPS if a.size == "30k" else GROUPS_10K
    res: dict = {"pi": PI if a.size == "30k" else {}, "groups": {}, "combined": {}}
    keys = [
        "best_single_val",
        "best_single_test_oracle",
        "best_single_test_measured",
        "mean_single_test_oracle",
        "top5_mean_oracle",
        "top10_mean_oracle",
        "top20_mean_oracle",
        "top5_stack_oracle",
        "top10_stack_oracle",
        "top20_stack_oracle",
        "all_stack_oracle",
        "all_stack_measured",
    ]
    allruns = []
    for g, pat in groups.items():
        runs = {}
        for d in sorted(glob.glob(pat)):
            if os.path.exists(f"{d}/labels.npz"):
                runs[os.path.basename(d)] = run_metrics(d)
        res["groups"][g] = runs
        allruns += [r for r in runs.values() if r.get("n_models", 0) >= 6]
        print(g, {k: v.get("n_models") for k, v in runs.items()}, flush=True)
    if a.size == "10k":
        for name, dirs in COMBINE_10K.items():
            if all(os.path.exists(f"{d}/labels.npz") for d in dirs):
                res["combined"][name] = run_metrics(dirs)
                print(name, res["combined"][name].get("n_models"), flush=True)
    summary = {}
    for k in keys:
        x = np.array([r[k] for r in allruns if k in r], float)
        if len(x):
            summary[k] = {
                "n_runs": int(len(x)),
                "mean": float(x.mean()),
                "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                "min": float(x.min()),
                "max": float(x.max()),
            }
    thresholds = {
        "best_single_test_oracle": PI["best_single_test"],
        "top5_mean_oracle": PI["top5_test"],
        "top10_mean_oracle": PI["top10_test"],
        "top20_mean_oracle": PI["top20_test"],
        "top5_stack_oracle": PI["top5_test"],
        "top10_stack_oracle": PI["top10_test"],
        "top20_stack_oracle": PI["top20_test"],
        "mean_single_test_oracle": PI["best_3seed_mean_test"],
    }
    for k, th in thresholds.items() if a.size == "30k" else []:
        x = np.array([r[k] for r in allruns if k in r], float)
        if len(x) and k in summary:
            summary[k]["pi_reference"] = th
            summary[k]["runs_at_or_above_pi"] = int((x >= th).sum())
    res["all_runs_summary"] = summary
    os.makedirs(a.out, exist_ok=True)
    stem = "data_pi30k" if a.size == "30k" else f"data_size{a.size}"
    json.dump(res, open(f"{a.out}/{stem}.json", "w"), indent=1, default=float)
    print(f"{len(allruns)} runs with >= 6 models")
    for k, v in summary.items():
        ref = (
            f" | PI {v['pi_reference']:.4f}, runs at or above: {v['runs_at_or_above_pi']}/{v['n_runs']}"
            if "pi_reference" in v
            else ""
        )
        print(
            f"{k:28s} mean {v['mean']:.4f} sd {v['sd']:.4f} min {v['min']:.4f} max {v['max']:.4f}{ref}"
        )
    for name, r in res["combined"].items():
        print(name, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)})
    print("wrote", f"{a.out}/{stem}.json")


if __name__ == "__main__":
    main()
