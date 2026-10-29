# ruff: noqa
#!/usr/bin/env python
"""Steering experiment of the AutoResearch loop (cells st38_<arm>_<reservoir>_<D>k_r<k>, see scripts/program_oct22/register_st38.py).

For every finished or running search (outputs/archsearch_reps_oct22/st38_*/rep*) and for the standard control runs (genomic_30k_ctx35 for genomic 30k):
  n_models        trained models
  stack_<set>     test r of the positive-ElasticNet stack of the run's models, fitted on V0c (genomic test, designed set, random sequences)
  best_single     test r of the model with the best validation r
  member_corr     mean pairwise correlation of the members' genomic-test predictions (lower = more diverse)
  n_features      how many of the 17 architectural features of albench.arch_search.agent_loop._FEATURES appear in at least one model; plus the counts per feature
  n_recipes       distinct (optimizer, lr, wd, batch, schedule) recipes
  tokens_per_model  approximate LLM tokens per trained model including CLI overhead (agent_logs)
Aggregated per arm with the mean, the standard deviation across runs and the number of runs.

    python scripts/analysis/analyze_st38.py [--out outputs/justification_oct22]      (cpuq)
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
from albench.arch_search.agent_loop import _FEATURES, summarize_usage  # noqa: E402
from albench.ensemble.stack import fit_zscored_stack  # noqa: E402

SETS = ("genomic", "ood", "random_32k")


def pearson(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def analyse(d):
    lab = np.load(f"{d}/labels.npz")
    vlab = lab["val_labels"].astype(np.float64)
    ms = []
    for mf in sorted(glob.glob(f"{d}/r*_arch_*_meta.json")):
        try:
            z = np.load(mf.replace("_meta.json", ".npz"))
            m = json.load(open(mf))
            t = {s: z[f"test_pred_{s}"].astype(np.float64) for s in SETS}
            v = z["val_pred"].astype(np.float64)
        except Exception:
            continue
        if not np.isfinite(v).all() or any(
            not np.isfinite(x).all() or np.abs(x).max() > 1e3 for x in t.values()
        ):
            continue
        ms.append(
            {
                "v": v,
                "t": t,
                "src": m.get("source") or "",
                "tc": m.get("train_config", {}),
                "valr": pearson(v, vlab),
            }
        )
    out = {"n_models": len(ms)}
    if len(ms) < 6:
        return out
    V = np.stack([m["v"] for m in ms])
    st = fit_zscored_stack(V, vlab)
    for s in SETS:
        T = np.stack([m["t"][s] for m in ms])
        out[f"stack_{s}"] = pearson(st.predict(T), lab[f"oracle_{s}"])
        out[f"mean_single_{s}"] = float(np.mean([pearson(x, lab[f"oracle_{s}"]) for x in T]))
    best = int(np.argmax([m["valr"] for m in ms]))
    out["best_single"] = pearson(ms[best]["t"]["genomic"], lab["oracle_genomic"])
    G = np.stack([m["t"]["genomic"] for m in ms])
    C = np.corrcoef(G)
    out["member_corr"] = float(C[np.triu_indices(len(ms), 1)].mean())
    cnt = {k: sum(bool(re.search(p, m["src"])) for m in ms) for k, p in _FEATURES.items()}
    out["features"] = cnt
    out["n_features"] = int(sum(v > 0 for v in cnt.values()))
    out["n_recipes"] = len({tuple(sorted((k, str(v)) for k, v in m["tc"].items())) for m in ms})
    try:
        u = summarize_usage(d)
        out["tokens_per_model"] = float(u.get("approx_tokens_incl_cli_overhead", np.nan)) / max(
            len(ms), 1
        )
    except Exception:
        out["tokens_per_model"] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    a = ap.parse_args()
    runs = {}
    for d in sorted(glob.glob("outputs/archsearch_reps_oct22/st38_*/rep*")):
        arm = d.split("/")[-2]  # st38_a0_genomic_10k
        runs.setdefault(arm, {})[d.split("/")[-1]] = analyse(d)
    for r in range(
        1, 5
    ):  # control for genomic 30k: the standard runs launched before the arms existed
        d = f"outputs/archsearch_reps_oct22/genomic_30k_ctx35/rep{r}"
        if os.path.exists(f"{d}/labels.npz"):
            runs.setdefault("ctx35_standard_genomic_30k", {})[f"rep{r}"] = analyse(d)
    agg = {}
    for arm, rr in runs.items():
        ok = [v for v in rr.values() if "stack_genomic" in v]
        agg[arm] = {"n_runs": len(ok)}
        for k in (
            "stack_genomic",
            "stack_ood",
            "stack_random_32k",
            "best_single",
            "member_corr",
            "n_features",
            "n_recipes",
            "tokens_per_model",
            "n_models",
        ):
            xs = [v[k] for v in ok if k in v and np.isfinite(v[k])]
            if xs:
                agg[arm][k] = {
                    "mean": float(np.mean(xs)),
                    "sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else float("nan"),
                }
        print(
            arm,
            json.dumps(
                {
                    k: (round(v["mean"], 4) if isinstance(v, dict) else v)
                    for k, v in agg[arm].items()
                }
            ),
            flush=True,
        )
    os.makedirs(a.out, exist_ok=True)
    json.dump(
        {"runs": runs, "agg": agg}, open(f"{a.out}/data_st38.json", "w"), indent=1, default=float
    )
    print("wrote", f"{a.out}/data_st38.json")


if __name__ == "__main__":
    main()
