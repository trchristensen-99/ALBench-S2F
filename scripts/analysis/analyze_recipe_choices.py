# ruff: noqa
#!/usr/bin/env python
"""Why do the 20 menu configurations share one training recipe? What the proposer chose in all default-loop 30k search runs, and what survived selection.

The proposer may write a TRAIN line (optimizer, lr, weight decay, batch size, epochs, schedule) for every architecture it proposes (albench/arch_search/arch_autoresearch.py TRAIN_NOTE).
This reads every trained candidate of the default-loop 30k runs (outputs/archsearch_reps_oct22/<reservoir>_30k/rep*/r*_meta.json) and records: the distribution of each setting
over all candidates, over the 3 best candidates of each run (what the menu draws from), the within-run difference of validation r between Muon and AdamW candidates, and how the
share of each setting moves over rounds.

    python scripts/analysis/analyze_recipe_choices.py [--out outputs/justification_oct22]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics as st

KEYS = ("optimizer", "lr", "weight_decay", "batch_size", "lr_schedule", "epochs")
EXCLUDE = ("agent", "explore", "noguid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    a = ap.parse_args()
    fs = [
        f
        for f in sorted(glob.glob("outputs/archsearch_reps_oct22/*_30k/rep*/r*_meta.json"))
        if not any(x in f for x in EXCLUDE)
    ]
    rows = []
    for f in fs:
        m = json.load(open(f))
        tc = m.get("train_config") or {}
        if m.get("val_pearson") is None:
            continue
        rows.append(
            dict(
                run=f.rsplit("/", 1)[0],
                rd=int(m["round"]),
                v=float(m["val_pearson"]),
                **{k: tc.get(k) for k in KEYS},
            )
        )
    runs = sorted({r["run"] for r in rows})
    out = {"n_candidates": len(rows), "n_runs": len(runs)}
    top = []
    for run in runs:
        top += sorted((r for r in rows if r["run"] == run), key=lambda r: -r["v"])[:3]
    out["share_all"], out["share_top3"], out["val_by_setting"] = {}, {}, {}
    for k in KEYS:
        ca = collections.Counter(str(r[k]) for r in rows)
        ct = collections.Counter(str(r[k]) for r in top)
        out["share_all"][k] = {v: c / len(rows) for v, c in ca.items()}
        out["share_top3"][k] = {v: c / len(top) for v, c in ct.items()}
        g = collections.defaultdict(list)
        for r in rows:
            g[str(r[k])].append(r["v"])
        out["val_by_setting"][k] = {
            v: {"n": len(x), "mean": st.mean(x), "best": max(x)}
            for v, x in g.items()
            if len(x) >= 15
        }
    d = []
    for run in runs:
        m_ = [r["v"] for r in rows if r["run"] == run and r["optimizer"] == "muon"]
        a_ = [r["v"] for r in rows if r["run"] == run and r["optimizer"] == "adamw"]
        if len(m_) >= 3 and len(a_) >= 3:
            d.append(
                {
                    "run": run,
                    "diff": st.mean(m_) - st.mean(a_),
                    "n_muon": len(m_),
                    "n_adamw": len(a_),
                }
            )
    out["muon_minus_adamw_by_run"] = d
    bins = {"0-2": (0, 2), "3-6": (3, 6), "7-11": (7, 11)}
    out["share_by_round"] = {}
    for name, (lo, hi) in bins.items():
        s = [r for r in rows if lo <= r["rd"] <= hi]
        out["share_by_round"][name] = {
            "n": len(s),
            "muon": sum(r["optimizer"] == "muon" for r in s) / len(s),
            "lr0.002": sum(r["lr"] == 0.002 for r in s) / len(s),
            "wd1e-4": sum(r["weight_decay"] == 0.0001 for r in s) / len(s),
            "bs256": sum(r["batch_size"] == 256 for r in s) / len(s),
            "onecycle": sum(r["lr_schedule"] == "onecycle" for r in s) / len(s),
        }
    # the recipe is a package: how often does each optimizer come with each weight decay
    out["wd_by_optimizer"] = {
        o: dict(collections.Counter(str(r["weight_decay"]) for r in rows if r["optimizer"] == o))
        for o in ("muon", "adamw")
    }
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/data_recipes.json", "w"), indent=1)
    print(
        "wrote",
        f"{a.out}/data_recipes.json",
        "|",
        len(rows),
        "candidates in",
        len(runs),
        "runs; muon better within run in",
        sum(x["diff"] > 0 for x in d),
        "of",
        len(d),
    )


if __name__ == "__main__":
    main()
