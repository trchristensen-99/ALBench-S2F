#!/usr/bin/env python
"""Build the evidence page (a single HTML file with inline SVG charts) from the stored loop and bake-off results.

Every hand-entered number lives in `template.html` (object `K`, each with its source named in the card under it); the numbers computed from stored runs are computed here:
the epochs-against-validation scatter, best validation r so far by round for LegNet-named and independent designs, the share of runs with a new best by round (from
`extract_diversity_funnel.py` output), and the bake-off table (`bake_partial.json`, written by the cluster-side script `scratchpad/bake_table.py`).

    python scripts/analysis/evidence_page/build_evidence_page.py --funnel data_diversity_funnel_30k.json --bake bake_partial.json --out evidence.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STD_ARMS = (
    "rr40",
    "rr39",
    "ef39_20",
    "ef39_60",
    "ef39_100",
    "mgr38_ref",
)  # V0c-validated standard runs (vc39 used another validation set)


def funnel_data(path: str) -> dict:
    d = json.load(open(path))
    runs = [(a, r, v) for a in STD_ARMS if a in d for r, v in d[a].items()]
    scatter = [
        [
            int(float(m["epochs"])),
            round(m["val"], 4),
            0 if m["text_legnet"] else 1,
            0 if m["round"] < 6 else 1,
        ]
        for _, _, v in runs
        for m in v["models"]
        if m["epochs"] is not None
    ]
    best = {"named": [], "indep": []}
    for rd in range(15):
        for g, named in (("named", True), ("indep", False)):
            vals = []
            for _, _, v in runs:
                xs = [
                    m["val"] for m in v["models"] if m["text_legnet"] == named and m["round"] <= rd
                ]
                if xs:
                    vals.append(max(xs))
            best[g].append(round(float(np.mean(vals)), 4) if vals else None)
    new_best = []
    for rd in range(1, 15):
        hit = n = 0
        for _, _, v in runs:
            prev = [m["val"] for m in v["models"] if m["round"] < rd]
            cur = [m["val"] for m in v["models"] if m["round"] == rd]
            if prev and cur:
                n += 1
                hit += max(cur) > max(prev)
        new_best.append(round(hit / max(n, 1), 3))
    return {"scatter": scatter, "bestByRound": best, "newBest": new_best, "nRuns": len(runs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--funnel", required=True)
    ap.add_argument("--bake", default="")
    ap.add_argument("--stack", default="", help="data_bake_stack.json from analyze_bake_stack.py")
    ap.add_argument("--template", default=str(Path(__file__).with_name("template.html")))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = funnel_data(a.funnel)
    data["bake"] = json.load(open(a.bake)) if a.bake else []
    data["stack"] = json.load(open(a.stack)) if a.stack else {}
    html = Path(a.template).read_text()
    assert html.count("/*__DATA__*/{}") == 1
    Path(a.out).write_text(html.replace("/*__DATA__*/{}", json.dumps(data, separators=(",", ":"))))
    print(
        "wrote",
        a.out,
        len(html) // 1024,
        "KB template;",
        data["nRuns"],
        "runs;",
        len(data["scatter"]),
        "scatter points;",
        len(data["bake"]),
        "bake designs",
    )


if __name__ == "__main__":
    main()
