# ruff: noqa: E731
#!/usr/bin/env python
"""Build menu v1 from existing AutoResearch runs: 3 runs from each of 5 reservoir families (genomic, random, motif CT-enriched, evoaug, evo2) and 3 runs on the mixed corpus (18 runs);
the top configuration by search validation r of every run, plus the 2 best of the remaining candidates (top 3 of each run were retrained at 30k), 20 configurations in all.

A candidate must have trained without failing on all four 30k target pools it was retrained on (genomic, random, motif, mixed); the three older configurations that failed the CPU
forward-time check or exploded are therefore left out automatically.

    python scripts/analysis/build_menu_v1.py --out albench/ensemble/menus/menu_v1.json   (run in the repository root on the cluster; reads outputs/menus_oct22 and outputs/menu_retrain_v0)
"""

from __future__ import annotations

import argparse
import glob
import json
import os

FAMILY = {
    "son55_wide": "genomic",
    "random": "random",
    "motif_ct_enriched": "motif CT-enriched",
    "evo2": "evo2",
    "evoaug_paper2023": "evoaug",
    "GRM_30k": "mixed corpus (genomic + random + motif)",
}
MR = "outputs/menu_retrain_v0"


def available(cond: str) -> set[int]:
    """Menu indices that trained on every one of the four 30k target pools."""
    sets = []
    for t in ("genomic", "random", "motif", "grm"):
        ok = {
            json.load(open(f))["menu_index"]
            for f in glob.glob(f"{MR}/as30k_{cond}_{t}/r00_menu_*_meta.json")
        }
        sets.append(ok)
    return set.intersection(*sets)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--first-n",
        type=int,
        default=2,
        help="extra configurations taken from the remaining candidates",
    )
    args = ap.parse_args()
    mf, mx = (json.load(open(f"outputs/menus_oct22/{c}.json")) for c in ("MF", "MX"))
    ok_mf, ok_mx = available("MF"), available("MX")
    cands = []
    for i, e in enumerate(mf):
        if i in ok_mf:
            cands.append(e)
    for i, e in enumerate(mx):
        if i in ok_mx and e["cell"] in {f"GRM_30k/rep{r}" for r in (1, 2, 3)}:
            cands.append(e)
    runs: dict[str, list] = {}
    for e in cands:
        runs.setdefault(e["cell"], []).append(e)
    print(f"{len(runs)} runs, {len(cands)} usable candidates")
    top, rest = [], []
    for run, es in runs.items():
        es.sort(key=lambda e: -(e["orig_val"] or -1))
        top.append(es[0])
        rest += es[1:]
    top.sort(key=lambda e: -(e["orig_val"] or -1))
    rest.sort(key=lambda e: -(e["orig_val"] or -1))
    chosen = top + rest[: args.first_n]
    for e in chosen:
        e["family"] = FAMILY[e["cell"].split("/")[0]]
    json.dump(chosen, open(args.out, "w"), indent=1)
    per = {
        "menu": "v1",
        "rule": "given",
        "criterion": "one weighted ensemble over every configuration trained (positive ElasticNetCV on z-scored validation predictions); no subset of 8 is selected. 20 configurations at every size up to 100k; 12 from 300k (tentative: cost)",
        "construction": "top configuration by search validation r of each of 18 runs (3 on each of genomic, random, motif CT-enriched, evoaug, evo2 and 3 on the mixed corpus), plus the best 2 of the remaining candidates; built by scripts/analysis/build_menu_v1.py from existing 30k runs",
        "skip": {},
        "order": [e["id"] for e in chosen],
        "sizes": {
            str(D): {
                "k": 20 if D <= 100_000 else 12,
                "seeds": 1,
                "patience": 16,
                "max_minutes_h100": {10_000: 15.0, 30_000: 25.0, 100_000: 60.0, 300_000: 120.0}[D],
            }
            for D in (10_000, 30_000, 100_000, 300_000)
        },
    }
    json.dump(
        per, open(os.path.join(os.path.dirname(args.out) or ".", "per_size_v1.json"), "w"), indent=1
    )
    print(f"menu v1: {len(chosen)} configurations -> {args.out}")
    for e in chosen:
        print(
            f"  {e['id']:<40s} {e['family']:<42s} search val {e['orig_val']:.4f} {e.get('n_params') or 0:>10,} params"
        )


if __name__ == "__main__":
    main()
