"""How many ROUNDS are needed before more rounds stop helping?

Peter's constraint is that high-D runs must not be dominated by HP search. Model
count is one lever (answered: k=40); rounds are the other, and they are what the
LLM arms actually pay for -- each round is a bridge call and a serialisation point.

For each cell, score the deploy ensemble using only models proposed in rounds
0..R. The knee of that curve is the round budget.
"""
from __future__ import annotations
import glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.analysis.hp_protocol_study import greedy, load_cell


def main():
    out = {}
    for cd in sorted(glob.glob("outputs/cadence/*_*/")):
        cell = os.path.basename(cd.rstrip("/"))
        models, lab = load_cell(cd.rstrip("/"))
        if not models or len(models) < 10:
            continue
        yv, yt = lab
        rnd = {}
        for f in glob.glob(f"{cd}/r*_meta.json"):
            m = re.match(r"r(\d+)_", os.path.basename(f))
            if not m:
                continue
            try:
                mid = json.load(open(f)).get("model_id")
            except Exception:
                continue
            if mid:
                rnd[mid] = int(m.group(1))
        maxr = max(rnd.values()) if rnd else 0
        curve = []
        for R in range(0, maxr + 1):
            pool = [m for m in models if rnd.get(m["id"], 999) <= R]
            if len(pool) < 5:
                continue
            _, s = greedy(pool, yv, yt)
            curve.append((R + 1, len(pool), float(s)))
        if not curve:
            continue
        best = max(c[2] for c in curve)
        # first round-count within the 0.0044 noise floor of this cell's best
        knee = next((c[0] for c in curve if c[2] >= best - 0.0044), curve[-1][0])
        out[cell] = {"curve": curve, "best": best, "rounds_to_knee": knee,
                     "total_rounds": maxr + 1}
        print("  %-32s rounds=%2d  best=%.4f  knee at round %d  (%.0f%% of rounds)"
              % (cell, maxr + 1, best, knee, 100 * knee / (maxr + 1)))
    json.dump(out, open("outputs/figures/rounds_to_converge.json", "w"), indent=2)
    print(f"\n  wrote outputs/figures/rounds_to_converge.json ({len(out)} cells)")


if __name__ == "__main__":
    main()
