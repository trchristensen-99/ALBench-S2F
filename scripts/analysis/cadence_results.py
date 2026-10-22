"""Deploy-ensemble score vs model budget, per cadence arm.

Each cell trained its 60 models IN ORDER, so evaluating the greedy n=5 ensemble
over the first k models reproduces what that arm would have delivered on a budget
of k -- the whole 6/12/.../60 curve from one run.
"""
from __future__ import annotations
import glob, json, os, re, sys
import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.analysis.hp_protocol_study import greedy, load_cell   # reuse the same selector

BUDGETS = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]

def ordered(cell_dir):
    """Models in the order the search produced them (round, then index)."""
    out = []
    for f in glob.glob(os.path.join(cell_dir, "r*_meta.json")):
        m = re.match(r"r(\d+)_", os.path.basename(f))
        try: d = json.load(open(f))
        except Exception: continue
        out.append((int(m.group(1)) if m else 999, d.get("model_id", ""), f))
    return [f for _, _, f in sorted(out)]

def main():
    cells = sorted(glob.glob("outputs/cadence/*_*/"))
    res = {}
    for cd in cells:
        cell = os.path.basename(cd.rstrip("/"))
        models, lab = load_cell(cd.rstrip("/"))
        if not models or len(models) < 6: continue
        yv, yt = lab
        order = {os.path.basename(f).replace("_meta.json",""): i
                 for i, f in enumerate(ordered(cd.rstrip("/")))}
        models.sort(key=lambda m: order.get(m["id"], 999))
        curve = []
        for k in BUDGETS:
            if k > len(models): break
            _, s = greedy(models[:k], yv, yt)
            curve.append((k, float(s)))
        res[cell] = {"n": len(models), "curve": curve}
        print(f"  {cell:<34} " + "  ".join(f"{k}:{s:.4f}" for k, s in curve[-4:]))
    json.dump(res, open("outputs/figures/cadence_results.json","w"), indent=2)
    print("\n  === ARM COMPARISON (mean over reservoirs) ===")
    arms = {}
    for cell, r in res.items():
        arm = cell.split("_")[-1]
        for k, s in r["curve"]:
            arms.setdefault(arm, {}).setdefault(k, []).append(s)
    ks = sorted({k for a in arms.values() for k in a})
    print("  %-10s " % "arm" + "".join(f"{k:>9}" for k in ks))
    for arm in sorted(arms):
        row = "".join(f"{np.mean(arms[arm][k]):>9.4f}" if k in arms[arm] else f"{'-':>9}" for k in ks)
        print(f"  {arm:<10} {row}")

if __name__ == "__main__":
    main()
