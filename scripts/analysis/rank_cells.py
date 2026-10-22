"""Rank ensembled cells at a FIXED ensemble size (default 5 members).

Only cells with a full model pool are comparable: an arm that has had 62 configs
to greedily select 5 from has a real advantage over one that has had 40, so
mixing them ranks pool size as much as strategy. Incomplete cells are still
printed, but below the line and marked, never inside the comparison.
"""

from __future__ import annotations

import glob
import json
import os
import re

N_ENS = int(os.environ.get("N_ENS", "5"))        # deploy ensemble size
FULL = int(os.environ.get("FULL_POOL", "60"))    # models required to be comparable


def load(path):
    try:
        d = json.load(open(path))
    except Exception:
        return None
    curve = d.get("curve") or []
    pt = next((c for c in curve if c["n"] == N_ENS), None)
    if pt is None:
        return None
    cell = os.path.basename(os.path.dirname(os.path.dirname(path)))
    m = re.match(r"^(.*)_d(\d+)$", cell)
    arm, D = (m.group(1), int(m.group(2))) if m else (cell, -1)
    # gate on the model files actually present, not n_models_valid (which drops
    # models that failed to load and would mark a finished cell as incomplete)
    pool = len(glob.glob(os.path.join(os.path.dirname(os.path.dirname(path)), "*_meta.json")))
    return {
        "arm": arm, "D": D, "pool": pool,
        "r": pt.get("test_oracle_pearson"), "mse": pt.get("test_oracle_mse"),
        "full": pool >= FULL,
    }


rows = [r for r in (load(p) for p in
        sorted(glob.glob("outputs/stage4_v2_d*/*/ensemble/greedy_deploy.json"))) if r]
if not rows:
    raise SystemExit("no ensembled cells yet")

for D in sorted({r["D"] for r in rows}, reverse=True):
    sub = [r for r in rows if r["D"] == D]
    ok = sorted([r for r in sub if r["full"]], key=lambda r: -(r["r"] or -9))
    part = sorted([r for r in sub if not r["full"]], key=lambda r: -(r["r"] or -9))
    print(f"\n=== added D={D:,}  (base=0)  ensemble n={N_ENS} ===")
    print(f"  {'arm':<26} {'pool':>4} {'r@' + str(N_ENS):>8} {'mse':>8}")
    for r in ok:
        print(f"  {r['arm']:<26} {r['pool']:>4} {r['r']:>8.4f} {r['mse']:>8.4f}")
    if not ok:
        print("  (no complete cells yet)")
    if part:
        print(f"  -- below: pool < {FULL}, NOT comparable --")
        for r in part:
            print(f"  {r['arm']:<26} {r['pool']:>4} {r['r']:>8.4f} {r['mse']:>8.4f}  *")

n_full = sum(1 for r in rows if r["full"])
print(f"\ncomparable cells: {n_full}/{len(rows)}   (ensemble n={N_ENS}, full pool >= {FULL})")
