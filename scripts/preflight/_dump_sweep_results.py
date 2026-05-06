"""Quick-dump utility to show test_metrics + bias_eval for a sweep dir.

Usage:
    uv run --no-sync python scripts/preflight/_dump_sweep_results.py outputs/oracle_neg_sweep/debias_sweep_v4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    base = Path(sys.argv[1])
    rows = []
    for d in sorted(base.glob("c*/fold_0")):
        tm = d / "test_metrics.json"
        if not tm.exists():
            continue
        name = d.parent.name
        m = json.loads(tm.read_text())["test_metrics"]
        be = d / "bias_eval.json"
        b = json.loads(be.read_text()) if be.exists() else {}
        rows.append(
            {
                "name": name,
                "in_dist": m["in_distribution"]["pearson_r"],
                "ood": m["ood"]["pearson_r"],
                "ood_mse": m["ood"]["mse"],
                "rdna": b.get("random_dna", {}).get("mean", 0),
                "shuffled": b.get("shuffled", {}).get("mean", 0),
                "intergenic": b.get("intergenic", {}).get("mean", 0),
            }
        )
    print(
        f"{'name':<35}  {'in_d':>6}  {'OOD':>6}  {'OOD_MSE':>7}  {'rdna':>7}  {'shuf':>7}  {'inter':>7}"
    )
    for r in rows:
        print(
            f"{r['name']:<35}  {r['in_dist']:>6.3f}  {r['ood']:>6.3f}  {r['ood_mse']:>7.2f}  "
            f"{r['rdna']:>+7.2f}  {r['shuffled']:>+7.2f}  {r['intergenic']:>+7.2f}"
        )


if __name__ == "__main__":
    main()
