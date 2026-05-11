"""Quick summary of debias sweep winners across v10-v17."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
rows = []
for sw in ["debias_sweep_v10", "debias_sweep_v11", "debias_sweep_v12", "debias_sweep_v13",
          "debias_sweep_v14", "debias_sweep_v15", "debias_sweep_v17"]:
    base = REPO / f"outputs/oracle_neg_sweep/{sw}"
    if not base.is_dir():
        continue
    for cell in base.iterdir():
        if not cell.is_dir():
            continue
        for f in [cell / "fold_0" / "test_metrics.json", cell / "test_metrics.json"]:
            if f.exists():
                t = json.loads(f.read_text())
                inner = t.get("test_metrics", {})
                be = f.parent / "bias_eval.json"
                rm = 0
                if be.exists():
                    rm = json.loads(be.read_text()).get("random_dna", {}).get("mean", 0)
                rows.append({
                    "sweep": sw.replace("debias_sweep_", ""),
                    "label": cell.name,
                    "test_id": inner.get("in_distribution", {}).get("pearson_r", 0),
                    "ood": inner.get("ood", {}).get("pearson_r", 0),
                    "snv_d": inner.get("snv_delta", {}).get("pearson_r", 0),
                    "random_mean": rm,
                })
                break

print(f"Total v10-v17 cells: {len(rows)}")

print("\nTop 12 by composite (test_id + ood - 0.3*bias):")
SW = "sweep"; LB = "label"; TI = "test_id"; OD = "ood"; SD = "snv_d"; BS = "bias"
print(f"{SW:<8} {LB:<35} {TI:>8} {OD:>7} {SD:>7} {BS:>7}")
print("-" * 75)
rows.sort(key=lambda r: -(r["test_id"] + r["ood"] - 0.3 * r["random_mean"]))
for r in rows[:12]:
    print(f"{r['sweep']:<8} {r['label']:<35} {r['test_id']:8.4f} {r['ood']:7.3f} {r['snv_d']:7.3f} {r['random_mean']:7.3f}")

print("\nBest by BIAS REDUCTION (with test_id > 0.93):")
print(f"{SW:<8} {LB:<35} {TI:>8} {OD:>7} {SD:>7} {BS:>7}")
print("-" * 75)
rows_b = [r for r in rows if r["test_id"] > 0.93]
rows_b.sort(key=lambda r: r["random_mean"])
for r in rows_b[:8]:
    print(f"{r['sweep']:<8} {r['label']:<35} {r['test_id']:8.4f} {r['ood']:7.3f} {r['snv_d']:7.3f} {r['random_mean']:7.3f}")
