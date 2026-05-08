"""Summarize v9 debias combos + c86 10-fold ensemble."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")


def get_metrics(fold_dir: Path):
    tm = fold_dir / "test_metrics.json"
    be = fold_dir / "bias_eval.json"
    if not tm.exists():
        return None
    t = json.loads(tm.read_text())
    inner = t.get("test_metrics", {})
    out = {
        "val_R": t.get("best_val_pearson", 0),
        "test_id": inner.get("in_distribution", {}).get("pearson_r", 0),
        "ood": inner.get("ood", {}).get("pearson_r", 0),
        "snv_d": inner.get("snv_delta", {}).get("pearson_r", 0),
    }
    if be.exists():
        b = json.loads(be.read_text())
        out["random_mean"] = b.get("random_dna", {}).get("mean", 0)
        out["interg_mean"] = b.get("intergenic", {}).get("mean", 0)
    return out


print("=== v9 DEBIAS COMBOS (single fold, ranked by composite score) ===")
print(f"{'label':<35} {'val':>6} {'test_id':>8} {'OOD':>6} {'snv_d':>6} {'rand':>6} {'interg':>7}")
print("-" * 80)
v9_dir = REPO / "outputs/oracle_neg_sweep/debias_sweep_v9"
rows = []
for d in sorted(v9_dir.iterdir()):
    if not d.is_dir():
        continue
    m = get_metrics(d / "fold_0")
    if m is None:
        continue
    rows.append((d.name, m))

# rank by test_id + ood - 0.3*random_mean (composite)
rows.sort(key=lambda r: -(r[1]["test_id"] + r[1]["ood"] - 0.3 * r[1].get("random_mean", 0)))
for name, m in rows:
    print(
        f"{name:<35} {m['val_R']:6.3f} {m['test_id']:8.4f} {m['ood']:6.3f} "
        f"{m['snv_d']:6.3f} {m.get('random_mean', 0):6.3f} {m.get('interg_mean', 0):7.3f}"
    )

print("\n=== c86 10-FOLD ENSEMBLE (per-fold mean ± std) ===")
c86 = REPO / "outputs/oracle_neg_sweep/debias_c86_10fold"
fold_rows = []
for fold in range(10):
    m = get_metrics(c86 / f"fold_{fold}")
    if m is not None:
        fold_rows.append(m)
print(f"  n_folds completed: {len(fold_rows)}/10")
for k in ["val_R", "test_id", "ood", "snv_d", "random_mean"]:
    vals = [r[k] for r in fold_rows if k in r]
    if vals:
        print(
            f"  {k:>11}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  "
            f"(range {min(vals):.4f} - {max(vals):.4f})"
        )

print("\n=== Reference 10-fold ensembles (existing) ===")
print("  baseline: test_id=0.929 OOD=0.748 snv_d=0.390  (no debias)")
print("  c63:      test_id=0.933 OOD=0.754 snv_d=0.389  (Sahu 3% + cpg_inv)")
print("  c28:      test_id=0.929 OOD=0.743 snv_d=0.389  (dinuc 3% + cpg_inv)")
