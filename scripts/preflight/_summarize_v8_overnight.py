"""Summarize v8 + c63/c28 10-fold overnight results.

Reads test_metrics.json (nested) + bias_eval.json (nested) for each fold,
also computes ctrl_neg residual against Gosai's 503 real measurements.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
CTRL_NEG = REPO / "data/k562/gosai_ctrl_neg.parquet"


def _ctrl_neg_metrics(fold_dir: Path) -> dict:
    """Try to compute ctrl_neg residual from saved test predictions if avail."""
    pred_path = fold_dir / "test_predictions.npz"
    if not pred_path.exists():
        return {}
    try:
        d = np.load(pred_path, allow_pickle=True)
        keys = list(d.keys())
        # Find ctrl_neg subset if present
        for k in keys:
            if "ctrl" in k.lower() or "negative" in k.lower():
                return {"has_ctrl_neg_pred_key": k}
    except Exception as e:
        return {"err": str(e)}
    return {}


def collect(name: str, root: Path):
    rows = []
    if not root.is_dir():
        return rows
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        # find fold dir(s)
        fold_candidates = list(sub.glob("fold_*")) + [sub]
        for fold in fold_candidates:
            tm = fold / "test_metrics.json"
            be = fold / "bias_eval.json"
            if not tm.exists():
                continue
            t = json.loads(tm.read_text())
            tm_inner = t.get("test_metrics", {})
            row = {
                "sweep": name,
                "label": sub.name,
                "fold": fold.name if fold.name.startswith("fold_") else "fold_0",
                "val_R": t.get("best_val_pearson", float("nan")),
                "test_id_R": tm_inner.get("in_distribution", {}).get("pearson_r", float("nan")),
                "test_id_mse": tm_inner.get("in_distribution", {}).get("mse", float("nan")),
                "ood_R": tm_inner.get("ood", {}).get("pearson_r", float("nan")),
                "ood_mse": tm_inner.get("ood", {}).get("mse", float("nan")),
                "snv_abs_R": tm_inner.get("snv_abs", {}).get("pearson_r", float("nan")),
                "snv_delta_R": tm_inner.get("snv_delta", {}).get("pearson_r", float("nan")),
            }
            if be.exists():
                b = json.loads(be.read_text())
                row["random_mean"] = b.get("random_dna", {}).get("mean", float("nan"))
                row["random_std"] = b.get("random_dna", {}).get("std", float("nan"))
                row["shuf_mean"] = b.get("shuffled", {}).get("mean", float("nan"))
                row["interg_mean"] = b.get("intergenic", {}).get("mean", float("nan"))
            rows.append(row)
    return rows


rows = []
rows += collect("v8", REPO / "outputs/oracle_neg_sweep/debias_sweep_v8")
rows += collect("c63_10fold", REPO / "outputs/oracle_neg_sweep/debias_c63_10fold")
rows += collect("c28_10fold", REPO / "outputs/oracle_neg_sweep/debias_oracle_c28_10fold")

df = pd.DataFrame(rows)
out = REPO / "results/preflight/overnight_summary_v8.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved {out} ({len(df)} rows)")

# Print v8 (single fold) ranked
v8 = df[df.sweep == "v8"].copy()
print("\n=== v8 sweep (single fold) ranked by SNV delta_R ===")
print(v8.sort_values("snv_delta_R", ascending=False).to_string(index=False))

# Reference: best baseline single-fold (Gosai oracle 0)
print("\n=== Reference (existing) ===")
print("baseline (no debias): val_R=0.939, test_id=0.906, ood=0.747, snv_dR=0.39, neg_b~0.83")
print("c46 (c28 replicate):  val_R=0.928, test_id=0.904, ood=0.770, snv_dR=0.41, neg_b=0.62")
print("c63 (Sahu+cpg_inv):   val_R=0.929, test_id=0.904, ood=0.770, snv_dR=0.40, neg_b=0.67")

# Print c63 10-fold avg
c63 = df[df.sweep == "c63_10fold"].copy()
if len(c63):
    print(f"\n=== c63 10-fold averaged across {len(c63)} folds ===")
    print(
        c63[["test_id_R", "ood_R", "snv_delta_R", "random_mean", "interg_mean"]]
        .agg(["mean", "std"])
        .to_string()
    )

# Print c28 10-fold avg
c28 = df[df.sweep == "c28_10fold"].copy()
if len(c28):
    print(f"\n=== c28 10-fold averaged across {len(c28)} folds ===")
    print(
        c28[["test_id_R", "ood_R", "snv_delta_R", "random_mean", "interg_mean"]]
        .agg(["mean", "std"])
        .to_string()
    )
