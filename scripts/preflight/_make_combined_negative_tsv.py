"""Create a combined negative-augmentation TSV that mixes:
  - 50% dinuc-shuffled (synthetic, mean ~0)
  - 30% Sahu-corrected (synthetic with real STARR-seq label distribution)
  - 20% gosai_ctrl_neg_calibration (real Gosai measurements, mean +0.27)

Hypothesis: stacking multiple inactive-distribution priors gives the model
a richer signal for what "non-regulatory DNA looks like" — combined real
+ synthetic anchors may reduce bias more reliably than any single source.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

DINUC = REPO / "data/synthetic_negatives/dinuc_shuffled_negatives.tsv"
SAHU = REPO / "data/synthetic_negatives_sahu/dinuc_shuffled_sahu.tsv"
CTRL = REPO / "data/synthetic_negatives_calibration/gosai_ctrl_neg_calibration.tsv"

OUT_DIR = REPO / "data/synthetic_negatives_combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load(p, source_tag):
    df = pd.read_csv(p, sep="\t")
    if "category" not in df.columns:
        df["category"] = source_tag
    if "K562_log2FC" not in df.columns:
        for c in df.columns:
            if "log2FC" in c or "label" in c.lower():
                df = df.rename(columns={c: "K562_log2FC"})
                break
    df["source"] = source_tag
    return df[["sequence", "K562_log2FC", "category", "source"]]


dinuc = load(DINUC, "dinuc_shuffled")
sahu = load(SAHU, "sahu_corrected")
ctrl = load(CTRL, "gosai_ctrl_neg_real")

print(f"  dinuc: {len(dinuc):,}  mean={dinuc['K562_log2FC'].mean():+.3f}")
print(f"  sahu:  {len(sahu):,}  mean={sahu['K562_log2FC'].mean():+.3f}")
print(f"  ctrl:  {len(ctrl):,}  mean={ctrl['K562_log2FC'].mean():+.3f}")

# Subsample: 50% dinuc, 30% Sahu, 20% ctrl_neg by COUNT
target_total = 100_000  # generous pool, neg_fraction picks subset
n_dinuc = int(target_total * 0.5)
n_sahu = int(target_total * 0.3)
n_ctrl = int(target_total * 0.2)

# Cap at available
n_dinuc = min(n_dinuc, len(dinuc))
n_sahu = min(n_sahu, len(sahu))
n_ctrl = min(n_ctrl, len(ctrl))
# If ctrl_neg is small (only 503), upsample to ~5000 by repetition
if n_ctrl < 5000:
    ctrl = ctrl.sample(5000, replace=True, random_state=42)
    n_ctrl = len(ctrl)

dinuc_sub = dinuc.sample(n_dinuc, random_state=42)
sahu_sub = sahu.sample(n_sahu, random_state=42)
ctrl_sub = ctrl.sample(n_ctrl, random_state=42) if n_ctrl != len(ctrl) else ctrl

combined = pd.concat([dinuc_sub, sahu_sub, ctrl_sub], ignore_index=True)
combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

out_path = OUT_DIR / "dinuc_sahu_ctrl_combined.tsv"
combined.to_csv(out_path, sep="\t", index=False)
print(f"\nSaved {out_path}")
print(f"  Total: {len(combined):,} sequences")
print(f"  Mean K562_log2FC: {combined['K562_log2FC'].mean():+.3f}")
print(f"  Std K562_log2FC: {combined['K562_log2FC'].std():.3f}")
print(f"  Source breakdown: {combined['source'].value_counts().to_dict()}")
