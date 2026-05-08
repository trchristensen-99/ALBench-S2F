"""Create a TSV of Gosai's ctrl_neg sequences with their REAL measured
K562_log2FC values, suitable as neg-aug source for direct calibration
on real episomal-MPRA-measured non-regulatory DNA.

Output: data/synthetic_negatives_calibration/gosai_ctrl_neg_calibration.tsv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "data/k562/gosai_ctrl_neg.parquet"
OUT_DIR = REPO / "data/synthetic_negatives_calibration"


def main():
    df = pd.read_parquet(SOURCE)
    print(f"Loaded {len(df):,} ctrl_neg sequences with REAL K562 measurements")
    print(f"  K562_log2FC: mean={df['K562_log2FC'].mean():+.3f}  std={df['K562_log2FC'].std():.3f}")
    out_df = df[["sequence", "K562_log2FC"]].copy()
    out_df["category"] = "gosai_ctrl_neg_real"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "gosai_ctrl_neg_calibration.tsv"
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved {out_path}")
    print(f"  These sequences will train the model with their REAL measured")
    print(f"  K562 episomal MPRA values (mean +0.27, NOT a synthetic distribution)")


if __name__ == "__main__":
    main()
