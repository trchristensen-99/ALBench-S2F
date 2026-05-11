"""Create POSITIVE-augmentation TSVs from high-activity Gosai sequences.

Premise: oracles under-predict high-activity sequences (regression-to-mean).
Adding high-activity sequences with their REAL labels as supplementary
training data should counteract this compression.

Outputs:
  data/positive_augmentation/gosai_top5pct.tsv      (top 5% of K562_log2FC)
  data/positive_augmentation/gosai_top10pct.tsv     (top 10%)
  data/positive_augmentation/gosai_top25pct.tsv     (top 25%)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/positive_augmentation"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    # Use full Gosai dataset (from k562_full.py)
    from data.k562_full import K562_FULL_DATASET_PATH
    full_path = Path(K562_FULL_DATASET_PATH)
    if not full_path.exists():
        print(f"Gosai full dataset not found at {full_path}")
        return
    df = pd.read_parquet(full_path)
    print(f"Loaded {len(df):,} Gosai sequences; K562_log2FC mean={df['K562_log2FC'].mean():+.3f}")

    for pct in (5, 10, 25):
        thresh = df["K562_log2FC"].quantile(1 - pct / 100)
        high = df[df["K562_log2FC"] >= thresh].copy()
        print(f"\nTop {pct}% (threshold log2FC >= {thresh:+.3f}):")
        print(f"  n={len(high):,}  mean K562_log2FC = {high['K562_log2FC'].mean():+.3f}")
        out = OUT / f"gosai_top{pct}pct.tsv"
        # Use neg-aug compatible columns: sequence, K562_log2FC, category
        out_df = high[["sequence", "K562_log2FC"]].copy()
        out_df["category"] = f"gosai_top_{pct}pct"
        out_df.to_csv(out, sep="\t", index=False)
        print(f"  Saved {out}")


if __name__ == "__main__":
    main()
