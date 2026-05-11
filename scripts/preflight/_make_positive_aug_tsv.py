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
    # Load the Gosai dataset from the raw text file
    full_path = REPO / "data/k562/DATA-Table_S2__MPRA_dataset.txt"
    if not full_path.exists():
        print(f"Gosai full dataset not found at {full_path}")
        return
    df = pd.read_csv(full_path, sep="\t")
    # Find K562 column (may be named differently)
    k562_col = None
    for c in df.columns:
        if "K562" in c and ("log2FC" in c or "_log2" in c.lower()):
            k562_col = c
            break
    if k562_col is None:
        print(f"No K562 log2FC column found; available cols: {list(df.columns)[:15]}")
        return
    if k562_col != "K562_log2FC":
        df = df.rename(columns={k562_col: "K562_log2FC"})
    df = df.dropna(subset=["K562_log2FC"])
    seq_col = "sequence" if "sequence" in df.columns else next(c for c in df.columns if "seq" in c.lower())
    if seq_col != "sequence":
        df = df.rename(columns={seq_col: "sequence"})
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
