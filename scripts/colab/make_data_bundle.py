"""Bundle the D=20k LegNet training data + best model for the Peter Colab.

Reads:
  outputs/oracle_pseudolabels_k562_ag_s2_refalt/pool/{train,val,test}.parquet
  outputs/oracle_pseudolabels_k562_ag_s2_refalt/*_oracle_labels.npz
  results/preflight/colab_d20k_legnet/best.pt + result.json

Writes:
  scripts/colab/k562_d20k/train_d20k.parquet  (20k subsample with sequence + label)
  scripts/colab/k562_d20k/val.parquet         (all val rows; label = K562_log2FC real)
  scripts/colab/k562_d20k/test.parquet        (all test rows; label = K562_log2FC real)
  scripts/colab/k562_d20k/best_model.pt       (the LegNet checkpoint)
  scripts/colab/k562_d20k/README.md           (notes on labels + splits)
  scripts/colab/bundle_d20k.tar.gz            (gzipped tarball of the directory)

Notes on label choices (for Peter):
  - train uses oracle (OOF) labels — what the HP-search runs use, less noisy
  - val/test use REAL K562_log2FC — what eval should be against
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ORACLE = REPO / "outputs/oracle_pseudolabels_k562_ag_s2_refalt"
MODEL_DIR = REPO / "results/preflight/colab_d20k_legnet"
OUT_DIR = REPO / "scripts/colab/k562_d20k"
BUNDLE = REPO / "scripts/colab/bundle_d20k.tar.gz"

D = 20000
SEED = 42


def main():
    assert ORACLE.exists(), f"Oracle dir missing: {ORACLE}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- train ----------
    train_pool = pd.read_parquet(ORACLE / "pool/train.parquet")
    train_npz = np.load(ORACLE / "train_oracle_labels.npz")
    print(f"Full train pool: {len(train_pool):,} sequences")

    # AG oracle OOF labels for train
    train_pool = train_pool.copy()
    train_pool["label"] = train_npz["oof_oracle"].astype(np.float32)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(train_pool), size=D, replace=False)
    train_subset = train_pool.iloc[idx][["sequence", "label"]].reset_index(drop=True)
    train_subset.to_parquet(OUT_DIR / "train_d20k.parquet", index=False)
    print(
        f"  wrote train_d20k.parquet ({len(train_subset):,} rows, "
        f"label_mean={train_subset['label'].mean():.3f})"
    )

    # ---------- val + test ----------
    # Use REAL labels (K562_log2FC column from the parquet, NOT the oracle ensemble mean)
    for split in ["val", "test"]:
        df = pd.read_parquet(ORACLE / f"pool/{split}.parquet").copy()
        # Find the real-label column. Common names:
        for col in ["K562_log2FC", "log2FC", "k562_log2FC", "k562"]:
            if col in df.columns:
                df["label"] = df[col].astype(np.float32)
                src_col = col
                break
        else:
            # Fallback to oracle mean if real column missing
            npz = np.load(ORACLE / f"{split}_oracle_labels.npz")
            df["label"] = npz["oracle_mean"].astype(np.float32)
            src_col = "oracle_mean (fallback)"
        out_df = df[["sequence", "label"]].reset_index(drop=True)
        out_df.to_parquet(OUT_DIR / f"{split}.parquet", index=False)
        print(f"  wrote {split}.parquet ({len(out_df):,} rows, label source: {src_col})")

    # ---------- model ----------
    best_pt = MODEL_DIR / "best.pt"
    if not best_pt.exists():
        print(f"  WARN: {best_pt} not found — bundle will skip model.pt")
    else:
        # Convert run_single.py's checkpoint format to one with model_info
        import torch

        ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
        hp = ckpt.get("hp", {})
        out_ckpt = {
            "model_state_dict": ckpt["state_dict"],
            "block_sizes": hp.get("block_sizes", [256, 256, 128, 128, 64, 64, 32, 32]),
            "ks": hp.get("ks", 5),
            "dropout": hp.get("dropout", 0.1),
            "epoch": ckpt.get("epoch"),
            "model_info": {
                "arch": "legnet",
                "trained_d": D,
                "hp": hp,
            },
        }
        torch.save(out_ckpt, OUT_DIR / "best_model.pt")
        print(f"  wrote best_model.pt (epoch={ckpt.get('epoch')})")

        # Save the result.json too for context
        result_json = MODEL_DIR / "result.json"
        if result_json.exists():
            shutil.copy(result_json, OUT_DIR / "result.json")
            print("  wrote result.json")

    # ---------- README ----------
    (OUT_DIR / "README.md").write_text(
        f"""# K562 D=20k MPRA Subset

Bundled for the Peter Colab notebook (May 11, 2026).

## Files
- `train_d20k.parquet` — {D:,} train sequences (AG oracle OOF labels, seed={SEED})
- `val.parquet` — held-out validation (chromosomes 19/21/X, real K562_log2FC labels)
- `test.parquet` — held-out test (chromosomes 7/13, real K562_log2FC labels)
- `best_model.pt` — LegNet trained on `train_d20k.parquet`
- `result.json` — training summary (best epoch, val_mse, gpu_hrs, etc.)

## Label sources
- Train: AG-oracle out-of-fold predictions (denoised pseudolabels — what our HP
  search uses internally; smoother loss surface than real labels at small D)
- Val/test: real K562_log2FC measurements (so eval numbers are interpretable)

## Sampling
Train is a deterministic uniform random subsample (`np.random.default_rng({SEED}).choice`)
from the chromosome-split train pool.
"""
    )

    # ---------- tar ----------
    if BUNDLE.exists():
        BUNDLE.unlink()
    subprocess.run(
        ["tar", "-czf", str(BUNDLE), "-C", str(OUT_DIR.parent), OUT_DIR.name],
        check=True,
    )
    print(f"\nBundle: {BUNDLE}  ({BUNDLE.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
