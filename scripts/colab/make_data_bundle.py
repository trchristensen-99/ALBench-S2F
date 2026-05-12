"""Bundle the full K562 train pool + val + test + multiple model checkpoints.

Reads:
  outputs/oracle_pseudolabels_k562_ag_s2_refalt/pool/{train,val,test}.parquet
  outputs/oracle_pseudolabels_k562_ag_s2_refalt/*_oracle_labels.npz
  results/preflight/shootout_d20k_legnet/*/best.pt + result.json
       (any LegNet checkpoint dirs to bundle, see CHECKPOINTS list)

Writes (under scripts/colab/k562_data/):
  train_full.parquet      — full train pool (~617k rows, oracle OOF labels)
  val.parquet             — chr 19/21/X, real K562_log2FC labels
  test.parquet            — chr 7/13, real K562_log2FC labels
  models/<name>.pt        — one per checkpoint in CHECKPOINTS
  models/manifest.json    — list of available models with metadata
  README.md
scripts/colab/bundle_d20k.tar.gz  — gzipped tarball of the directory
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ORACLE = REPO / "outputs/oracle_pseudolabels_k562_ag_s2_refalt"
OUT_DIR = REPO / "scripts/colab/k562_data"
BUNDLE = REPO / "scripts/colab/bundle_d20k.tar.gz"

SEED = 42

# Single reference checkpoint to bundle. Best D=20k LegNet across all overnight
# searches — winner is shootout_d20k_fillin/low_dropout_aggressive
# (val=0.6234, test=0.5117 vs prior published_default 0.6290/0.5189).
REFERENCE_CHECKPOINT = (
    "legnet_d20k_low_dropout_aggressive",
    REPO / "results/preflight/shootout_d20k_fillin/low_dropout_aggressive",
    20000,
    "ag_oracle",
)


def main():
    import torch

    assert ORACLE.exists(), f"Oracle dir missing: {ORACLE}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "models").mkdir(exist_ok=True)

    # ---------- train: full pool + a fixed D=20k subset ----------
    # train_d20k.parquet is the "default" file the notebook loads — small, fast.
    # train_full.parquet is the full pool; the notebook has an optional cell that
    # subsamples it to any D the user picks.
    train_pool = pd.read_parquet(ORACLE / "pool/train.parquet")
    train_npz = np.load(ORACLE / "train_oracle_labels.npz")
    print(f"Full train pool: {len(train_pool):,} sequences")

    train_full = train_pool[["sequence"]].copy()
    train_full["label"] = train_npz["oof_oracle"].astype(np.float32)
    train_full.to_parquet(OUT_DIR / "train_full.parquet", index=False)
    print(
        f"  wrote train_full.parquet ({len(train_full):,} rows, "
        f"label_mean={train_full['label'].mean():.3f})"
    )

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(train_full), size=20000, replace=False)
    train_d20k = train_full.iloc[idx].reset_index(drop=True)
    train_d20k.to_parquet(OUT_DIR / "train_d20k.parquet", index=False)
    print(f"  wrote train_d20k.parquet ({len(train_d20k):,} rows, seed={SEED})")

    # ---------- val + test (real labels) ----------
    for split in ["val", "test"]:
        df = pd.read_parquet(ORACLE / f"pool/{split}.parquet").copy()
        for col in ["K562_log2FC", "log2FC", "k562_log2FC", "k562"]:
            if col in df.columns:
                df["label"] = df[col].astype(np.float32)
                src_col = col
                break
        else:
            npz = np.load(ORACLE / f"{split}_oracle_labels.npz")
            df["label"] = npz["oracle_mean"].astype(np.float32)
            src_col = "oracle_mean (fallback)"
        df[["sequence", "label"]].reset_index(drop=True).to_parquet(
            OUT_DIR / f"{split}.parquet", index=False
        )
        print(f"  wrote {split}.parquet ({len(df):,} rows, label source: {src_col})")

    # ---------- single reference checkpoint ----------
    name, src_dir, training_d, label_src = REFERENCE_CHECKPOINT
    best_pt = src_dir / "best.pt"
    result_json = src_dir / "result.json"
    ckpt_line = "_(no reference model bundled yet)_"
    if not best_pt.exists():
        print(f"  WARN: reference {best_pt} not found — bundle ships without a reference model")
    else:
        ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
        hp = ckpt.get("hp", {})
        result = json.loads(result_json.read_text()) if result_json.exists() else {}
        out_ckpt = {
            "model_state_dict": ckpt["state_dict"],
            "block_sizes": hp.get("block_sizes", [256, 256, 128, 128, 64, 64, 32, 32]),
            "ks": hp.get("ks", 5),
            "block_class": "eff",
            "conv_dropout": hp.get("dropout", 0.0),
            "dense_dims": [],
            "dense_dropout": 0.0,
            "epoch": ckpt.get("epoch"),
            "training_d": training_d,
            "training_label_source": label_src,
            "hp": hp,
            "meta": {
                "best_val_mse": result.get("best_val_mse"),
                "test_mse_at_best_val": result.get("test_mse_at_best_val"),
                "best_epoch": result.get("best_epoch"),
                "n_params": result.get("n_params"),
            },
        }
        out_path = OUT_DIR / "best_model.pt"
        torch.save(out_ckpt, out_path)
        v = result.get("best_val_mse", float("nan"))
        t = result.get("test_mse_at_best_val", float("nan"))
        print(
            f"  wrote best_model.pt  (D={training_d:,}, labels={label_src}, val_mse={v:.4f}, test_mse={t:.4f})"
        )
        ckpt_line = f"- `best_model.pt` — D={training_d:,}, labels={label_src}, val_mse={v:.4f}, test_mse={t:.4f}"

    # ---------- README ----------
    (OUT_DIR / "README.md").write_text(
        f"""# K562 MPRA training/eval bundle

## Files
- `train_d20k.parquet` — 20,000-sequence subset of the train pool (seed={SEED},
  AG-oracle OOF labels). Loaded by the notebook by default.
- `train_full.parquet` — full chromosome-split train pool ({len(train_full):,}
  sequences). Optional cell at the bottom of the notebook subsamples this to
  any D.
- `val.parquet` — held-out validation (chr 19/21/X, real K562_log2FC labels).
- `test.parquet` — held-out test (chr 7/13, real K562_log2FC labels).
- `best_model.pt` — reference LegNet checkpoint.

## Bundled model
{ckpt_line}

## Label sources
- Train: AG-oracle out-of-fold predictions (denoised pseudolabels).
- Val/test: real K562_log2FC measurements.
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
