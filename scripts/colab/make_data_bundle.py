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

# Checkpoints to bundle. Each entry is (display_name, source_dir, training_d).
# Empty/missing source_dirs are skipped — bundle tolerates partial.
CHECKPOINTS: list[tuple[str, Path, int]] = [
    (
        "legnet_published_default",
        REPO / "results/preflight/shootout_d20k_legnet/legnet_published_default",
        20000,
    ),
    (
        "legnet_optimized_default",
        REPO / "results/preflight/shootout_d20k_legnet/current_colab_default",
        20000,
    ),
    (
        "legnet_wider_arch",
        REPO / "results/preflight/shootout_d20k_legnet/wider_arch",
        20000,
    ),
    (
        "legnet_with_shift_aug",
        REPO / "results/preflight/shootout_d20k_legnet/with_shift_aug",
        20000,
    ),
]


def main():
    import torch

    assert ORACLE.exists(), f"Oracle dir missing: {ORACLE}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "models").mkdir(exist_ok=True)

    # ---------- train (FULL pool — notebook subsamples to any D) ----------
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

    # ---------- checkpoints ----------
    manifest = []
    for name, src_dir, training_d in CHECKPOINTS:
        best_pt = src_dir / "best.pt"
        result_json = src_dir / "result.json"
        if not best_pt.exists():
            print(f"  skip {name}: {best_pt} not found")
            continue
        ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
        hp = ckpt.get("hp", {})
        result = json.loads(result_json.read_text()) if result_json.exists() else {}
        out_ckpt = {
            "model_state_dict": ckpt["state_dict"],
            "block_sizes": hp.get("block_sizes", [256, 256, 128, 128, 64, 64, 32, 32]),
            "ks": hp.get("ks", 5),
            "block_class": "eff",  # All current checkpoints use EffBlock
            "conv_dropout": hp.get("dropout", 0.0),
            "dense_dims": [],
            "dense_dropout": 0.0,
            "epoch": ckpt.get("epoch"),
            "training_d": training_d,
            "hp": hp,
        }
        out_path = OUT_DIR / "models" / f"{name}.pt"
        torch.save(out_ckpt, out_path)
        meta = {
            "name": name,
            "file": f"models/{name}.pt",
            "training_d": training_d,
            "best_val_mse": result.get("best_val_mse"),
            "test_mse_at_best_val": result.get("test_mse_at_best_val"),
            "best_epoch": result.get("best_epoch"),
            "n_params": result.get("n_params"),
            "hp_summary": {
                "lr": hp.get("lr"),
                "batch_size": hp.get("batch_size"),
                "weight_decay": hp.get("weight_decay"),
                "dropout": hp.get("dropout"),
                "block_sizes": hp.get("block_sizes"),
            },
        }
        manifest.append(meta)
        print(f"  wrote {out_path.name}  (val_mse={result.get('best_val_mse', float('nan')):.4f})")

    (OUT_DIR / "models" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n  manifest: {len(manifest)} checkpoints bundled")

    # ---------- README ----------
    ckpt_lines = (
        "\n".join(
            f"- `models/{m['name']}.pt` — val_mse={m['best_val_mse']:.4f}"
            f" (trained at D={m['training_d']:,})"
            for m in manifest
        )
        or "_(no checkpoints bundled yet)_"
    )
    (OUT_DIR / "README.md").write_text(
        f"""# K562 MPRA training/eval bundle

## Files
- `train_full.parquet` — full chromosome-split train pool ({len(train_full):,} sequences,
  AG-oracle OOF pseudolabels). Subsample to any D in the notebook.
- `val.parquet` — held-out validation (chr 19/21/X, real K562_log2FC labels).
- `test.parquet` — held-out test (chr 7/13, real K562_log2FC labels).
- `models/manifest.json` — list of bundled checkpoints with HPs + val metrics.
- `models/*.pt` — LegNet checkpoints; see README in notebook for usage.

## Bundled checkpoints
{ckpt_lines}

## Label sources
- Train: AG-oracle out-of-fold predictions (denoised pseudolabels).
- Val/test: real K562_log2FC measurements.

## Subsampling
Use `np.random.default_rng({SEED}).choice(N, D, replace=False)` for deterministic
D-sized training subsets.
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
