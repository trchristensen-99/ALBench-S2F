"""Train one LegNet model with a fixed HP config on a (reservoir, D, seed) cell.

Used by the focused experiment to verify the random_d1M finding across
3 robust HP configs × 2 reservoir-sampling seeds × 6 reservoirs × 7 D values.

Outputs land in outputs/focused_train/k562_{reservoir}_d{D}_seed{S}/config_{A,C,D}/
with the same npz schema (val_pred + test_pred_*) as the main HP-search pipeline,
so analyze_cell.py can ensemble them downstream.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scaling_hp_search import (
    REPO,
    HPConfig,
    build_block_sizes,
    load_all_test_sets,
    load_chr_test_genomic,
    load_chr_train_pool,
    train_one_model,
)

# Fixed HP recipes — block_class + optimizer + bs + lr + dropout
CONFIGS = {
    "A": dict(
        block_class="eff",
        optimizer="muon",
        batch_size=256,
        lr=0.003,
        conv_dropout=0.15,
        dense_dropout=0.15,
    ),
    "C": dict(
        block_class="plain",
        optimizer="muon",
        batch_size=1024,
        lr=0.002,
        conv_dropout=0.05,
        dense_dropout=0.05,
    ),
    "D": dict(
        block_class="ag",
        optimizer="adam",
        batch_size=32,
        lr=0.002,
        conv_dropout=0.20,
        dense_dropout=0.20,
    ),
    "E": dict(
        block_class="ag",
        optimizer="adamw",
        batch_size=256,
        lr=0.002,
        conv_dropout=0.15,
        dense_dropout=0.15,
    ),
}


def build_hp(config_id: str, seed: int) -> HPConfig:
    base = CONFIGS[config_id]
    return HPConfig(
        lr=base["lr"],
        batch_size=base["batch_size"],
        conv_dropout=base["conv_dropout"],
        dense_dropout=base["dense_dropout"],
        n_layers=8,
        width_base=80,
        width_jitter=None,
        block_class=base["block_class"],
        ks=7,
        pct_start=0.3,
        optimizer=base["optimizer"],
        weight_decay=1e-6,
        use_shift_aug=True,
        shift_max=20,
        use_evoaug=False,
        seed=seed,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reservoir", required=True)
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument(
        "--reservoir_cache",
        default=None,
        help="Path to pre-built reservoir cache npz (skip for genomic)",
    )
    ap.add_argument("--config_id", required=True, choices=["A", "C", "D", "E"])
    ap.add_argument("--seed", type=int, required=True, help="Reservoir+train seed")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--early_stop_patience", type=int, default=5)
    ap.add_argument(
        "--chr_val",
        action="store_true",
        help="Use chr19+21+X validation (for genomic-based reservoirs)",
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Skip if already done
    npz_done = out / "model.npz"
    if npz_done.exists():
        print(f"  already done: {npz_done}")
        return

    # Adjust batch_size if D < cfg_bs: pick the largest power-of-2 ≤ D//2 (≥32)
    cfg_bs = CONFIGS[args.config_id]["batch_size"]
    if args.D < cfg_bs:
        new_bs = 32
        for bs in [1024, 512, 256, 128, 64, 32]:
            if bs <= args.D // 2:
                new_bs = bs
                break
        print(f"  Adjusted batch_size: {cfg_bs} → {new_bs} (D={args.D} too small for original)")
        CONFIGS[args.config_id]["batch_size"] = new_bs

    # Load train+val with the specified reservoir
    if args.reservoir == "genomic":
        train_seqs, train_labels, val_seqs, val_labels = load_chr_train_pool(
            D=args.D,
            ref_only=True,
            chr_val=True,
            seed=args.seed,
        )
    else:
        # Load from reservoir cache
        z = np.load(args.reservoir_cache, allow_pickle=True)
        train_seqs = z["sequences"]
        train_labels = z["oracle_labels"].astype(np.float32)
        finite = np.isfinite(train_labels)
        train_seqs = train_seqs[finite]
        train_labels = train_labels[finite]
        # Use chr_val for val labels (same as the main full_sweep)
        _, _, val_seqs, val_labels = load_chr_train_pool(
            D=args.D,
            ref_only=True,
            chr_val=args.chr_val,
            seed=args.seed,
        )

    test_seqs, test_oracle, test_real = load_chr_test_genomic()
    extra_test_sets = load_all_test_sets()

    hp = build_hp(args.config_id, args.seed)
    print(f"  config={args.config_id} hp={hp}")

    result = train_one_model(
        hp=hp,
        train_seqs=train_seqs,
        train_labels=train_labels,
        val_seqs=val_seqs,
        val_labels=val_labels,
        test_seqs=test_seqs,
        epochs=args.epochs,
        early_stopping_patience=args.early_stop_patience,
        extra_test_sets=extra_test_sets,
    )

    # Save predictions in the same format as the main pipeline
    save_data = {
        "val_pred": result["val_pred"].astype(np.float32),
        "test_pred": result["test_pred"].astype(np.float32),
        "test_pred_genomic": result["test_pred"].astype(np.float32),
    }
    # train_one_model stores extra test predictions as flat test_pred_<set_name> keys
    for k, v in result.items():
        if k.startswith("test_pred_") and k != "test_pred_genomic":
            save_data[k] = v.astype(np.float32)
    np.savez(out / "model.npz", **save_data)

    # Save labels (same as the main pipeline's labels.npz)
    labels_data = {
        "val_labels": val_labels.astype(np.float32),
        "test_oracle": test_oracle.astype(np.float32),
        "test_real_label": test_real.astype(np.float32),
    }
    for set_name, (seqs, oracle) in extra_test_sets.items():
        labels_data[f"oracle_{set_name}"] = oracle.astype(np.float32)
    np.savez(out / "labels.npz", **labels_data)

    # Save meta
    meta = {
        "config_id": args.config_id,
        "reservoir": args.reservoir,
        "D": args.D,
        "seed": args.seed,
        "hp_config": CONFIGS[args.config_id],
        "val_pearson": result["val_pearson"],
        "test_pearson": result.get("test_pearson"),
        "epochs_run": result.get("epochs_run"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  saved: {out} val_pearson={result['val_pearson']:.4f}")


if __name__ == "__main__":
    main()
