"""Train a single LegNet model on the chr-split K562 train pool, with either real
or oracle (AG_S2) labels. Used for the AG vs LegNet × real vs oracle scaling-law
replication.

Picks a single fixed HP config (eff + muon + bs=256 + lr=0.003 + dropout=0.15) —
shown to be robust across reservoirs and D values.

The --seed parameter controls BOTH the random subsample of N sequences from the
chr-train pool AND the model-init seed (so 3 different seeds = 3 different
training-data realizations × 3 model inits).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scaling_hp_search import (
    REPO,
    HPConfig,
    build_block_sizes,
    load_chr_test_genomic,
    train_one_model,
)
from test_set_guards import assert_battery_provenance, assert_mono_snv

HP_FIXED = dict(
    lr=0.003,
    batch_size=256,
    conv_dropout=0.15,
    dense_dropout=0.15,
    n_layers=8,
    width_base=80,
    width_jitter=None,
    block_class="eff",
    ks=7,
    pct_start=0.3,
    optimizer="muon",
    weight_decay=1e-6,
    use_shift_aug=True,
    shift_max=20,
    use_evoaug=False,
)


def subsample_chr_train(N: int, seed: int, label_source: str):
    """Load chr_train pool, filter NaN labels, return N random subsample.

    Uses real_labels or oracle_labels depending on label_source.
    """
    z = np.load(REPO / "outputs/chr_split_cache/chr_train_ref_only.npz", allow_pickle=True)
    seqs = z["sequences"]
    labels = z[f"{label_source}_labels"].astype(np.float32)
    finite = np.isfinite(labels)
    seqs = seqs[finite]
    labels = labels[finite]
    rng = np.random.default_rng(seed)
    if N > len(seqs):
        N = len(seqs)
    idx = rng.choice(len(seqs), size=N, replace=False)
    return seqs[idx], labels[idx]


def load_chr_val(label_source: str):
    z = np.load(REPO / "outputs/chr_split_cache/chr_val_ref_only.npz", allow_pickle=True)
    seqs = z["sequences"]
    labels = z[f"{label_source}_labels"].astype(np.float32)
    finite = np.isfinite(labels)
    return seqs[finite], labels[finite]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--label_source", required=True, choices=["oracle", "real"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--early_stop_patience", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "model.npz").exists():
        print(f"  already done: {out}")
        return

    train_seqs, train_labels = subsample_chr_train(args.N, args.seed, args.label_source)
    val_seqs, val_labels = load_chr_val(args.label_source)
    test_seqs, test_oracle, test_real = load_chr_test_genomic()
    # SNV + OOD extra test sets — gate on canonical battery provenance first so a
    # stale / differently-scored battery can never be silently trained against.
    battery_dir = REPO / "data/k562/test_sets_ag_s2_chrsplit"
    assert_battery_provenance(battery_dir)
    snv = np.load(battery_dir / "snv_oracle.npz", allow_pickle=True)
    assert_mono_snv(snv, battery_dir / "snv_oracle.npz")
    snv_ref_seqs = list(snv["ref_sequences"])
    snv_alt_seqs = list(snv["alt_sequences"])
    snv_delta_oracle = snv["delta_mean"].astype(np.float32)
    snv_delta_real = snv["true_delta"].astype(np.float32)
    ood = np.load(battery_dir / "ood_oracle.npz", allow_pickle=True)
    ood_seqs = list(ood["sequences"])
    ood_oracle_y = ood["oracle_mean"].astype(np.float32)
    ood_real_y = ood["true_label"].astype(np.float32)

    print(f"  N={args.N}  seed={args.seed}  label_source={args.label_source}")
    print(
        f"  train={len(train_seqs)}  val={len(val_seqs)}  test={len(test_seqs)}  snv={len(snv_ref_seqs)}  ood={len(ood_seqs)}"
    )

    hp = HPConfig(**HP_FIXED, seed=args.seed)
    # Pass extra test sets so train_one_model produces per-set predictions
    extra_test_sets = {
        "snv_ref": (np.array(snv_ref_seqs), snv["ref_mean"].astype(np.float32)),
        "snv_alt": (np.array(snv_alt_seqs), snv["alt_mean"].astype(np.float32)),
        "ood": (np.array(ood_seqs), ood_oracle_y),
    }
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

    from scipy.stats import pearsonr

    def m_against(pred, y):
        mm = np.isfinite(pred) & np.isfinite(y)
        if mm.sum() < 8:
            return None
        r = float(pearsonr(pred[mm], y[mm])[0])
        mse = float(((pred[mm] - y[mm]) ** 2).mean())
        return {"pearson_r": r, "mse": mse, "n": int(mm.sum())}

    pred = result["test_pred"]
    pred_ood = result["test_pred_ood"]
    pred_snv_ref = result["test_pred_snv_ref"]
    pred_snv_alt = result["test_pred_snv_alt"]
    pred_snv_delta = pred_snv_alt - pred_snv_ref

    summary = {
        "N": args.N,
        "seed": args.seed,
        "label_source": args.label_source,
        "hp": HP_FIXED,
        "val_pearson": result["val_pearson"],
        "val_mse": result["val_mse"],
        "test_vs_oracle": m_against(pred, test_oracle),
        "test_vs_real": m_against(pred, test_real),
        "ood_vs_oracle": m_against(pred_ood, ood_oracle_y),
        "ood_vs_real": m_against(pred_ood, ood_real_y),
        "snv_delta_vs_oracle": m_against(pred_snv_delta, snv_delta_oracle),
        "snv_delta_vs_real": m_against(pred_snv_delta, snv_delta_real),
    }
    np.savez(
        out / "model.npz",
        val_pred=result["val_pred"].astype(np.float32),
        test_pred=pred.astype(np.float32),
        test_pred_ood=pred_ood.astype(np.float32),
        test_pred_snv_ref=pred_snv_ref.astype(np.float32),
        test_pred_snv_alt=pred_snv_alt.astype(np.float32),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  saved: {out}  val_pearson={result['val_pearson']:.4f}")


if __name__ == "__main__":
    main()
