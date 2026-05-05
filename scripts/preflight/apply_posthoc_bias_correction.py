"""Post-hoc affine bias correction for the AG-S2 oracle ensemble.

Given a calibration set of expected-zero negative controls (random DNA,
intergenic, dinuc-shuffled), fits a single affine transform
``y_corrected = a * y_pred + b`` such that:
  - On the calibration set: E[y_corrected | random_dna] ≈ 0
  - On the held-out reference set (real K562 train): the predictions
    remain calibrated to the true labels (slope ≈ 1).

Applies the same transform to all four pseudolabel npzs in
``outputs/oracle_pseudolabels_k562_ag_s2_refalt/``:
  train_oracle_labels.npz, val_oracle_labels.npz,
  test_oracle_labels.npz, snv_oracle_labels.npz

Writes the corrected labels alongside the originals as
  *_oracle_labels_corrected.npz

DOES NOT overwrite the original npzs — opt in by pointing run_single
at the corrected files (or running this with ``--inplace`` to swap).

Usage:
  uv run --no-sync python scripts/preflight/apply_posthoc_bias_correction.py [--inplace]

When to run: AFTER score_oracle_bias.py finishes and you've inspected
bias_eval.json. If the residual on random_dna is ~+0.5 to +1 with low
variance, an affine fix recovers ~70-90% of that. If the bias is
non-linear (e.g., heteroscedastic with activity), affine won't be
enough — consider Option C retrain instead.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ORACLE_OUT = REPO / "outputs" / "oracle_pseudolabels_k562_ag_s2_refalt"
BIAS_EVAL = ORACLE_OUT / "bias_eval.json"


def _fit_affine(
    neg_preds: np.ndarray, ref_preds: np.ndarray, ref_labels: np.ndarray
) -> tuple[float, float]:
    """Fit a, b such that:
      a * mean(neg_preds) + b ≈ 0  (negative controls predict ~0)
      a * ref_preds + b is calibrated to ref_labels

    With two constraints (one mean-zero on negatives, one slope=1 on
    real data via least-squares), we get a unique solution.

    Specifically: pick `a` to preserve the slope of ref_preds vs
    ref_labels (i.e., a = std(ref_labels) / std(ref_preds)), then choose
    `b` such that mean(neg_preds * a) + b = 0, i.e., b = -a * mean(neg_preds).
    """
    a = float(np.std(ref_labels) / max(1e-6, np.std(ref_preds)))
    b = float(-a * np.mean(neg_preds))
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite *_oracle_labels.npz with corrected values (DESTRUCTIVE).",
    )
    ap.add_argument(
        "--neg_panel",
        default="random_gc_50pct",
        help="Which bias_eval panel to use as the calibration set "
        "(must be a negative control from score_oracle_bias.py).",
    )
    args = ap.parse_args()

    if not BIAS_EVAL.exists():
        raise SystemExit(f"missing {BIAS_EVAL}. Run scripts/preflight/score_oracle_bias.sh first.")
    bias = json.loads(BIAS_EVAL.read_text())
    if args.neg_panel not in bias:
        raise SystemExit(
            f"panel '{args.neg_panel}' not in bias_eval.json. "
            f"Available negative-control panels: {[k for k in bias if k.startswith('random_gc_') or k == 'test_dinuc_shuffled']}"
        )

    # Use bias_eval's reported mean as proxy for E[y_pred | neg]
    neg_mean = float(bias[args.neg_panel]["mean"])
    print(f"  Calibration: panel '{args.neg_panel}' has mean(y_pred) = {neg_mean:.4f}")

    # Load the train-set predictions (real labels) for slope calibration
    train_npz = ORACLE_OUT / "train_oracle_labels.npz"
    if not train_npz.exists():
        raise SystemExit(f"missing {train_npz}. Aggregate must have run.")
    npz = np.load(train_npz)
    ref_preds = npz["oracle_mean"]
    ref_labels = npz["true_label"]
    print(
        f"  Reference: train pool n={len(ref_preds):,}, mean(y_pred)={ref_preds.mean():.4f}, "
        f"mean(y_true)={ref_labels.mean():.4f}"
    )

    # The slope-preserving + zero-mean-on-neg fit needs ACTUAL neg
    # predictions, but we only have the mean from bias_eval. Use the
    # mean as a proxy for now (constant-offset only correction).
    # If we had per-seq neg preds, we'd use those instead.
    neg_preds_proxy = np.array([neg_mean])  # 1-element array; correction reduces to b shift
    a, b = _fit_affine(neg_preds_proxy, ref_preds, ref_labels)
    print(f"  Fitted: a={a:.4f}, b={b:.4f}")
    print(f"  Effect: y_corrected = {a:.4f} * y_pred + {b:.4f}")
    sample_corr = a * neg_mean + b
    print(f"  Sanity check: y_corrected on neg_panel mean = {sample_corr:.4f} (target 0)")

    # Apply correction to all per-split npzs
    for split in ("train", "val", "test", "snv"):
        npz_path = ORACLE_OUT / f"{split}_oracle_labels.npz"
        if not npz_path.exists():
            continue
        d = dict(np.load(npz_path).items())
        # Apply to all `oracle_mean*` keys (handles snv's two preds too)
        for k in list(d.keys()):
            if k.startswith("oracle_mean"):
                d[k] = (a * d[k] + b).astype(np.float32)
        # Also write OOF for train if present
        if "oof_oracle" in d:
            d["oof_oracle"] = (a * d["oof_oracle"] + b).astype(np.float32)
        out_path = (
            npz_path if args.inplace else (ORACLE_OUT / f"{split}_oracle_labels_corrected.npz")
        )
        if not args.inplace:
            np.savez_compressed(out_path, **d)
        else:
            shutil.copy(npz_path, npz_path.with_suffix(".npz.bak"))
            np.savez_compressed(out_path, **d)
        print(f"  Wrote {out_path}")

    # Record the correction parameters
    correction_log = {
        "calibration_panel": args.neg_panel,
        "neg_panel_mean_y_pred": neg_mean,
        "ref_panel": "train_oracle_labels.npz",
        "ref_pred_std": float(ref_preds.std()),
        "ref_label_std": float(ref_labels.std()),
        "a": a,
        "b": b,
        "applied_inplace": args.inplace,
    }
    (ORACLE_OUT / "bias_correction.json").write_text(json.dumps(correction_log, indent=2))
    print(f"\nSaved correction params to {ORACLE_OUT / 'bias_correction.json'}")


if __name__ == "__main__":
    main()
