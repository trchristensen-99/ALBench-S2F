"""Compare multiple post-hoc bias correction strategies on the AG-S2
oracle ensemble.

Strategies tested:
  1. constant_offset: y_corr = y_pred - mu_neg (single scalar shift)
  2. gc_affine: y_corr = y_pred - (a + b·GC) (linear in GC content)
  3. gc_poly2: y_corr = y_pred - (a + b·GC + c·GC²) (quadratic)
  4. isotonic: monotonic remapping fit on train pool (preserves rank;
     fixes activity-stratified pull-to-mean)
  5. gc_affine + isotonic: chain (1) then (2)

For each strategy reports on the test pool:
  - Pearson R on real labels (should be preserved)
  - MSE on real labels (should improve or stay flat)
  - Activity-stratified residuals at decile 1 / 5 / 10 (should flatten)
  - Predicted bias on random_gc_{25,50,75}pct (should drop toward 0)

Outputs:
  outputs/oracle_pseudolabels_k562_ag_s2_refalt/bias_correction_compare.json
  outputs/oracle_pseudolabels_k562_ag_s2_refalt/bias_correction_compare.csv

Strategy is chosen by the user after inspecting the comparison; the
chosen strategy can then be applied via apply_posthoc_bias_correction.py
to produce corrected pseudolabel npzs for the main sweep.

Usage:
  uv run --no-sync python scripts/preflight/bias_correction_strategies.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.isotonic import IsotonicRegression

REPO = Path(__file__).resolve().parents[2]
ORACLE_OUT = REPO / "outputs" / "oracle_pseudolabels_k562_ag_s2_refalt"


def _gc_content(seq: str) -> float:
    s = seq.upper()
    if not s:
        return 0.5
    return (s.count("G") + s.count("C")) / len(s)


def _load_data():
    """Load oracle predictions + true labels + per-seq GC content for
    train/val/test pools."""
    pool_dir = ORACLE_OUT / "pool"
    out = {}
    for split in ("train", "val", "test"):
        npz = np.load(ORACLE_OUT / f"{split}_oracle_labels.npz")
        df = pd.read_parquet(pool_dir / f"{split}.parquet")
        out[split] = {
            "y_pred": npz["oracle_mean"].astype(np.float64),
            "y_true": npz["true_label"].astype(np.float64),
            "gc": df["sequence"].astype(str).map(_gc_content).to_numpy(np.float64),
        }
    bias = json.loads((ORACLE_OUT / "bias_eval.json").read_text())
    return out, bias


def _bias_at_gc_levels(bias: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (gc_values, predicted_random_means) from the random_gc_*pct
    panels."""
    gc_vals = []
    means = []
    for k in sorted(bias):
        if not k.startswith("random_gc_"):
            continue
        try:
            pct = float(k.replace("random_gc_", "").replace("pct", ""))
        except ValueError:
            continue
        gc_vals.append(pct / 100.0)
        means.append(bias[k]["mean"])
    return np.array(gc_vals), np.array(means)


# ── Correction strategies ─────────────────────────────────────────────────
def correct_constant(data, bias):
    """y_corr = y_pred - mu_neg using random_gc_50pct as the calibration point."""
    mu = bias.get("random_gc_50pct", {}).get("mean", 0.0)
    return {split: d["y_pred"] - mu for split, d in data.items()}, {"mu": mu}


def correct_gc_affine(data, bias):
    """Fit bias(GC) = a + b·GC from the 7 GC-stratified random panels,
    subtract bias(GC) from each per-seq prediction."""
    gc_vals, means = _bias_at_gc_levels(bias)
    if len(gc_vals) < 2:
        raise ValueError("Need >=2 GC-stratified panels for affine fit")
    A = np.vstack([np.ones_like(gc_vals), gc_vals]).T
    a, b = np.linalg.lstsq(A, means, rcond=None)[0]
    out = {}
    for split, d in data.items():
        bias_pred = a + b * d["gc"]
        out[split] = d["y_pred"] - bias_pred
    return out, {"a": float(a), "b": float(b), "fit_gc": gc_vals.tolist(), "fit_mean": means.tolist()}


def correct_gc_poly2(data, bias):
    """Quadratic in GC."""
    gc_vals, means = _bias_at_gc_levels(bias)
    if len(gc_vals) < 3:
        raise ValueError("Need >=3 GC panels for quadratic fit")
    coeffs = np.polyfit(gc_vals, means, deg=2)  # [c, b, a] where bias = c·GC² + b·GC + a
    out = {}
    for split, d in data.items():
        bias_pred = np.polyval(coeffs, d["gc"])
        out[split] = d["y_pred"] - bias_pred
    return out, {"coeffs_high_to_low": coeffs.tolist()}


def correct_isotonic(data, bias):
    """Monotonic remapping fit on train pool: y_corr = isotonic_fn(y_pred)
    where isotonic_fn maps y_pred → calibrated label. Preserves rank
    order (Pearson R), fixes activity-stratified pull-to-mean."""
    y_pred_train = data["train"]["y_pred"]
    y_true_train = data["train"]["y_true"]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_pred_train, y_true_train)
    out = {split: iso.predict(d["y_pred"]) for split, d in data.items()}
    return out, {"n_breakpoints": int(len(iso.X_thresholds_))}


def correct_gc_affine_then_isotonic(data, bias):
    """Sequential: GC-affine first, then isotonic on the residual."""
    affine_out, affine_p = correct_gc_affine(data, bias)
    affine_data = {split: {**data[split], "y_pred": affine_out[split]} for split in data}
    iso_out, iso_p = correct_isotonic(affine_data, bias)
    return iso_out, {"stage1": affine_p, "stage2": iso_p}


# ── Evaluation ────────────────────────────────────────────────────────────
def _evaluate(corrected, data, bias, strategy_name):
    """Compute headline metrics on the test split + predicted bias on
    the random_gc panels."""
    test_pred = corrected["test"]
    test_true = data["test"]["y_true"]
    pearson_r = float(pearsonr(test_pred, test_true)[0])
    spearman_r = float(spearmanr(test_pred, test_true)[0])
    mse = float(np.mean((test_pred - test_true) ** 2))
    mean_residual = float(np.mean(test_pred - test_true))

    # Activity-stratified residual change (decile 1 = lowest, 10 = highest)
    deciles = np.quantile(test_true, np.linspace(0, 1, 11))
    by_decile = {}
    for i in range(10):
        lo, hi = deciles[i], deciles[i + 1]
        mask = (test_true >= lo) & (test_true <= hi if i == 9 else test_true < hi)
        if mask.sum() == 0:
            continue
        resid = test_pred[mask] - test_true[mask]
        by_decile[i + 1] = {"mean_resid": float(resid.mean()), "n": int(mask.sum())}

    # Predicted bias at the random_gc panels — for affine/poly we can
    # exactly compute the corrected mean. For isotonic we'd need actual
    # random-DNA predictions which we don't have, so we report 'n/a'.
    gc_vals, original_means = _bias_at_gc_levels(bias)
    predicted_bias_after = {}
    if strategy_name in ("constant_offset",):
        mu = original_means.mean()  # not used; we used 50pct as calibration
        # Actually this strategy applied a fixed shift, so corrected = original - mu_50pct
        mu_50 = bias.get("random_gc_50pct", {}).get("mean", 0.0)
        for gc_v, m in zip(gc_vals, original_means):
            predicted_bias_after[f"gc_{int(gc_v * 100)}pct"] = float(m - mu_50)
    elif strategy_name in ("gc_affine", "gc_poly2"):
        # For affine: corrected mean at GC=g is original_mean(g) - bias_fn(g) ≈ 0
        # We expect ~0 if the fit was perfect; residual = original_mean - fitted
        # Fit is to the same data, so residuals are small but non-zero
        # Re-fit to evaluate residuals
        if strategy_name == "gc_affine":
            A = np.vstack([np.ones_like(gc_vals), gc_vals]).T
            a, b = np.linalg.lstsq(A, original_means, rcond=None)[0]
            for gc_v, m in zip(gc_vals, original_means):
                predicted_bias_after[f"gc_{int(gc_v * 100)}pct"] = float(m - (a + b * gc_v))
        else:
            coeffs = np.polyfit(gc_vals, original_means, deg=2)
            for gc_v, m in zip(gc_vals, original_means):
                predicted_bias_after[f"gc_{int(gc_v * 100)}pct"] = float(m - np.polyval(coeffs, gc_v))
    else:
        # Isotonic and gc_affine_then_isotonic: would need per-seq random
        # DNA predictions. Mark as not directly evaluable.
        predicted_bias_after = {f"gc_{int(g * 100)}pct": "needs_per_seq_random" for g in gc_vals}

    return {
        "strategy": strategy_name,
        "test_pearson_r": pearson_r,
        "test_spearman_r": spearman_r,
        "test_mse": mse,
        "test_mean_residual": mean_residual,
        "decile_1_mean_resid": by_decile.get(1, {}).get("mean_resid"),
        "decile_5_mean_resid": by_decile.get(5, {}).get("mean_resid"),
        "decile_10_mean_resid": by_decile.get(10, {}).get("mean_resid"),
        "predicted_random_bias_after_correction": predicted_bias_after,
    }


def main():
    data, bias = _load_data()
    print("=== Loaded ===")
    for split, d in data.items():
        print(f"  {split}: n={len(d['y_pred']):,}  y_pred μ={d['y_pred'].mean():.3f}  "
              f"y_true μ={d['y_true'].mean():.3f}  GC μ={d['gc'].mean():.3f}")

    # Baseline (no correction)
    baseline = {split: d["y_pred"] for split, d in data.items()}
    results = [_evaluate(baseline, data, bias, "baseline_no_correction")]
    results[0]["params"] = {}

    strategies = [
        ("constant_offset", correct_constant),
        ("gc_affine", correct_gc_affine),
        ("gc_poly2", correct_gc_poly2),
        ("isotonic", correct_isotonic),
        ("gc_affine_then_isotonic", correct_gc_affine_then_isotonic),
    ]
    corrected_predictions = {}
    for name, fn in strategies:
        try:
            corr, params = fn(data, bias)
            r = _evaluate(corr, data, bias, name)
            r["params"] = params
            results.append(r)
            corrected_predictions[name] = corr
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")

    # Pretty print
    print()
    print(f"=== Strategy comparison (test pool) ===")
    fmt = "{:<28s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s}"
    print(fmt.format("strategy", "pearson", "spearman", "mse", "resid_d1", "resid_d5", "resid_d10"))
    print("-" * 80)
    for r in results:
        print(fmt.format(
            r["strategy"][:28],
            f"{r['test_pearson_r']:.4f}",
            f"{r['test_spearman_r']:.4f}",
            f"{r['test_mse']:.4f}",
            f"{r['decile_1_mean_resid'] or 0:+.3f}",
            f"{r['decile_5_mean_resid'] or 0:+.3f}",
            f"{r['decile_10_mean_resid'] or 0:+.3f}",
        ))

    # Predicted random-DNA bias remaining after correction (per strategy)
    print()
    print("=== Predicted random-DNA bias AFTER correction (lower abs = better; n/a needs re-eval) ===")
    print(f"{'strategy':<28s} {'gc=25%':>10s} {'gc=50%':>10s} {'gc=75%':>10s}")
    print("-" * 60)
    for r in results:
        rb = r.get("predicted_random_bias_after_correction") or {}
        print(f"{r['strategy'][:28]:<28s} "
              f"{str(rb.get('gc_25pct', 'n/a'))[:10]:>10s} "
              f"{str(rb.get('gc_50pct', 'n/a'))[:10]:>10s} "
              f"{str(rb.get('gc_75pct', 'n/a'))[:10]:>10s}")

    # Save
    out_json = ORACLE_OUT / "bias_correction_compare.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_json}")

    out_csv = ORACLE_OUT / "bias_correction_compare.csv"
    with out_csv.open("w") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "strategy", "test_pearson_r", "test_spearman_r", "test_mse",
                "test_mean_residual", "decile_1_mean_resid", "decile_5_mean_resid",
                "decile_10_mean_resid",
            ],
        )
        w.writeheader()
        for r in results:
            row = {k: r.get(k) for k in w.fieldnames}
            w.writerow(row)
    print(f"Saved {out_csv}")

    # Save the corrected predictions for the WINNING strategies so they
    # can be compared in score_eval_sets / used as drop-in replacements.
    for name, corr in corrected_predictions.items():
        out_dir = ORACLE_OUT / f"correction_{name}"
        out_dir.mkdir(exist_ok=True)
        for split, y in corr.items():
            np.savez_compressed(
                out_dir / f"{split}_oracle_labels_corrected.npz",
                oracle_mean=y.astype(np.float32),
                true_label=data[split]["y_true"].astype(np.float32),
            )
    print(f"Per-strategy corrected npzs saved under {ORACLE_OUT}/correction_*/")


if __name__ == "__main__":
    main()
