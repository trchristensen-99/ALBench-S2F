"""Post-hoc affine recalibration of chr-split model predictions.

For each cell:
  1. Load val_pred + val_labels (from model.npz or chr_val cache)
  2. Fit α + β by minimizing val MSE: val_labels ≈ α + β · val_pred
  3. Apply transform to test predictions on all panels
  4. Compute recalibrated metrics; add to summary.json under _calibrated keys

Writes back into outputs/{legnet,ag}_chrsplit_scaling/*/summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[1]

CHR_VAL = REPO / "outputs/chr_split_cache/chr_val_ref_only.npz"
TEST_GENOMIC = REPO / "data/k562/test_sets_ag_s2_chrsplit/genomic_oracle.npz"
TEST_OOD = REPO / "data/k562/test_sets_ag_s2_chrsplit/ood_oracle.npz"
TEST_SNV = REPO / "data/k562/test_sets_ag_s2_chrsplit/snv_oracle.npz"


def fit_affine(val_pred: np.ndarray, val_labels: np.ndarray):
    """Closed-form affine fit minimizing MSE: y ≈ α + β·x."""
    m = np.isfinite(val_pred) & np.isfinite(val_labels)
    x, y = val_pred[m].astype(np.float64), val_labels[m].astype(np.float64)
    if len(x) < 8:
        return 0.0, 1.0
    x_mean, y_mean = x.mean(), y.mean()
    num = ((x - x_mean) * (y - y_mean)).sum()
    den = ((x - x_mean) ** 2).sum()
    beta = num / den if den > 0 else 1.0
    alpha = y_mean - beta * x_mean
    return float(alpha), float(beta)


def metric(pred, y):
    m = np.isfinite(pred) & np.isfinite(y)
    if m.sum() < 8:
        return None
    r = float(pearsonr(pred[m], y[m])[0])
    mse = float(((pred[m] - y[m]) ** 2).mean())
    return {"pearson_r": r, "mse": mse, "n": int(m.sum())}


def get_val_labels(model_npz: dict, label_source: str):
    """Try model.npz first (AG saves it), else load chr_val cache (LegNet)."""
    if "val_labels" in model_npz:
        return model_npz["val_labels"].astype(np.float32)
    z = np.load(CHR_VAL, allow_pickle=True)
    labels = z[f"{label_source}_labels"].astype(np.float32)
    finite = np.isfinite(labels)
    return labels[finite]


def recalibrate_cell(cell_dir: Path, label_source: str):
    model_npz_path = cell_dir / "model.npz"
    summary_path = cell_dir / "summary.json"
    if not (model_npz_path.exists() and summary_path.exists()):
        return None
    z = np.load(model_npz_path, allow_pickle=True)
    if "val_pred" not in z.files:
        return None  # No val_pred → skip
    val_pred = z["val_pred"]
    val_labels = get_val_labels(z, label_source)
    if len(val_pred) != len(val_labels):
        # Length mismatch — try align by truncate
        n = min(len(val_pred), len(val_labels))
        val_pred = val_pred[:n]
        val_labels = val_labels[:n]
    alpha, beta = fit_affine(val_pred, val_labels)

    # Load test labels
    gen_z = np.load(TEST_GENOMIC, allow_pickle=True)
    test_oracle = gen_z["oracle_mean"].astype(np.float32)
    test_real = gen_z["true_label"].astype(np.float32)
    ood_z = np.load(TEST_OOD, allow_pickle=True)
    ood_oracle = ood_z["oracle_mean"].astype(np.float32)
    ood_real = ood_z["true_label"].astype(np.float32)
    snv_z = np.load(TEST_SNV, allow_pickle=True)
    snv_delta_oracle = snv_z["delta_mean"].astype(np.float32)
    snv_delta_real = snv_z["true_delta"].astype(np.float32)

    # Apply affine to test predictions
    test_cal = alpha + beta * z["test_pred"]
    ood_cal = alpha + beta * z["test_pred_ood"]
    snv_ref_cal = alpha + beta * z["test_pred_snv_ref"]
    snv_alt_cal = alpha + beta * z["test_pred_snv_alt"]
    snv_delta_cal = snv_alt_cal - snv_ref_cal  # affine cancels for delta when same β

    # Compute recalibrated metrics
    cal = {
        "alpha": alpha,
        "beta": beta,
        "test_vs_oracle_calibrated": metric(test_cal, test_oracle),
        "test_vs_real_calibrated": metric(test_cal, test_real),
        "ood_vs_oracle_calibrated": metric(ood_cal, ood_oracle),
        "ood_vs_real_calibrated": metric(ood_cal, ood_real),
        "snv_delta_vs_oracle_calibrated": metric(snv_delta_cal, snv_delta_oracle),
        "snv_delta_vs_real_calibrated": metric(snv_delta_cal, snv_delta_real),
    }

    # Write back into summary.json
    summary = json.loads(summary_path.read_text())
    summary["calibration"] = cal
    summary_path.write_text(json.dumps(summary, indent=2))
    return cal


def main():
    n_done = n_skip = 0
    for model_name in ["legnet", "ag"]:
        base = REPO / f"outputs/{model_name}_chrsplit_scaling"
        if not base.exists():
            continue
        for ls_dir in base.iterdir():
            if not ls_dir.is_dir():
                continue
            label_source = ls_dir.name
            for n_dir in ls_dir.iterdir():
                if not n_dir.is_dir():
                    continue
                for seed_dir in n_dir.iterdir():
                    if not seed_dir.is_dir():
                        continue
                    res = recalibrate_cell(seed_dir, label_source)
                    if res is None:
                        n_skip += 1
                    else:
                        n_done += 1
    print(f"  recalibrated: {n_done}  skipped (no val_pred): {n_skip}")


if __name__ == "__main__":
    main()
