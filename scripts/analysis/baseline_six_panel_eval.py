"""Evaluate baseline V0 (10-fold AG-S2 oracle ensemble) on all 6 eval panels.
This is the FLOOR before any debiasing."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, ks_2samp

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")

def cpg_density(s):
    s = str(s).upper()
    return sum(1 for i in range(len(s)-1) if s[i]=="C" and s[i+1]=="G") / max(len(s)-1, 1)

def safe_pearson(p, t):
    p = np.asarray(p, np.float64); t = np.asarray(t, np.float64)
    m = np.isfinite(p) & np.isfinite(t)
    return float(pearsonr(p[m], t[m])[0]) if m.sum() > 3 else float("nan")

def safe_mse(p, t):
    p = np.asarray(p, np.float64); t = np.asarray(t, np.float64)
    m = np.isfinite(p) & np.isfinite(t)
    return float(((p[m] - t[m])**2).mean()) if m.sum() > 3 else float("nan")

# Load positive panel predictions + real labels
chr_npz = np.load(REPO / "outputs/oracle_pseudolabels_k562_chr_split_ag_s2/chr_7_13_all_oracle.npz")
ood_npz = np.load(REPO / "outputs/oracle_pseudolabels_k562_chr_split_ag_s2/ood_designed_oracle.npz")
snv_npz = np.load(REPO / "outputs/oracle_pseudolabels_k562_chr_split_ag_s2/snv_pairs_full_oracle.npz")

# Genomic ref_32k
ref_mask = chr_npz["allele"] == "R"
genomic_pred = chr_npz["oracle_mean"][ref_mask]
genomic_real = chr_npz["true_label"][ref_mask]

# OOD
ood_pred = ood_npz["oracle_mean"]
ood_real = ood_npz["true_label"]

# SNV delta
snv_csv = pd.read_csv(REPO / "data/k562/test_sets/test_snv_pairs.tsv", sep="\t")
mono_mask = snv_npz["mono_mask"]
if "K562_log2FC_ref" in snv_csv.columns and "K562_log2FC_alt" in snv_csv.columns:
    snv_real_delta = (snv_csv["K562_log2FC_alt"] - snv_csv["K562_log2FC_ref"]).values[mono_mask]
    snv_pred_delta = (snv_npz["alt_oracle_mean"] - snv_npz["ref_oracle_mean"])[mono_mask]
else:
    snv_real_delta = snv_npz["delta_oracle_mean"][mono_mask]
    snv_pred_delta = snv_real_delta

# Negative panel predictions
ctrl_pred = np.load(REPO / "outputs/oracle_pseudolabels_diverse_k562/gosai_ctrl_neg_real_oracle.npz")["oracle_mean"]
dinuc_pred = np.load(REPO / "outputs/oracle_pseudolabels_diverse_k562/dinuc_shuffled_oracle.npz")["oracle_mean"]
random_pred = np.load(REPO / "outputs/oracle_pseudolabels_diverse_k562/random_uniform_oracle.npz")["oracle_mean"]

# Real ctrl_neg labels
ctrl_real = pd.read_parquet(REPO / "data/k562/gosai_ctrl_neg.parquet")["K562_log2FC"].values

results = {}

# === POSITIVES ===
results["genomic_ref_32k"] = {
    "n": int(len(genomic_pred)),
    "pred_mean": float(genomic_pred.mean()),
    "pred_std": float(genomic_pred.std()),
    "real_mean": float(genomic_real.mean()),
    "real_std": float(genomic_real.std()),
    "pearson_r": safe_pearson(genomic_pred, genomic_real),
    "mse": safe_mse(genomic_pred, genomic_real),
    "bias": float(genomic_pred.mean() - genomic_real.mean()),
}

results["snv_delta_30k_mono"] = {
    "n": int(len(snv_pred_delta)),
    "pred_mean": float(snv_pred_delta.mean()),
    "pred_std": float(snv_pred_delta.std()),
    "real_mean": float(snv_real_delta.mean()),
    "real_std": float(snv_real_delta.std()),
    "pearson_r": safe_pearson(snv_pred_delta, snv_real_delta),
    "mse": safe_mse(snv_pred_delta, snv_real_delta),
    "bias": float(snv_pred_delta.mean() - snv_real_delta.mean()),
}

results["ood_designed_22k"] = {
    "n": int(len(ood_pred)),
    "pred_mean": float(ood_pred.mean()),
    "pred_std": float(ood_pred.std()),
    "real_mean": float(ood_real.mean()),
    "real_std": float(ood_real.std()),
    "pearson_r": safe_pearson(ood_pred, ood_real),
    "mse": safe_mse(ood_pred, ood_real),
    "bias": float(ood_pred.mean() - ood_real.mean()),
}

# === NEGATIVES ===
# ctrl_neg has REAL labels
results["gosai_ctrl_neg_503"] = {
    "n": int(len(ctrl_pred)),
    "pred_mean": float(ctrl_pred.mean()),
    "pred_std": float(ctrl_pred.std()),
    "real_mean": float(ctrl_real.mean()),
    "real_std": float(ctrl_real.std()),
    "pearson_r": safe_pearson(ctrl_pred, ctrl_real),
    "mse": safe_mse(ctrl_pred, ctrl_real),
    "bias": float(ctrl_pred.mean() - ctrl_real.mean()),
    "ks_vs_ctrl_real": float(ks_2samp(ctrl_pred, ctrl_real).statistic),
}

# dinuc_shuffled: no real labels → distributional comparison vs ctrl_neg
results["dinuc_shuffled_5k"] = {
    "n": int(len(dinuc_pred)),
    "pred_mean": float(dinuc_pred.mean()),
    "pred_std": float(dinuc_pred.std()),
    "ctrl_neg_mean_for_ref": float(ctrl_real.mean()),
    "ctrl_neg_std_for_ref": float(ctrl_real.std()),
    "bias_from_ctrl_neg": float(dinuc_pred.mean() - ctrl_real.mean()),
    "ks_vs_ctrl_real": float(ks_2samp(dinuc_pred, ctrl_real).statistic),
}

# random_uniform: similar
results["random_uniform_5k"] = {
    "n": int(len(random_pred)),
    "pred_mean": float(random_pred.mean()),
    "pred_std": float(random_pred.std()),
    "ctrl_neg_mean_for_ref": float(ctrl_real.mean()),
    "ctrl_neg_std_for_ref": float(ctrl_real.std()),
    "bias_from_ctrl_neg": float(random_pred.mean() - ctrl_real.mean()),
    "ks_vs_ctrl_real": float(ks_2samp(random_pred, ctrl_real).statistic),
}

# Display
print("=" * 80)
print("BASELINE V0 (10-fold AG-S2 oracle ensemble) — 6-PANEL EVALUATION")
print("=" * 80)
print()
print("--- POSITIVE PANELS (real labels available) ---")
for name in ["genomic_ref_32k", "snv_delta_30k_mono", "ood_designed_22k", "gosai_ctrl_neg_503"]:
    r = results[name]
    print(f"\n{name}  (n={r['n']})")
    print(f"  pred:  μ={r['pred_mean']:+.3f}  σ={r['pred_std']:.3f}")
    if 'real_mean' in r:
        print(f"  real:  μ={r['real_mean']:+.3f}  σ={r['real_std']:.3f}")
        print(f"  ACCURACY: Pearson r={r['pearson_r']:.4f}  MSE={r['mse']:.4f}  bias={r['bias']:+.3f}")
    if 'ks_vs_ctrl_real' in r:
        print(f"  DIST (KS vs ctrl_neg real): {r['ks_vs_ctrl_real']:.3f}")

print()
print("--- NEGATIVE PANELS (synthetic, distributional only) ---")
for name in ["dinuc_shuffled_5k", "random_uniform_5k"]:
    r = results[name]
    print(f"\n{name}  (n={r['n']})")
    print(f"  pred:  μ={r['pred_mean']:+.3f}  σ={r['pred_std']:.3f}")
    print(f"  target μ (ctrl_neg empirical): +{r['ctrl_neg_mean_for_ref']:.3f}, σ={r['ctrl_neg_std_for_ref']:.3f}")
    print(f"  Δμ from target: {r['bias_from_ctrl_neg']:+.3f}")
    print(f"  KS vs ctrl_neg distribution: {r['ks_vs_ctrl_real']:.3f}")

# Save
out_path = REPO / "results/debias_variants/V0_baseline_eval.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(results, indent=2))
print(f"\nSaved {out_path}")
