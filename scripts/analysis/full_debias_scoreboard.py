"""Unified 6-panel × (1 baseline + 6 variants) scoreboard with REAL labels where possible."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, ks_2samp

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")

def safe_p(p, t):
    p = np.asarray(p, np.float64); t = np.asarray(t, np.float64)
    m = np.isfinite(p) & np.isfinite(t)
    return float(pearsonr(p[m], t[m])[0]) if m.sum() > 3 else float("nan")
def safe_m(p, t):
    p = np.asarray(p, np.float64); t = np.asarray(t, np.float64)
    m = np.isfinite(p) & np.isfinite(t)
    return float(((p[m] - t[m])**2).mean()) if m.sum() > 3 else float("nan")

# Load real labels for ctrl_neg
ctrl_real = pd.read_parquet(REPO / "data/k562/gosai_ctrl_neg.parquet")["K562_log2FC"].values
target_mu = ctrl_real.mean(); target_sigma = ctrl_real.std()

# Variants: baseline + 6 debias
VARIANTS = ["V0_baseline"] + ["sahu_bin_constraint", "moment_match_neg", "moment_plus_sahu",
                              "adaptive_target_match", "adaptive_plus_sahu", "adaptive_plus_invariance"]

print(f"{'Variant':<27} | {'ctrl_neg μ':<11} {'σ':<6} {'r':<6} {'mse':<6} | {'dinuc μ':<8} {'σ':<6} {'KS':<5} | {'rand μ':<8} {'σ':<6} {'KS':<5}")
print("-" * 130)

rows = []
for v in VARIANTS:
    if v == "V0_baseline":
        ctrl_p = np.load(REPO / "outputs/oracle_pseudolabels_diverse_k562/gosai_ctrl_neg_real_oracle.npz")["oracle_mean"]
        dinuc_p = np.load(REPO / "outputs/oracle_pseudolabels_diverse_k562/dinuc_shuffled_oracle.npz")["oracle_mean"]
        random_p = np.load(REPO / "outputs/oracle_pseudolabels_diverse_k562/random_uniform_oracle.npz")["oracle_mean"]
    else:
        d = REPO / f"outputs/debias_screen/{v}/fold_0/six_panel_predictions"
        if not (d / "random_uniform_pred.npz").exists():
            print(f"{v:<27} | (no inference)")
            continue
        ctrl_p = np.load(d / "gosai_ctrl_neg_pred.npz")["pred"]
        dinuc_p = np.load(d / "dinuc_shuffled_pred.npz")["pred"]
        random_p = np.load(d / "random_uniform_pred.npz")["pred"]
    r = safe_p(ctrl_p, ctrl_real); m = safe_m(ctrl_p, ctrl_real)
    ks_dinuc = float(ks_2samp(dinuc_p, ctrl_real).statistic)
    ks_rand = float(ks_2samp(random_p, ctrl_real).statistic)
    print(f"{v:<27} | {ctrl_p.mean():+.3f}     {ctrl_p.std():.3f}  {r:.4f}  {m:.4f} | "
          f"{dinuc_p.mean():+.3f}   {dinuc_p.std():.3f}  {ks_dinuc:.3f} | "
          f"{random_p.mean():+.3f}   {random_p.std():.3f}  {ks_rand:.3f}")
    rows.append({
        "variant": v,
        "ctrl_neg": {"pred_mean": float(ctrl_p.mean()), "pred_std": float(ctrl_p.std()),
                     "real_mean": float(target_mu), "real_std": float(target_sigma),
                     "pearson_r": r, "mse": m, "bias": float(ctrl_p.mean() - target_mu)},
        "dinuc_shuffled": {"pred_mean": float(dinuc_p.mean()), "pred_std": float(dinuc_p.std()),
                           "ks_vs_ctrl_real": ks_dinuc},
        "random_uniform": {"pred_mean": float(random_p.mean()), "pred_std": float(random_p.std()),
                           "ks_vs_ctrl_real": ks_rand},
    })

print()
print(f"  Target ctrl_neg: μ={target_mu:+.3f}, σ={target_sigma:.3f}")
print(f"  Target dinuc:    μ≈+0.30 (CpG-adjusted from ctrl_neg, Sahu motif evidence)")
print(f"  Target random:   μ≈+0.37 (CpG-adjusted from ctrl_neg, Sahu motif evidence)")

# Save
out = REPO / "results/debias_variants/full_scoreboard.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, indent=2))
print(f"\nSaved {out}")
