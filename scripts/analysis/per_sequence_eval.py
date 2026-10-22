"""Per-sequence eval using Sahu motif coefficients.

For each test sequence in 3 negative panels, compute:
  - Motif counts (SP1, E2F, CRE, EBOX, YY1, GATA, CTCF, ETS, TATA)
  - CpG, GC
  - Oracle prediction
  - Expected activity from motif features (using Sahu coefs)

Then for each variant, identify:
  - Sequences with NO motifs but HIGH prediction (model over-predicts)
  - Sequences with motifs but LOW prediction (model under-predicts)
  - For ctrl_neg (real labels): which features drive prediction RESIDUALS
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")

# Sahu motif coefficients
SAHU = json.loads((REPO / "outputs/sahu_motif_analysis/motif_regression_summary.json").read_text())
COEFS = SAHU["full_model_coefs"]
print(f"Sahu coefs: {COEFS}")

MOTIFS_RAW = {
    "SP1":    ["GGGGCGGGGC", "GCCCCGCCCC"],
    "E2F":    ["TTTCGCGC", "GCGCGAAA"],
    "CRE":    ["TGACGTCA", "ATGACGTCA"],
    "EBOX":   ["CACGTG", "CAGCTG", "CATGTG"],
    "GATA":   ["AGATAA", "TTATCT"],
    "CTCF":   ["CCCTC", "GAGGG"],
    "ETS":    ["GGAA", "TTCC"],
    "TATA":   ["TATAAA", "TTTATA"],
}
RC = str.maketrans("ACGTN", "TGCAN")
def rc(s): return s.translate(RC)[::-1]
MOTIF_LOOKUPS = {k: list(set(v + [rc(p) for p in v])) for k, v in MOTIFS_RAW.items() if all("N" not in p for p in v)}
MOTIF_KEYS = sorted(MOTIF_LOOKUPS.keys())

def motif_count(s, m):
    return sum(s.count(p) for p in MOTIF_LOOKUPS[m])

def cpg(s):
    return sum(1 for i in range(len(s)-1) if s[i]=="C" and s[i+1]=="G") / max(len(s)-1, 1)

def gc(s):
    return (s.count("G") + s.count("C")) / max(len(s), 1)

def features(seqs):
    rows = []
    for s in seqs:
        s = str(s).upper()
        row = {"cpg": cpg(s), "gc": gc(s)}
        for m in MOTIF_KEYS:
            row[m] = motif_count(s, m)
        rows.append(row)
    return pd.DataFrame(rows)

def expected_log_count(feats_df):
    out = COEFS["const"] + COEFS["cpg"] * feats_df["cpg"] + COEFS["gc"] * feats_df["gc"]
    for m in MOTIF_KEYS:
        if m in COEFS:
            out += COEFS[m] * feats_df[m]
    return out

# Load test sets + features
print("\nComputing features for 3 negative panels...")
panels = {}
for name, csv_path in [
    ("ctrl_neg", REPO / "outputs/diverse_test_sets_k562/gosai_ctrl_neg_real.csv"),
    ("dinuc_shuffled", REPO / "outputs/diverse_test_sets_k562/dinuc_shuffled.csv"),
    ("random_uniform", REPO / "outputs/diverse_test_sets_k562/random_uniform.csv"),
]:
    df = pd.read_csv(csv_path)
    seqs = df["sequence"].astype(str).tolist()
    feats = features(seqs)
    feats["seq"] = seqs
    if "real_label" in df.columns:
        feats["real_label"] = df["real_label"].values
    feats["expected_log_count"] = expected_log_count(feats)
    panels[name] = feats
    print(f"  {name}: n={len(feats)}, motif_total mean={feats[MOTIF_KEYS].sum(axis=1).mean():.2f}, cpg mean={feats.cpg.mean():.4f}")

# For each variant + baseline, attach predictions
VARIANTS = ["V0_baseline", "sahu_bin_constraint", "moment_match_neg", "moment_plus_sahu",
            "adaptive_target_match", "adaptive_plus_sahu", "adaptive_plus_invariance"]

print("\n=== Per-sequence residuals on ctrl_neg (real labels) ===")
print(f"{'Variant':<27} | {'bias_no_motif':<13} | {'bias_with_motif':<15} | {'top_residual_motif':<20}")
print("-" * 100)

# Identify sequences with/without motifs
ctrl_feat = panels["ctrl_neg"]
ctrl_feat["motif_total"] = ctrl_feat[MOTIF_KEYS].sum(axis=1)
ctrl_feat["has_motif"] = ctrl_feat["motif_total"] > 0
print(f"  ctrl_neg has_motif distribution: {ctrl_feat['has_motif'].value_counts().to_dict()}")

for v in VARIANTS:
    if v == "V0_baseline":
        pred = np.load(REPO / "outputs/oracle_pseudolabels_diverse_k562/gosai_ctrl_neg_real_oracle.npz")["oracle_mean"]
    else:
        path = REPO / f"outputs/debias_screen/{v}/fold_0/six_panel_predictions/gosai_ctrl_neg_pred.npz"
        if not path.exists():
            print(f"{v:<27} | (no preds)"); continue
        pred = np.load(path)["pred"]
    if "real_label" not in ctrl_feat.columns: continue
    real = ctrl_feat["real_label"].values
    resid = pred - real
    # Bias by motif presence
    bias_no_motif = resid[~ctrl_feat["has_motif"]].mean() if (~ctrl_feat["has_motif"]).sum() else float("nan")
    bias_with_motif = resid[ctrl_feat["has_motif"]].mean() if ctrl_feat["has_motif"].sum() else float("nan")
    # Which motif is most associated with prediction residual?
    from scipy.stats import spearmanr
    motif_corrs = {}
    for m in MOTIF_KEYS:
        if ctrl_feat[m].std() > 0:
            r = spearmanr(ctrl_feat[m].values, resid).correlation
            motif_corrs[m] = r
    top_motif = max(motif_corrs.items(), key=lambda x: abs(x[1]) if x[1] == x[1] else 0)
    print(f"{v:<27} | {bias_no_motif:+.4f}       | {bias_with_motif:+.4f}         | {top_motif[0]} (rho={top_motif[1]:+.3f})")

# Bias analysis on random_uniform (no real labels)
print(f"\n=== Random_uniform per-sequence analysis ===")
print(f"  Sequences with NO motifs (motif_total=0): {(panels['random_uniform'][MOTIF_KEYS].sum(axis=1) == 0).sum()}")
print(f"  Sequences with 1+ motifs:                  {(panels['random_uniform'][MOTIF_KEYS].sum(axis=1) > 0).sum()}")
print(f"  Sequences with 3+ motifs:                  {(panels['random_uniform'][MOTIF_KEYS].sum(axis=1) >= 3).sum()}")
print()
print(f"{'Variant':<27} | {'μ no-motif':<11} | {'μ 1-motif':<11} | {'μ 3+ motif':<11}")
print("-" * 80)

rand_feat = panels["random_uniform"]
rand_feat["motif_total"] = rand_feat[MOTIF_KEYS].sum(axis=1)
no_motif_mask = (rand_feat["motif_total"] == 0).values
one_motif_mask = (rand_feat["motif_total"] == 1).values
many_motif_mask = (rand_feat["motif_total"] >= 3).values

for v in VARIANTS:
    if v == "V0_baseline":
        pred = np.load(REPO / "outputs/oracle_pseudolabels_diverse_k562/random_uniform_oracle.npz")["oracle_mean"]
    else:
        path = REPO / f"outputs/debias_screen/{v}/fold_0/six_panel_predictions/random_uniform_pred.npz"
        if not path.exists(): continue
        pred = np.load(path)["pred"]
    mean_no = pred[no_motif_mask].mean() if no_motif_mask.sum() else float("nan")
    mean_one = pred[one_motif_mask].mean() if one_motif_mask.sum() else float("nan")
    mean_many = pred[many_motif_mask].mean() if many_motif_mask.sum() else float("nan")
    print(f"{v:<27} | {mean_no:+.3f}     | {mean_one:+.3f}     | {mean_many:+.3f}")

out = REPO / "results/debias_variants/per_sequence_analysis.json"
out.parent.mkdir(parents=True, exist_ok=True)
# Save panel features
for name, df in panels.items():
    df.to_parquet(REPO / f"results/debias_variants/{name}_features.parquet")
print(f"\nSaved per-panel features to results/debias_variants/")
