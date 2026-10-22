"""Cleaner Agarwal↔Gosai mapping comparison: 4 methods + KS validation."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from collections import Counter

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")


def cpg(s):
    s = s.upper()
    return sum(1 for i in range(len(s)-1) if s[i]=="C" and s[i+1]=="G") / max(len(s)-1, 1)

def gc(s):
    s = s.upper()
    return (s.count("G") + s.count("C")) / max(len(s), 1)

def feats_for(seqs):
    out = []
    for s in seqs:
        s = str(s).upper()
        c = cpg(s); g = gc(s)
        # dinuc entropy
        dn = Counter(s[i:i+2] for i in range(len(s)-1))
        n = max(sum(dn.values()), 1)
        ent = -sum((v/n)*np.log(v/n) for v in dn.values() if v > 0)
        out.append([c, g, ent])
    return np.array(out)


# Load + clean
ag = pd.read_csv(REPO / "data/agarwal_2025/k562_dinucleotide_shuffled_controls.csv")
ag = ag.dropna(subset=["element_200nt", "log2_mean"]).reset_index(drop=True)
ag_seqs = ag["element_200nt"].astype(str).str.upper().values
ag_labels = ag["log2_mean"].to_numpy(dtype=np.float64)
ag_feats = feats_for(ag_seqs)
print(f"Agarwal: n={len(ag)}  mean={ag_labels.mean():+.3f}  std={ag_labels.std():.3f}")

gos = pd.read_parquet(REPO / "data/k562/gosai_ctrl_neg.parquet")
gos = gos.dropna(subset=["sequence", "K562_log2FC"]).reset_index(drop=True)
gos_seqs = gos["sequence"].astype(str).str.upper().values
gos_labels = gos["K562_log2FC"].to_numpy(dtype=np.float64)
gos_feats = feats_for(gos_seqs)
print(f"Gosai ctrl_neg: n={len(gos)}  mean={gos_labels.mean():+.3f}  std={gos_labels.std():.3f}")
print()

# M1: constant offset
diff_means = gos_labels.mean() - ag_labels.mean()
print(f"=== M1: constant offset ===  delta = {diff_means:+.3f}")
m1 = ag_labels + diff_means

# M2: per-quartile CpG offset
ag_cpg = ag_feats[:, 0]; gos_cpg = gos_feats[:, 0]
overlap_max = min(ag_cpg.max(), gos_cpg.max())
bins = np.linspace(0, overlap_max, 5)
ag_q = np.digitize(ag_cpg, bins) - 1
gos_q = np.digitize(gos_cpg, bins) - 1
shifts = np.zeros(4)
print(f"=== M2: per-CpG-quartile offset ===")
for q in range(4):
    ag_mask = ag_q == q
    gos_mask = gos_q == q
    if ag_mask.sum() > 5 and gos_mask.sum() > 5:
        a_mean = ag_labels[ag_mask].mean()
        g_mean = gos_labels[gos_mask].mean()
        shifts[q] = g_mean - a_mean
        print(f"  Q{q}: ag_n={ag_mask.sum()}, gos_n={gos_mask.sum()}, ag_mean={a_mean:+.3f}, gos_mean={g_mean:+.3f}, shift={shifts[q]:+.3f}")
    else:
        shifts[q] = diff_means  # fallback to constant
        print(f"  Q{q}: insufficient data, fallback shift={shifts[q]:+.3f}")
m2 = ag_labels.copy()
for q in range(4):
    m2[ag_q == q] += shifts[q]
# any out-of-range agarwal seqs (cpg > overlap_max) get constant fallback
out_of_range = ag_cpg > overlap_max
m2[out_of_range] = ag_labels[out_of_range] + diff_means

# M3: Ridge regression on features (predict shift from feats)
# Use 10-NN Gosai mean as the target Gosai label for each Agarwal seq
nn = NearestNeighbors(n_neighbors=10).fit(gos_feats)
_, idx = nn.kneighbors(ag_feats)
nn_means = gos_labels[idx].mean(axis=1)
shift_targets = nn_means - ag_labels
print(f"=== M3: Ridge on (CpG, GC, dinuc_entropy) ===")
ridge = Ridge(alpha=1.0).fit(ag_feats, shift_targets)
m3 = ag_labels + ridge.predict(ag_feats)
print(f"  Ridge coefs: cpg={ridge.coef_[0]:+.3f}, gc={ridge.coef_[1]:+.3f}, entropy={ridge.coef_[2]:+.3f}, b0={ridge.intercept_:+.3f}")

# M4: pure 10-NN blending
m4 = nn_means
print(f"=== M4: 10-NN blending ===")

# Validation
print()
print(f"=== Validation: predicted distributions vs Gosai ctrl_neg ===")
print(f"  Reference Gosai: mean={gos_labels.mean():+.3f}, std={gos_labels.std():.3f}, range=[{gos_labels.min():+.3f}, {gos_labels.max():+.3f}]")
for name, v in [("agarwal_orig", ag_labels), ("M1_const", m1), ("M2_quartile", m2), ("M3_ridge", m3), ("M4_kNN", m4)]:
    ks_stat, ks_p = ks_2samp(v, gos_labels)
    print(f"  {name:<14}: mean={v.mean():+.3f} (Δ={v.mean()-gos_labels.mean():+.3f})  std={v.std():.3f} (ratio={v.std()/gos_labels.std():.2f})  KS={ks_stat:.3f}  p={ks_p:.2g}")

# Save
out_dir = REPO / "outputs/agarwal_gosai_mapping"
out_dir.mkdir(parents=True, exist_ok=True)
out_df = pd.DataFrame({
    "name": ag.get("name", pd.Series([""] * len(ag))).values,
    "sequence": ag_seqs,
    "agarwal_label": ag_labels,
    "cpg": ag_cpg,
    "gc": ag_feats[:, 1],
    "m1_label": m1,
    "m2_label": m2,
    "m3_label": m3,
    "m4_label": m4,
})
out_df.to_csv(out_dir / "agarwal_shuffled_with_gosai_estimates.csv", index=False)
print(f"\nSaved {len(out_df)} rows to {out_dir / 'agarwal_shuffled_with_gosai_estimates.csv'}")

summary = {
    "agarwal_n": int(len(ag)),
    "gosai_ctrl_neg_n": int(len(gos)),
    "agarwal_mean": float(ag_labels.mean()),
    "agarwal_std": float(ag_labels.std()),
    "gosai_mean": float(gos_labels.mean()),
    "gosai_std": float(gos_labels.std()),
    "M1_const_offset": float(diff_means),
    "M2_quartile_shifts": [float(s) for s in shifts],
    "M3_ridge_coefs": {"cpg": float(ridge.coef_[0]), "gc": float(ridge.coef_[1]), "ent": float(ridge.coef_[2]), "b0": float(ridge.intercept_)},
    "stats": {},
}
for name, v in [("M1_const", m1), ("M2_quartile", m2), ("M3_ridge", m3), ("M4_kNN", m4)]:
    summary["stats"][name] = {"mean": float(v.mean()), "std": float(v.std()), "ks_vs_gosai": float(ks_2samp(v, gos_labels).statistic)}
(out_dir / "mapping_summary.json").write_text(json.dumps(summary, indent=2))
print(f"Summary saved to {out_dir / 'mapping_summary.json'}")
