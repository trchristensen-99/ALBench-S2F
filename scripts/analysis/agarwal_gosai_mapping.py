"""Test 4 Agarwal-to-Gosai assay-shift mapping methods.

Each method translates Agarwal's 250 real K562 lentiviral shuffled-control labels
into estimated Gosai-scale (episomal) labels.

Validation criteria (no per-sequence ground truth — Agarwal ≠ Gosai sequences):
- KS distance vs Gosai ctrl_neg empirical distribution
- Quartile-mean alignment with Gosai quartile means
- σ preservation (predicted ≈ Gosai σ ~0.49)

Methods:
  M1 = constant offset (median Gosai_q_mean - Agarwal_q_mean across quartiles)
  M2 = per-CpG-quartile offset (4 offsets, one per quartile)
  M3 = ridge regression on (CpG, GC, dinuc_entropy, motif_count) features
  M4 = nearest-neighbor blending (k=10 nearest Gosai ctrl_neg by feature)
"""
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

def dinuc_entropy(s):
    s = s.upper()
    dinucs = [s[i:i+2] for i in range(len(s)-1)]
    if not dinucs: return 0.0
    cnt = Counter(dinucs); n = sum(cnt.values())
    return -sum((v/n) * np.log(v/n) for v in cnt.values() if v > 0)

# Simple motif-density proxy: count overlapping motifs from a small set
# (CTCF, GATA, ETS, FOX, AP-1 cores)
COMMON_MOTIFS = ["CCCTC", "GATAA", "CCGGAA", "TGACTCA", "TTGTTT", "CAAT", "TATA", "ACGCGT"]
def motif_count(s):
    s = s.upper()
    return sum(s.count(m) for m in COMMON_MOTIFS)

def features(s):
    return np.array([cpg(s), gc(s), dinuc_entropy(s), motif_count(s)])


def main():
    # Load datasets
    ag = pd.read_csv(REPO / "data/agarwal_2025/k562_dinucleotide_shuffled_controls.csv")
    ag["sequence"] = ag["element_200nt"].str.upper()
    ag["label"] = ag["log2_mean"]
    print(f"Agarwal shuffled: n={len(ag)}, mean={ag.label.mean():.3f}, std={ag.label.std():.3f}")

    gos = pd.read_parquet(REPO / "data/k562/gosai_ctrl_neg.parquet")
    gos["label"] = gos["K562_log2FC"]
    gos["sequence"] = gos["sequence"].str.upper()
    print(f"Gosai ctrl_neg: n={len(gos)}, mean={gos.label.mean():.3f}, std={gos.label.std():.3f}")

    # Compute features
    print("Computing features...")
    ag_feats = np.stack([features(s) for s in ag.sequence])
    gos_feats = np.stack([features(s) for s in gos.sequence])
    ag["cpg"] = ag_feats[:, 0]; ag["gc"] = ag_feats[:, 1]
    gos["cpg"] = gos_feats[:, 0]; gos["gc"] = gos_feats[:, 1]
    print(f"  Agarwal feat shape: {ag_feats.shape}, Gosai: {gos_feats.shape}")

    # === M1: constant offset ===
    # Compute mean shift per CpG quartile (use quartiles defined on combined CpG range)
    overlap_max = min(ag.cpg.max(), gos.cpg.max())
    ag_in = ag[ag.cpg <= overlap_max]; gos_in = gos[gos.cpg <= overlap_max]
    quart_bins = np.linspace(0, overlap_max, 5)
    ag_in["q"] = pd.cut(ag_in["cpg"], quart_bins, include_lowest=True, labels=False)
    gos_in["q"] = pd.cut(gos_in["cpg"], quart_bins, include_lowest=True, labels=False)
    shifts = []
    for q in range(4):
        a = ag_in[ag_in.q == q].label.mean()
        g = gos_in[gos_in.q == q].label.mean()
        shifts.append(g - a)
        print(f"  Q{q} (CpG [{quart_bins[q]:.4f},{quart_bins[q+1]:.4f}]): Ag={a:+.3f}, Gos={g:+.3f}, shift={g-a:+.3f}")
    M1_shift = np.nanmedian(shifts)
    print(f"  M1 constant offset (median across quartiles): {M1_shift:+.3f}")

    ag["m1_label"] = ag.label + M1_shift

    # === M2: per-quartile offset ===
    ag["m2_label"] = np.nan
    for q in range(4):
        if not np.isnan(shifts[q]):
            mask = (ag_in.q == q).reindex(ag.index, fill_value=False)
            ag.loc[mask, "m2_label"] = ag.loc[mask, "label"] + shifts[q]
    # Fallback for any unassigned: constant offset
    ag.loc[ag.m2_label.isna(), "m2_label"] = ag.loc[ag.m2_label.isna(), "label"] + M1_shift

    # === M3: ridge regression on features ===
    # Train on a paired-sample fit: for each Agarwal seq, find its 10 NN in Gosai by feature
    # then compute expected shift; fit ridge on this
    nn_g = NearestNeighbors(n_neighbors=10).fit(gos_feats)
    _, idx = nn_g.kneighbors(ag_feats)
    nn_g_means = np.array([np.nanmean(gos.iloc[i].label.values) for i in idx])
    shift_targets = nn_g_means - ag.label.to_numpy()
    # Filter rows with finite shifts
    valid = np.isfinite(shift_targets)
    if valid.sum() > 10:
        ridge = Ridge(alpha=1.0).fit(ag_feats[valid], shift_targets[valid])
        ag["m3_label"] = ag.label + ridge.predict(ag_feats)
    else:
        ridge = None
        ag["m3_label"] = ag.label + M1_shift

    # === M4: nearest-neighbor blending ===
    ag["m4_label"] = nn_g_means  # use the 10-NN mean Gosai label directly

    # === Validation: KS, quartile, σ ===
    print("\n=== Validation against Gosai ctrl_neg distribution ===")
    gos_lab = gos.label.to_numpy()
    for col in ["label", "m1_label", "m2_label", "m3_label", "m4_label"]:
        v = ag[col].to_numpy()
        ks_stat, ks_p = ks_2samp(v, gos_lab)
        diff_mean = v.mean() - gos_lab.mean()
        ratio_std = v.std() / gos_lab.std()
        print(f"  {col:<12}: mean={v.mean():+.3f} (Δ={diff_mean:+.3f}), std={v.std():.3f} (ratio={ratio_std:.2f}), KS={ks_stat:.3f} (p={ks_p:.2g})")

    # Save labelmap_table for downstream use
    out_dir = REPO / "outputs/agarwal_gosai_mapping"
    out_dir.mkdir(parents=True, exist_ok=True)
    ag_out = ag[["name", "sequence", "label", "m1_label", "m2_label", "m3_label", "m4_label", "cpg", "gc"]]
    ag_out.to_csv(out_dir / "agarwal_shuffled_with_gosai_estimates.csv", index=False)
    print(f"\nSaved to {out_dir / 'agarwal_shuffled_with_gosai_estimates.csv'}")
    if ridge is not None:
        print(f"  Ridge coefs: cpg={ridge.coef_[0]:+.3f}, gc={ridge.coef_[1]:+.3f}, entropy={ridge.coef_[2]:+.3f}, motif={ridge.coef_[3]:+.3f}, b0={ridge.intercept_:+.3f}")

    # Summary stats
    summary = {
        "ag_n": len(ag),
        "gos_n": len(gos),
        "ag_orig_mean": float(ag.label.mean()),
        "ag_orig_std": float(ag.label.std()),
        "gos_mean": float(gos.label.mean()),
        "gos_std": float(gos.label.std()),
        "M1_constant_offset": float(M1_shift),
        "M2_quartile_shifts": [float(s) for s in shifts],
        "M3_ridge_coefs": ({"cpg": float(ridge.coef_[0]), "gc": float(ridge.coef_[1]),
                           "entropy": float(ridge.coef_[2]), "motif": float(ridge.coef_[3]),
                           "intercept": float(ridge.intercept_)} if ridge is not None else None),
    }
    for col in ["m1_label", "m2_label", "m3_label", "m4_label"]:
        v = ag[col].to_numpy()
        summary[f"{col}_mean"] = float(v.mean())
        summary[f"{col}_std"] = float(v.std())
        summary[f"{col}_ks_vs_gos"] = float(ks_2samp(v, gos_lab).statistic)
    (out_dir / "mapping_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
