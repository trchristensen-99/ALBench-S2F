"""Re-analyze Sahu N150 random library FASTQ to disentangle CpG from motif effects.

Counts per-sequence:
  - CpG density
  - GC content
  - Hits for K562-relevant motifs (SP1, E2F, CRE, EBOX, YY1, CTCF, GATA)
  - Read count (activity proxy)

Then fits multivariate regression: log(count) ~ CpG + GC + motif_counts.
Identifies which features are CAUSAL (motif) vs CONFOUNDED (CpG).
"""
import gzip
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
FQ = REPO / "external/sahu_geo/sra/SRR15147156.fastq.gz"
OUT_DIR = REPO / "outputs/sahu_motif_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# K562 TF motif catalog (forward strand consensus; we'll check both strands)
# Source: JASPAR core motifs for K562-enriched TFs
MOTIFS = {
    "SP1":    ["GGGGCGGGGC", "GCCCCGCCCC"],          # GC-box, palindrome
    "E2F":    ["TTTCGCGC", "GCGCGAAA"],
    "CRE":    ["TGACGTCA", "ATGACGTCA"],             # AP-1/CREB
    "EBOX":   ["CACGTG", "CAGCTG", "CATGTG"],        # Myc/Max/USF
    "YY1":    ["CCGCCATNTT", "CCATNTTG"],            # NOT used (N is tricky)
    "GATA":   ["AGATAA", "TTATCT"],
    "CTCF":   ["CCCTC", "GAGGG"],
    "ETS":    ["GGAA", "TTCC"],
    "TATA":   ["TATAAA", "TTTATA"],
}


def cpg(s):
    return sum(1 for i in range(len(s)-1) if s[i]=="C" and s[i+1]=="G") / max(len(s)-1, 1)

def gc(s):
    return (s.count("G") + s.count("C")) / max(len(s), 1)

# Compile motif counters as raw substring search (forward + RC)
RC_MAP = str.maketrans("ACGTN", "TGCAN")
def rc(s): return s.translate(RC_MAP)[::-1]

motif_lookups = {}
for name, patterns in MOTIFS.items():
    pats = set()
    for p in patterns:
        if "N" in p: continue  # skip ambiguous
        pats.add(p)
        pats.add(rc(p))
    motif_lookups[name] = list(pats)


def motif_count(seq, name):
    return sum(seq.count(p) for p in motif_lookups[name])

motif_keys = sorted([k for k, v in motif_lookups.items() if v])
print(f"Motif features: {motif_keys}")
print()

# --- Read FASTQ, aggregate per-sequence ---
print(f"Reading {FQ}...")
counts = Counter()
n_reads = 0
with gzip.open(FQ, "rt") as f:
    for i, line in enumerate(f):
        if i % 4 == 1:
            s = line.strip().upper()
            if "N" in s: continue
            counts[s] += 1
            n_reads += 1
            if n_reads % 2_000_000 == 0:
                print(f"  parsed {n_reads:,} reads, {len(counts):,} unique")

print(f"\nTotal reads: {n_reads:,}, unique: {len(counts):,}")

# Build feature matrix
# Subsample for speed if too many sequences
seqs = list(counts.keys())
if len(seqs) > 1_000_000:
    print(f"Subsampling 1M sequences for feature extraction")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(seqs), size=1_000_000, replace=False)
    seqs = [seqs[i] for i in idx]

print(f"Computing features for {len(seqs):,} sequences...")
features = {"cpg": [], "gc": [], "count": []}
for k in motif_keys: features[k] = []

for i, s in enumerate(seqs):
    features["cpg"].append(cpg(s))
    features["gc"].append(gc(s))
    features["count"].append(counts[s])
    for k in motif_keys:
        features[k].append(motif_count(s, k))
    if (i + 1) % 200_000 == 0:
        print(f"  processed {i+1:,}/{len(seqs):,}")

df = pd.DataFrame(features)
print(f"\nFeature df: {df.shape}")
print(df.describe())

# Save
df.to_parquet(OUT_DIR / "sahu_seq_features.parquet")
print(f"Saved {OUT_DIR / 'sahu_seq_features.parquet'}")

# Multivariate regression: log(count+1) ~ all features
import statsmodels.api as sm
y = np.log(df["count"] + 1)
X = df[["cpg", "gc"] + motif_keys]
X = sm.add_constant(X)

print("\n=== Multivariate OLS: log(count+1) ~ CpG + GC + motifs ===")
model = sm.OLS(y, X).fit()
print(model.summary().tables[1])

# Now: CpG-alone regression for comparison
print("\n=== CpG-only OLS ===")
X_cpg = sm.add_constant(df["cpg"])
model_cpg = sm.OLS(y, X_cpg).fit()
print(f"  CpG-alone slope: {model_cpg.params['cpg']:.4f}  R²={model_cpg.rsquared:.4f}")

# Motifs-only (no CpG)
print("\n=== Motifs-only OLS (no CpG) ===")
X_motifs = sm.add_constant(df[motif_keys + ["gc"]])
model_motifs = sm.OLS(y, X_motifs).fit()
print(f"  R²={model_motifs.rsquared:.4f}")
print(f"  vs full model R²={model.rsquared:.4f}")

# Save summary
summary = {
    "n_seqs": int(len(df)),
    "log_count_mean": float(y.mean()),
    "log_count_std": float(y.std()),
    "cpg_alone_slope": float(model_cpg.params["cpg"]),
    "cpg_alone_r2": float(model_cpg.rsquared),
    "motifs_only_r2": float(model_motifs.rsquared),
    "full_model_r2": float(model.rsquared),
    "full_model_coefs": {k: float(v) for k, v in model.params.to_dict().items()},
    "full_model_pvalues": {k: float(v) for k, v in model.pvalues.to_dict().items()},
}
(OUT_DIR / "motif_regression_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nSaved {OUT_DIR / 'motif_regression_summary.json'}")
