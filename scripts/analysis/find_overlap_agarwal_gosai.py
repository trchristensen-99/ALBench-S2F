"""Find sequence overlap between Agarwal 2025 K562 large-scale (~230k) and Gosai 2024 (798k)
and learn paired mapping for the assay shift.
"""
import pandas as pd
import numpy as np
from pathlib import Path

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")

# Load Agarwal Table_S3 K562 large-scale (sequences)
print("Loading Agarwal Table_S3 K562 large-scale...")
xls = pd.ExcelFile(REPO / "data/agarwal_2025/Table_S3_large_scale_lib_design.xlsx")
# Headers are weird — first row is annotation. Try header=2 or 3.
ag_lib = pd.read_excel(xls, "K562 large-scale", header=2)
print(f"  cols: {ag_lib.columns.tolist()}")
print(f"  shape: {ag_lib.shape}")
print(ag_lib.head(3))
print()

# Detect name + sequence column
name_col = None; seq_col = None
for c in ag_lib.columns:
    cstr = str(c).lower()
    if "name" in cstr and name_col is None:
        name_col = c
    if "sequence" in cstr or "230nt" in cstr or "200nt" in cstr:
        seq_col = c
print(f"  name_col={name_col}, seq_col={seq_col}")

if name_col is None or seq_col is None:
    # Try first column as name and last as sequence
    name_col = ag_lib.columns[0]
    seq_col = ag_lib.columns[-1]
    print(f"  Falling back: name_col={name_col}, seq_col={seq_col}")

ag_lib = ag_lib[[name_col, seq_col]].dropna()
ag_lib.columns = ["name", "seq230"]
print(f"  cleaned: {len(ag_lib)} rows with name + seq")
# Extract 200nt element from 230nt (15nt adaptors on each side)
ag_lib["seq200"] = ag_lib["seq230"].astype(str).str[15:215].str.upper()
print(f"  Sample 200nt: {ag_lib.seq200.iloc[0][:50]}...")

# Load ENCFF measurements
print("\nLoading ENCFF K562 measurements...")
enc = pd.read_csv(REPO / "data/agarwal_2025/ENCFF252GNM.tsv", sep="\t")
enc = enc[enc["condition"] == "K562"]
print(f"  ENCFF K562 rows: {len(enc)}")
# Aggregate to mean log2 per element name
enc_agg = enc.groupby("name")["log2"].agg(["mean", "std", "count"]).reset_index()
print(f"  Unique elements: {len(enc_agg)}")
enc_agg.columns = ["name", "log2_mean", "log2_std", "n_reps"]

# Join sequences + labels by name
print("\nJoining seq + label...")
ag_full = ag_lib.merge(enc_agg, on="name", how="inner")
print(f"  After inner join: n={len(ag_full)} elements with seq AND label")
print(f"  log2_mean range: [{ag_full.log2_mean.min():.3f}, {ag_full.log2_mean.max():.3f}]")
print(f"  log2_mean mean: {ag_full.log2_mean.mean():+.3f}, std: {ag_full.log2_mean.std():.3f}")

# Now search for overlap with Gosai 798k
print("\nLoading Gosai 798k...")
gos = pd.read_csv(REPO / "data/k562/DATA-Table_S2__MPRA_dataset.txt", sep="\t", low_memory=False)
gos = gos[["sequence", "K562_log2FC"]].dropna()
gos["sequence"] = gos["sequence"].astype(str).str.upper()
print(f"  Gosai n: {len(gos)}")
print(f"  Gosai mean: {gos.K562_log2FC.mean():+.3f}, std: {gos.K562_log2FC.std():.3f}")

# Build hash sets for fast overlap detection
print("\nSearching for exact 200nt overlap...")
gos_seqs = set(gos["sequence"])
ag_seqs_set = set(ag_full["seq200"])
overlap = gos_seqs & ag_seqs_set
print(f"  Exact 200nt overlap: {len(overlap)} sequences")

if len(overlap) > 0:
    # Build paired dataframe: (Agarwal label, Gosai label) for shared sequences
    ag_paired = ag_full[ag_full["seq200"].isin(overlap)][["seq200", "log2_mean"]].drop_duplicates("seq200")
    gos_paired = gos[gos["sequence"].isin(overlap)][["sequence", "K562_log2FC"]].drop_duplicates("sequence")
    paired = ag_paired.rename(columns={"seq200": "sequence", "log2_mean": "agarwal_label"}).merge(
        gos_paired.rename(columns={"K562_log2FC": "gosai_label"}), on="sequence")
    print(f"  Paired sequences: n={len(paired)}")
    print(f"  Agarwal: mean={paired.agarwal_label.mean():+.3f}, std={paired.agarwal_label.std():.3f}")
    print(f"  Gosai: mean={paired.gosai_label.mean():+.3f}, std={paired.gosai_label.std():.3f}")
    print(f"  Pearson correlation: {paired[['agarwal_label', 'gosai_label']].corr().iloc[0,1]:.4f}")
    print(f"  Mean shift (Gosai - Agarwal): {(paired.gosai_label - paired.agarwal_label).mean():+.3f}")
    print(f"  Std of shift: {(paired.gosai_label - paired.agarwal_label).std():.3f}")
    # Linear regression: Gosai = a*Agarwal + b
    from scipy.stats import linregress
    res = linregress(paired.agarwal_label, paired.gosai_label)
    print(f"  Linear fit: Gosai = {res.slope:+.3f} * Agarwal + {res.intercept:+.3f}  (R²={res.rvalue**2:.3f})")
    # Save
    out_dir = REPO / "outputs/agarwal_gosai_mapping"
    out_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out_dir / "paired_agarwal_gosai_K562.csv", index=False)
    print(f"  Saved {out_dir / 'paired_agarwal_gosai_K562.csv'}")
else:
    print("  NO direct sequence overlap found.")
    # Try suffix-prefix overlap (in case sequence boundaries differ)
    print("\n  Trying substring overlap on first 150nt...")
    gos_prefix = set(s[:150] for s in gos["sequence"])
    ag_prefix = set(s[:150] for s in ag_full["seq200"])
    overlap_pfx = gos_prefix & ag_prefix
    print(f"  150nt prefix overlap: {len(overlap_pfx)}")
