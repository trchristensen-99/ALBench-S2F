"""Process Sahu STARR-seq random library FASTQ to extract CpG-vs-activity
relationship.

Strategy (RNA-only, no input normalization):
- Each unique 150bp insert appears N times in RNA reads
- N is proportional to (plasmid abundance) × (transcriptional activity)
- For a TRULY RANDOM library (Sahu N150), plasmid abundance is roughly
  uniform across unique sequences (synthesized in equimolar pool)
- So read_count per unique sequence ≈ activity proxy

Caveats:
- Without DNA-input normalization, sequences with PCR amplification
  bias get counted higher. But the ENSEMBLE relationship should still
  hold if we average over many sequences within each CpG bin.
- The library has ~10⁹ possible 150bp seqs, so most are seen 0-1 times.
  We aggregate by CpG-density bin to extract signal from noise.

Outputs:
- outputs/sahu_random_lib_analysis/cpg_vs_count.csv (CpG bin × count distribution)
- outputs/sahu_random_lib_analysis/cpg_vs_count_correlation.json
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FQ = REPO / "external/sahu_geo/sra/SRR15147156.fastq.gz"
OUT_DIR = REPO / "outputs/sahu_random_lib_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def cpg_density(seq: str) -> float:
    if len(seq) <= 1:
        return 0.0
    return sum(1 for i in range(len(seq) - 1) if seq[i] == "C" and seq[i + 1] == "G") / (
        len(seq) - 1
    )


def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def main():
    print(f"Reading {FQ} ...")
    counts: Counter = Counter()
    n_reads = 0
    with gzip.open(FQ, "rt") as g:
        for i, line in enumerate(g):
            if i % 4 == 1:  # sequence line
                seq = line.strip().upper()
                # Only keep ACGT, skip reads with N
                if "N" in seq:
                    continue
                counts[seq] += 1
                n_reads += 1
                if n_reads % 1_000_000 == 0:
                    print(f"  parsed {n_reads:,} reads, {len(counts):,} unique")

    print(f"\nTotal valid reads: {n_reads:,}")
    print(f"Unique sequences: {len(counts):,}")

    # Distribution of duplicate counts
    dup_dist = Counter(counts.values())
    print("\n=== Duplicate-count distribution ===")
    for dc in sorted(dup_dist)[:15]:
        n_seqs = dup_dist[dc]
        print(f"  count={dc}: {n_seqs:,} seqs ({100 * n_seqs / len(counts):.1f}%)")

    # Compute features per unique sequence
    print("\nComputing CpG/GC features...")
    rows = []
    for seq, cnt in counts.items():
        rows.append(
            {
                "count": cnt,
                "cpg": cpg_density(seq),
                "gc": gc_content(seq),
                "len": len(seq),
            }
        )
    df = pd.DataFrame(rows)
    print(f"  feature df: {len(df):,} rows")
    print(
        f"  CpG density: mean={df['cpg'].mean():.4f} std={df['cpg'].std():.4f} q05={df['cpg'].quantile(0.05):.4f} q95={df['cpg'].quantile(0.95):.4f}"
    )
    print(f"  GC content: mean={df['gc'].mean():.4f} std={df['gc'].std():.4f}")
    print(
        f"  Read len: median={int(df['len'].median())}, range=[{df['len'].min()}, {df['len'].max()}]"
    )

    # CpG-bin aggregation (deciles)
    df["cpg_bin"] = pd.qcut(df["cpg"], 10, labels=False, duplicates="drop")
    bin_stats = (
        df.groupby("cpg_bin")
        .agg(
            n_seqs=("count", "size"),
            cpg_mean=("cpg", "mean"),
            cpg_min=("cpg", "min"),
            cpg_max=("cpg", "max"),
            count_mean=("count", "mean"),
            count_median=("count", "median"),
            count_max=("count", "max"),
        )
        .reset_index()
    )
    print("\n=== CpG-density bins vs read counts ===")
    print(bin_stats.to_string(index=False))

    # Correlation: CpG vs raw count (across all unique seqs)
    from scipy.stats import pearsonr, spearmanr

    r, p_r = pearsonr(df["cpg"], df["count"])
    rho, p_rho = spearmanr(df["cpg"], df["count"])
    print("\n=== Correlation: CpG density vs read count (RNA only, no DNA normalization) ===")
    print(f"  pearson r = {r:+.4f} (p={p_r:.2g})")
    print(f"  spearman ρ = {rho:+.4f} (p={p_rho:.2g})")

    # Save CSV
    out_csv = OUT_DIR / "cpg_vs_count_bins.csv"
    bin_stats.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    summary = {
        "fastq_source": str(FQ.relative_to(REPO)),
        "n_total_reads": int(n_reads),
        "n_unique_sequences": int(len(counts)),
        "median_read_length": int(df["len"].median()),
        "cpg_density": {
            "mean": float(df["cpg"].mean()),
            "std": float(df["cpg"].std()),
            "q05": float(df["cpg"].quantile(0.05)),
            "q95": float(df["cpg"].quantile(0.95)),
        },
        "gc_content": {
            "mean": float(df["gc"].mean()),
            "std": float(df["gc"].std()),
        },
        "cpg_vs_count_correlation": {
            "pearson_r": float(r),
            "pearson_p": float(p_r),
            "spearman_rho": float(rho),
            "spearman_p": float(p_rho),
            "note": "RNA-only — without DNA-input normalization this proxy is noisy but interpretable in aggregate",
        },
    }
    out_json = OUT_DIR / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
