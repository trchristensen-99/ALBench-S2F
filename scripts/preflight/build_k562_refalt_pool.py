"""Build the K562 ref+alt + boda2-filtered + chromosome-split pool.

Output (under outputs/oracle_pseudolabels_k562_ag_s2_refalt/pool/):
    train.parquet  — train pool sequences + log2FC labels
    val.parquet    — val pool (chr19+21+X)
    test.parquet   — held-out chromosomes (e.g. chr8+9)
    snv_pairs.parquet — REF/ALT pairs for SNV-effect eval
    summary.json   — counts + chromosome assignments

Bypasses K562Dataset (which requires BLAST for include_alt_alleles) by
loading the raw TSV directly and applying boda2 filters in-script.

Run once:
    uv run --no-sync python scripts/preflight/build_k562_refalt_pool.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

# Chromosome splits per existing project convention (chr19+21+X val,
# chr8+9 test). Train is everything else.
VAL_CHROMS = {"19", "21", "X"}
TEST_CHROMS = {"8", "9"}


def main():
    fp = REPO / "data" / "k562" / "DATA-Table_S2__MPRA_dataset.txt"
    print(f"Loading raw TSV from {fp} …")
    df = pd.read_csv(fp, sep="\t", dtype={"OL": str})
    print(f"  raw rows: {len(df):,}")

    # ── Allele filter: keep ref + alt + non-variant CRE rows ─────────────
    id_parts = df["IDs"].str.split(":", expand=True)
    chrom_str = id_parts[0]
    allele_type = id_parts[4]
    ref_col = id_parts[2]
    alt_col = id_parts[3]
    is_ref = allele_type == "R"
    is_alt = allele_type == "A"
    is_non_var = (ref_col == "NA") & (alt_col == "NA")
    df = df[is_ref | is_alt | is_non_var].copy()
    df["chrom"] = chrom_str.values
    df["allele_type"] = allele_type.values
    df["ref_col"] = ref_col.values
    df["alt_col"] = alt_col.values
    print(f"  after ref+alt allele filter: {len(df):,}")

    # ── boda2 quality filters ────────────────────────────────────────────
    n_pre = len(df)
    df = df[df["data_project"].isin(["UKBB", "GTEX", "CRE"])].copy()
    print(f"  after project filter (UKBB/GTEX/CRE):  {len(df):,}  (-{n_pre - len(df):,})")

    n_pre = len(df)
    se_cols = [c for c in df.columns if c.endswith("_lfcSE")]
    df = df[df[se_cols].max(axis=1) < 1.0]
    print(f"  after stderr filter (max SE < 1.0):    {len(df):,}  (-{n_pre - len(df):,})")

    n_pre = len(df)
    act_cols = [c for c in df.columns if c.endswith("_log2FC")]
    means = df[act_cols].mean().to_numpy()
    stds = df[act_cols].std().to_numpy()
    up = means + stds * 6 + 4
    down = means - stds * 6
    df = df[(df[act_cols] < up).all(axis=1) & (df[act_cols] > down).all(axis=1)]
    print(f"  after outlier filter (mean ± 6σ + 4u): {len(df):,}  (-{n_pre - len(df):,})")

    n_pre = len(df)
    df = df[df["sequence"].str.len() >= 198].reset_index(drop=True)
    print(f"  after length filter (>= 198bp):        {len(df):,}  (-{n_pre - len(df):,})")

    # ── Chromosome split ─────────────────────────────────────────────────
    is_val = df["chrom"].isin(VAL_CHROMS)
    is_test = df["chrom"].isin(TEST_CHROMS)
    is_train = ~(is_val | is_test)
    train = df[is_train].reset_index(drop=True)
    val = df[is_val].reset_index(drop=True)
    test = df[is_test].reset_index(drop=True)
    print(f"\nChromosome split:\n  train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")

    # ── SNV pairs: pair ref/alt rows on same locus (same chr:pos:ref:alt) ─
    snv_pairs_full = []
    for split_name, sub in [("train", train), ("val", val), ("test", test)]:
        # locus key: chrom:pos:ref_base:alt_base
        sub = sub.copy()
        sub["locus"] = (
            id_parts.loc[sub.index, 0].astype(str)
            + ":"
            + id_parts.loc[sub.index, 1].astype(str)
            + ":"
            + sub["ref_col"]
            + ":"
            + sub["alt_col"]
        )
        # variant rows only (R or A), pair by locus
        var_rows = sub[sub["allele_type"].isin(["R", "A"])]
        ref_rows = var_rows[var_rows["allele_type"] == "R"][
            ["locus", "sequence", "K562_log2FC"]
        ].rename(columns={"sequence": "sequence_ref", "K562_log2FC": "ref_log2FC"})
        alt_rows = var_rows[var_rows["allele_type"] == "A"][
            ["locus", "sequence", "K562_log2FC"]
        ].rename(columns={"sequence": "sequence_alt", "K562_log2FC": "alt_log2FC"})
        pairs = ref_rows.merge(alt_rows, on="locus", how="inner")
        pairs["split"] = split_name
        snv_pairs_full.append(pairs)
        print(f"  SNV pairs ({split_name}): {len(pairs):,}")
    snv_pairs = pd.concat(snv_pairs_full, ignore_index=True)

    # ── Save ─────────────────────────────────────────────────────────────
    out_dir = REPO / "outputs" / "oracle_pseudolabels_k562_ag_s2_refalt" / "pool"
    out_dir.mkdir(parents=True, exist_ok=True)

    keep_cols = ["sequence", "K562_log2FC", "HepG2_log2FC", "SKNSH_log2FC", "chrom", "allele_type"]
    train[keep_cols].to_parquet(out_dir / "train.parquet")
    val[keep_cols].to_parquet(out_dir / "val.parquet")
    test[keep_cols].to_parquet(out_dir / "test.parquet")
    snv_pairs.to_parquet(out_dir / "snv_pairs.parquet")

    summary = {
        "raw_rows": int(len(df)),
        "split_counts": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "snv_pairs_by_split": {
            s: int((snv_pairs["split"] == s).sum()) for s in ("train", "val", "test")
        },
        "filter_chain": [
            "ref+alt allele",
            "project ∈ {UKBB,GTEX,CRE}",
            "max SE < 1.0",
            "mean ± 6σ +4 upper",
            "len >= 198 bp",
        ],
        "val_chroms": sorted(VAL_CHROMS),
        "test_chroms": sorted(TEST_CHROMS),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved pool to {out_dir}\n  summary.json:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
