#!/usr/bin/env python3
"""Create test sets for HepG2 and SK-N-SH from existing K562 data files.

The raw MPRA data (DATA-Table_S2__MPRA_dataset.txt) already contains
expression measurements for all three cell lines. The same sequences
are measured in K562, HepG2, and SK-N-SH.

This script creates:
  - In-distribution test set (from HashFrag test split)
  - SNV pairs test set (reference + alternate allele pairs)
  - OOD designed test set (from OL46 validation library, if available)

For in-dist and SNV, we reuse the same HashFrag splits and sequences
as K562 — only the label column changes.

Usage:
    python scripts/create_cellline_test_sets.py --cell-line hepg2
    python scripts/create_cellline_test_sets.py --cell-line sknsh
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# Column mappings
CELL_LINE_COLS = {
    "k562": {"fc": "K562_log2FC", "se": "K562_lfcSE"},
    "hepg2": {"fc": "HepG2_log2FC", "se": "HepG2_lfcSE"},
    "sknsh": {"fc": "SKNSH_log2FC", "se": "SKNSH_lfcSE"},
}


def create_in_dist_test(cell_line: str, out_dir: Path) -> None:
    """Create in-distribution test set by copying K562 version with correct labels."""
    k562_file = REPO / "data" / "k562" / "test_sets" / "test_in_distribution_hashfrag.tsv"
    if not k562_file.exists():
        print(f"  SKIP in-dist: {k562_file} not found")
        return

    df = pd.read_csv(k562_file, sep="\t")
    cols = CELL_LINE_COLS[cell_line]

    # Check that the cell line columns exist
    if cols["fc"] not in df.columns:
        print(f"  SKIP in-dist: column {cols['fc']} not in data")
        return

    # Drop rows where this cell line has NaN
    n_before = len(df)
    df = df.dropna(subset=[cols["fc"]])
    print(f"  In-dist: {n_before} → {len(df)} after dropping NaN {cols['fc']}")

    out_path = out_dir / f"test_in_distribution_hashfrag.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"  Saved: {out_path} ({len(df)} rows)")


def create_snv_test(cell_line: str, out_dir: Path) -> None:
    """Create SNV pairs test set with cell-line-specific labels."""
    k562_file = REPO / "data" / "k562" / "test_sets" / "test_snv_pairs_hashfrag.tsv"
    if not k562_file.exists():
        print(f"  SKIP SNV: {k562_file} not found")
        return

    df = pd.read_csv(k562_file, sep="\t")
    cols = CELL_LINE_COLS[cell_line]

    # The SNV file has K562-specific columns. We need to check if there are
    # cell-line-specific columns or if we need to join from raw data.
    fc_ref_col = f"{cols['fc']}_ref"
    fc_alt_col = f"{cols['fc']}_alt"

    if fc_ref_col in df.columns:
        # Already has per-cell-line columns
        df[f"delta_{cols['fc']}"] = df[fc_alt_col] - df[fc_ref_col]
        df = df.dropna(subset=[fc_ref_col, fc_alt_col])
    else:
        # Need to join from raw data to get HepG2/SKNSH labels for these variants
        raw_file = REPO / "data" / "k562" / "DATA-Table_S2__MPRA_dataset.txt"
        if not raw_file.exists():
            print(f"  SKIP SNV: need raw data for {cell_line} labels")
            return

        raw = pd.read_csv(raw_file, sep="\t", usecols=["IDs", cols["fc"], cols["se"]])

        # Join ref labels
        ref_labels = raw.rename(columns={"IDs": "IDs_ref", cols["fc"]: fc_ref_col})
        df = df.merge(ref_labels[["IDs_ref", fc_ref_col]], on="IDs_ref", how="left")

        # Join alt labels
        alt_labels = raw.rename(columns={"IDs": "IDs_alt", cols["fc"]: fc_alt_col})
        df = df.merge(alt_labels[["IDs_alt", fc_alt_col]], on="IDs_alt", how="left")

        df[f"delta_{cols['fc']}"] = df[fc_alt_col] - df[fc_ref_col]
        n_before = len(df)
        df = df.dropna(subset=[fc_ref_col, fc_alt_col])
        print(f"  SNV: {n_before} → {len(df)} after dropping NaN")

    out_path = out_dir / f"test_snv_pairs_hashfrag.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"  Saved: {out_path} ({len(df)} rows)")


def create_ood_test(cell_line: str, out_dir: Path) -> None:
    """Create OOD test set from K562 OOD designed sequences.

    Since OL46 designed sequences were targeted at K562, we can still
    evaluate HepG2/SKNSH on these same sequences (cross-cell-line OOD).
    We also check for cell-line-specific designed sequences if available.
    """
    k562_ood = REPO / "data" / "k562" / "test_sets" / "test_ood_designed_k562.tsv"
    if not k562_ood.exists():
        print(f"  SKIP OOD: {k562_ood} not found")
        return

    # The K562 OOD file only has K562 labels. We need to get HepG2/SKNSH
    # labels for these same sequences from the OL46 validation data.
    # For now, mark this as needing the raw OL46 data with all cell lines.
    print(
        f"  OOD: Need OL46 validation data with {cell_line} expression. "
        f"Download DATA-MPRA_Datasets.zip from Zenodo 10698014."
    )

    # Create a placeholder manifest
    import json

    manifest = {
        "cell_line": cell_line,
        "note": "OOD test set requires OL46 validation data with per-cell-line expression",
        "zenodo": "10698014 / DATA-MPRA_Datasets.zip",
        "status": "pending",
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell-line",
        required=True,
        choices=["hepg2", "sknsh"],
    )
    args = parser.parse_args()

    out_dir = REPO / "data" / args.cell_line / "test_sets"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating test sets for {args.cell_line.upper()}")
    print(f"Output: {out_dir}")

    create_in_dist_test(args.cell_line, out_dir)
    create_snv_test(args.cell_line, out_dir)
    create_ood_test(args.cell_line, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
