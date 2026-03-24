#!/usr/bin/env python3
"""Build OOD designed test sets for K562, HepG2, and SK-N-SH from OL46 validation data.

Source: DATA-MPRA_Datasets.zip from Zenodo record 10698014
  - MPRA_Datasets/Validation/OL46.attributes  (design method + target cell)
  - MPRA_Datasets/Validation/OL46_K562_20220720.out  (K562 expression)
  - MPRA_Datasets/Validation/OL46_HepG2_20220720.out (HepG2 expression)
  - MPRA_Datasets/Validation/OL46_SKNSH_20220720.out (SK-N-SH expression)
  - MPRA_Datasets/Fastas/OL46_reference.fasta.gz (200bp sequences)

Filters to cell-line-targeting sequences from CODA design algorithms.

Usage:
    python scripts/build_multicell_ood_designed.py \
        --mpra-dir data/zenodo_10698014/MPRA_Datasets
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# Cell-line-specific designed-sequence project labels in OL46
CELL_LINE_PROJECTS = {
    "k562": {
        "BODA:fsp:k562",
        "BODA:fsp_uc:k562",
        "BODA:sa:k562",
        "BODA:al:k562",
    },
    "hepg2": {
        "BODA:fsp:hepg2",
        "BODA:fsp_uc:hepg2",
        "BODA:sa:hepg2",
        "BODA:al:hepg2",
    },
    "sknsh": {
        "BODA:fsp:sknsh",
        "BODA:fsp_uc:sknsh",
        "BODA:sa:sknsh",
        "BODA:al:sknsh",
    },
}

# Expression data files per cell line
EXPR_FILES = {
    "k562": "OL46_K562_20220720.out",
    "hepg2": "OL46_HepG2_20220720.out",
    "sknsh": "OL46_SKNSH_20220720.out",
}

LABEL_COLS = {
    "k562": "K562_log2FC",
    "hepg2": "HepG2_log2FC",
    "sknsh": "SKNSH_log2FC",
}

METHOD_MAP = {
    "BODA:fsp": "FastSeqProp",
    "BODA:sa": "Simulated_Annealing",
    "BODA:al": "AdaLead",
}


def load_fasta(path: Path) -> dict[str, str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    fasta: dict[str, str] = {}
    current_id = None
    with opener(path, "rt") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                current_id = line[1:]
                fasta[current_id] = []
            elif current_id is not None:
                fasta[current_id].append(line)
    return {k: "".join(v) for k, v in fasta.items()}


def build_ood_for_cell_line(
    cell_line: str,
    attrs_df: pd.DataFrame,
    fasta: dict[str, str],
    mpra_dir: Path,
) -> pd.DataFrame | None:
    """Build OOD test set for one cell line."""
    projects = CELL_LINE_PROJECTS.get(cell_line, set())
    fc_col = LABEL_COLS[cell_line]

    # Filter to this cell line's designed sequences
    mask = attrs_df["project"].isin(projects)
    designed = attrs_df[mask].copy()

    if len(designed) == 0:
        print(f"  {cell_line}: No designed sequences found for projects {projects}")
        # Try broader matching
        all_projects = sorted(attrs_df["project"].unique())
        cell_projects = [p for p in all_projects if cell_line in p.lower()]
        print(f"  Available projects containing '{cell_line}': {cell_projects}")
        if cell_projects:
            mask = attrs_df["project"].isin(cell_projects)
            designed = attrs_df[mask].copy()

    if len(designed) == 0:
        return None

    # Load expression data
    expr_file = mpra_dir / "Validation" / EXPR_FILES.get(cell_line, "")
    if not expr_file.exists():
        # Try alternative naming
        for f in (mpra_dir / "Validation").glob(f"OL46_{cell_line}*"):
            expr_file = f
            break

    if not expr_file.exists():
        print(f"  {cell_line}: Expression file not found: {expr_file}")
        return None

    expr_df = pd.read_csv(expr_file, sep="\t")
    print(f"  {cell_line}: Loaded {len(expr_df)} expression entries from {expr_file.name}")

    # Merge
    designed["sequence"] = designed.index.map(fasta)
    designed = designed.dropna(subset=["sequence"])

    # Extract method name
    designed["method"] = designed["project"].apply(
        lambda p: next((v for k, v in METHOD_MAP.items() if p.startswith(k)), p)
    )
    designed["target_cell"] = cell_line

    # Merge expression
    if "ID" in expr_df.columns:
        expr_df = expr_df.set_index("ID")
    elif expr_df.index.name != "ID":
        expr_df.index.name = "ID"

    # Try to merge on index
    merged = designed.join(expr_df[[c for c in expr_df.columns if "log2FC" in c or "lfcSE" in c]])

    # Rename columns to standard format
    for col in merged.columns:
        if "log2FC" in col and fc_col not in merged.columns:
            merged = merged.rename(columns={col: fc_col})
        if "lfcSE" in col:
            se_col = fc_col.replace("log2FC", "lfcSE")
            if se_col not in merged.columns:
                merged = merged.rename(columns={col: se_col})

    merged = merged.dropna(subset=[fc_col]) if fc_col in merged.columns else merged

    result = merged.reset_index()
    result = result.rename(columns={"index": "ID"}) if "ID" not in result.columns else result

    cols = ["ID", "sequence", "method", "target_cell"]
    if fc_col in result.columns:
        cols.append(fc_col)
    se_col = fc_col.replace("log2FC", "lfcSE")
    if se_col in result.columns:
        cols.append(se_col)

    result = result[[c for c in cols if c in result.columns]]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpra-dir", required=True, help="Path to extracted MPRA_Datasets/")
    args = parser.parse_args()

    mpra_dir = Path(args.mpra_dir)

    # Load OL46 attributes
    attrs_path = mpra_dir / "Validation" / "OL46.attributes"
    if not attrs_path.exists():
        # Try finding it
        for f in mpra_dir.rglob("OL46.attributes"):
            attrs_path = f
            break
    print(f"Loading attributes from {attrs_path}")
    attrs_df = pd.read_csv(attrs_path, sep="\t", index_col=0)
    print(f"  {len(attrs_df)} total OL46 entries")
    print(f"  Projects: {sorted(attrs_df['data_project'].unique())}")

    # Load sequences
    fasta_path = mpra_dir / "Fastas" / "OL46_reference.fasta.gz"
    if not fasta_path.exists():
        for f in mpra_dir.rglob("OL46_reference.fasta*"):
            fasta_path = f
            break
    print(f"Loading sequences from {fasta_path}")
    fasta = load_fasta(fasta_path)
    print(f"  {len(fasta)} sequences loaded")

    # Build OOD for each cell line
    for cell_line in ["k562", "hepg2", "sknsh"]:
        print(f"\n=== {cell_line.upper()} ===")
        result = build_ood_for_cell_line(cell_line, attrs_df, fasta, mpra_dir)
        if result is not None and len(result) > 0:
            out_dir = REPO / "data" / cell_line / "test_sets"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"test_ood_designed_{cell_line}.tsv"
            result.to_csv(out_path, sep="\t", index=False)
            print(f"  Saved {len(result)} sequences to {out_path}")

            # Counts by method
            if "method" in result.columns:
                print(f"  Methods: {dict(result['method'].value_counts())}")
        else:
            print(f"  FAILED: No OOD data for {cell_line}")

    print("\nDone.")


if __name__ == "__main__":
    main()
