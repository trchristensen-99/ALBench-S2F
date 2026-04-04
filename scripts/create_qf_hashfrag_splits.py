#!/usr/bin/env python
"""Create quality-filtered hashfrag split index files from existing unfiltered splits.

The unfiltered hashfrag splits at data/k562/hashfrag_splits/ contain indices
into the full ~401K ref-only dataset. After quality filtering (stderr < 1.0,
outlier removal, project filter), the dataset shrinks to ~377K sequences.

This script:
1. Loads the full data file and determines which rows pass quality filters
2. Creates a mapping from unfiltered indices to filtered indices
3. Remaps the existing hashfrag split indices to the filtered space
4. Saves to hashfrag_splits_qf/ (ref-only) and optionally hashfrag_splits_qf_alt/ (ref+alt)

This avoids needing BLAST+ to regenerate splits from scratch.

Usage:
    python scripts/create_qf_hashfrag_splits.py [--data-dir data/k562]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_and_filter(file_path: Path, include_alt: bool = False) -> tuple[np.ndarray, int]:
    """Load data, apply quality filters, return (kept_original_indices, n_total_before_filter).

    Returns indices into the allele-filtered (but not quality-filtered) dataset.
    """
    df = pd.read_csv(file_path, sep="\t", dtype={"OL": str})
    n_raw = len(df)

    # 1. Allele filter
    id_parts = df["IDs"].str.split(":", expand=True)
    allele_type = id_parts[4]
    ref_col = id_parts[2]
    alt_col = id_parts[3]

    if include_alt:
        is_valid = allele_type.isin(["R", "A"]) | ((ref_col == "NA") & (alt_col == "NA"))
    else:
        is_reference = allele_type == "R"
        is_non_variant = (ref_col == "NA") & (alt_col == "NA")
        is_valid = is_reference | is_non_variant

    df = df[is_valid].copy()
    n_after_allele = len(df)
    print(f"  Allele filter: {n_raw} -> {n_after_allele}")

    # 2. Project filter
    if "data_project" in df.columns:
        allowed = ["UKBB", "GTEX", "CRE"]
        mask = df["data_project"].isin(allowed).values
    else:
        mask = np.ones(len(df), dtype=bool)

    # 3. Stderr filter
    stderr_cols = [c for c in df.columns if c.endswith("_lfcSE")]
    if stderr_cols:
        mask &= (df[stderr_cols].max(axis=1) < 1.0).values

    # 4. Outlier filter
    activity_cols = [c for c in df.columns if c.endswith("_log2FC")]
    if activity_cols:
        means = df[activity_cols].mean().to_numpy()
        stds = df[activity_cols].std().to_numpy()
        up_cut = means + stds * 6.0 + 4.0
        down_cut = means - stds * 6.0
        mask &= (df[activity_cols].values < up_cut).all(axis=1)
        mask &= (df[activity_cols].values > down_cut).all(axis=1)

    # 5. Length filter
    mask &= (df["sequence"].str.len() >= 198).values

    n_after_qf = int(mask.sum())
    print(f"  Quality filter: {n_after_allele} -> {n_after_qf}")

    # Return the indices of the kept rows in the allele-filtered space
    kept_in_allele_space = np.where(mask)[0]
    return kept_in_allele_space, n_after_allele


def remap_splits(
    old_splits_dir: Path,
    new_splits_dir: Path,
    kept_indices: np.ndarray,
    n_allele_filtered: int,
) -> None:
    """Remap split index files from unfiltered to filtered space."""
    # Build reverse map: old_idx -> new_idx (or -1 if filtered out)
    remap = np.full(n_allele_filtered, -1, dtype=np.int64)
    remap[kept_indices] = np.arange(len(kept_indices))

    new_splits_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        old_file = old_splits_dir / f"{split_name}_indices.npy"
        if not old_file.exists():
            print(f"  WARNING: {old_file} not found, skipping")
            continue

        old_idx = np.load(old_file)
        # Filter: keep only indices that pass quality filter
        valid_mask = old_idx < n_allele_filtered
        old_idx_valid = old_idx[valid_mask]
        new_idx = remap[old_idx_valid]
        new_idx = new_idx[new_idx >= 0]  # drop filtered-out sequences

        out_file = new_splits_dir / f"{split_name}_indices.npy"
        np.save(out_file, new_idx)
        print(
            f"  {split_name}: {len(old_idx)} -> {len(new_idx)} (dropped {len(old_idx) - len(new_idx)})"
        )

    # Handle legacy pool_indices.npy
    pool_file = old_splits_dir / "pool_indices.npy"
    if pool_file.exists():
        old_idx = np.load(pool_file)
        valid_mask = old_idx < n_allele_filtered
        old_idx_valid = old_idx[valid_mask]
        new_idx = remap[old_idx_valid]
        new_idx = new_idx[new_idx >= 0]
        out_file = new_splits_dir / "pool_indices.npy"
        np.save(out_file, new_idx)
        print(f"  pool: {len(old_idx)} -> {len(new_idx)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/k562")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    file_path = data_dir / "DATA-Table_S2__MPRA_dataset.txt"
    old_splits_dir = data_dir / "hashfrag_splits"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    if not old_splits_dir.exists():
        raise FileNotFoundError(f"Old splits not found: {old_splits_dir}")

    # Ref-only QF splits
    print("\n=== Creating hashfrag_splits_qf (ref-only, quality-filtered) ===")
    kept_ref, n_ref = load_and_filter(file_path, include_alt=False)
    remap_splits(old_splits_dir, data_dir / "hashfrag_splits_qf", kept_ref, n_ref)

    # Ref+alt QF splits (if we had unfiltered ref+alt splits — we don't,
    # so skip unless they exist)
    alt_splits_dir = data_dir / "hashfrag_splits_alt"
    if alt_splits_dir.exists():
        print("\n=== Creating hashfrag_splits_qf_alt (ref+alt, quality-filtered) ===")
        kept_alt, n_alt = load_and_filter(file_path, include_alt=True)
        remap_splits(alt_splits_dir, data_dir / "hashfrag_splits_qf_alt", kept_alt, n_alt)
    else:
        print("\n  No ref+alt hashfrag splits found (hashfrag_splits_alt/). Skipping.")

    print("\nDone!")


if __name__ == "__main__":
    main()
