#!/usr/bin/env python
"""Score ctrl_neg (negative-control intergenic) Gosai sequences with the
chr-split AG_S2 oracle ensemble.

Produces data/k562/test_sets_ag_s2_chrsplit/ctrl_neg_oracle.npz with the
same schema as genomic_oracle.npz: sequences / true_label / oracle_mean.

The ctrl_neg subset is filtered out of the full Gosai dataset
(class == "ctrl_neg") with the standard quality filters applied so that
the comparison plot is on the same population the model trained against
on non-ctrl chromosomes.

Usage:
    uv run --no-sync python scripts/score_ctrl_neg_ag_s2.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def _quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same quality filters used by K562Dataset._load_and_filter_data."""
    # Reference alleles + non-variant only (matches default include_alt_alleles=False)
    id_parts = df["IDs"].str.split(":", expand=True)
    allele_type = id_parts[4]
    ref_col = id_parts[2]
    alt_col = id_parts[3]
    is_reference = allele_type == "R"
    is_non_variant = (ref_col == "NA") & (alt_col == "NA")
    df = df[is_reference | is_non_variant].copy()

    # Project filter
    if "data_project" in df.columns:
        df = df[df["data_project"].isin(["UKBB", "GTEX", "CRE"])].reset_index(drop=True)

    # Stderr quality filter
    stderr_cols = [c for c in df.columns if c.endswith("_lfcSE")]
    if stderr_cols:
        df = df[df[stderr_cols].max(axis=1) < 1.0].reset_index(drop=True)

    # Outlier removal (±6σ with +4 upper shift)
    activity_cols = [c for c in df.columns if c.endswith("_log2FC")]
    if activity_cols:
        means = df[activity_cols].mean().to_numpy()
        stds = df[activity_cols].std().to_numpy()
        up_cut = means + stds * 6.0 + 4.0
        down_cut = means - stds * 6.0
        b_up = (df[activity_cols] < up_cut).all(axis=1)
        b_down = (df[activity_cols] > down_cut).all(axis=1)
        df = df[b_up & b_down].reset_index(drop=True)

    # Length filter
    df["seq_len"] = df["sequence"].str.len()
    df = df[df["seq_len"] >= 198].copy().drop(columns=["seq_len"])
    return df


def _standardize_to_200bp(seq: str) -> str:
    if len(seq) < 200:
        pad = 200 - len(seq)
        return "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
    if len(seq) > 200:
        start = (len(seq) - 200) // 2
        return seq[start : start + 200]
    return seq


def main():
    from experiments.exp1_1_scaling import _load_oracle

    out_dir = REPO / "data" / "k562" / "test_sets_ag_s2_chrsplit"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = REPO / "data" / "k562"

    logger.info("Loading Gosai data...")
    df = pd.read_csv(
        data_path / "DATA-Table_S2__MPRA_dataset.txt", sep="\t", dtype={"OL": str}, low_memory=False
    )
    logger.info(f"  raw rows: {len(df):,}")

    logger.info("Applying quality filters...")
    df = _quality_filter(df)
    logger.info(f"  quality-filtered: {len(df):,}")

    if "class" not in df.columns:
        raise RuntimeError("ctrl_neg cannot be selected — no 'class' column in Gosai dataset")
    ctrl = df[df["class"] == "ctrl_neg"].reset_index(drop=True)
    logger.info(f"  ctrl_neg subset: {len(ctrl):,}")
    if len(ctrl) == 0:
        raise RuntimeError("No ctrl_neg rows after filter — abort")

    seqs = [_standardize_to_200bp(str(s)) for s in ctrl["sequence"].tolist()]
    true_labels = ctrl["K562_log2FC"].values.astype(np.float32)

    logger.info("Loading AG_S2 oracle (chr-split natural ensemble)...")
    oracle = _load_oracle("k562", oracle_type="ag_s2")

    logger.info(f"Scoring {len(seqs):,} ctrl_neg sequences with 10-fold ensemble...")
    oracle_mean = oracle.predict(seqs).astype(np.float32)

    out_path = out_dir / "ctrl_neg_oracle.npz"
    np.savez_compressed(
        out_path,
        sequences=np.array(seqs, dtype=object),
        true_label=true_labels,
        oracle_mean=oracle_mean,
    )
    logger.info(f"Saved {out_path}  (n={len(seqs):,})")
    logger.info(f"  true_label μ={true_labels.mean():+.3f} σ={true_labels.std():.3f}")
    logger.info(f"  oracle_mean μ={oracle_mean.mean():+.3f} σ={oracle_mean.std():.3f}")


if __name__ == "__main__":
    main()
