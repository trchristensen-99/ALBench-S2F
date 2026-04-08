#!/usr/bin/env python
"""Generate AG S2 oracle test labels for chr-split evaluation.

Creates oracle_labeled NPZ files for the chr-split test sets:
  - test_in_dist: chr7+13 sequences (~31K)
  - test_snv: SNV pairs on chr7+13 (~3K pairs)
  - test_ood: designed sequences (~23K)

Labels all sequences with the live AG S2 10-fold ensemble, then saves
in the format expected by evaluation/exp1_eval.py:
  {test_set}_oracle.npz with keys: sequences, oracle_mean, [ref/alt for SNV]

Usage:
    uv run --no-sync python scripts/generate_ag_s2_test_labels.py
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


def main():
    from data.k562 import K562Dataset
    from experiments.exp1_1_scaling import _load_oracle

    out_dir = REPO / "data" / "k562" / "test_sets_ag_s2_chrsplit"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = REPO / "data" / "k562"

    logger.info("Loading AG S2 oracle...")
    oracle = _load_oracle("k562", oracle_type="ag_s2")

    # 1. In-distribution (chr7+13)
    logger.info("Loading chr-split in-dist test set...")
    ds = K562Dataset(
        data_path=str(data_path),
        split="test",
        use_hashfrag=False,
        use_chromosome_fallback=True,
    )
    seqs = list(ds.sequences)
    logger.info(f"  {len(seqs)} in-dist sequences (chr7+13)")

    logger.info("  Labeling with AG S2 oracle...")
    labels = oracle.predict(seqs)
    np.savez_compressed(
        out_dir / "genomic_oracle.npz",
        sequences=np.array(seqs, dtype=object),
        oracle_mean=labels,
        true_label=ds.labels.astype(np.float32),
    )
    logger.info(f"  Saved genomic_oracle.npz ({len(seqs)} seqs)")

    # 2. SNV pairs (chr7+13 only)
    logger.info("Loading SNV pairs...")
    snv_path = data_path / "test_sets" / "test_snv_pairs_hashfrag.tsv"
    snv_df = pd.read_csv(snv_path, sep="\t")

    # Filter to chr7+13
    chroms = snv_df["IDs_ref"].str.split(":", expand=True)[0]
    test_chrs = {"chr7", "chr13", "7", "13"}
    mask = chroms.isin(test_chrs)
    snv_chr = snv_df[mask].reset_index(drop=True)
    logger.info(f"  {len(snv_chr)} SNV pairs on chr7+13 (of {len(snv_df)} total)")

    ref_seqs = snv_chr["sequence_ref"].tolist()
    alt_seqs = snv_chr["sequence_alt"].tolist()

    logger.info("  Labeling ref sequences...")
    ref_labels = oracle.predict(ref_seqs)
    logger.info("  Labeling alt sequences...")
    alt_labels = oracle.predict(alt_seqs)
    delta_labels = alt_labels - ref_labels

    np.savez_compressed(
        out_dir / "snv_oracle.npz",
        ref_sequences=np.array(ref_seqs, dtype=object),
        alt_sequences=np.array(alt_seqs, dtype=object),
        ref_mean=ref_labels,
        alt_mean=alt_labels,
        delta_mean=delta_labels,
        true_alt_label=snv_chr.get(
            "K562_log2FC_alt", snv_chr.get("alt_label", np.zeros(len(snv_chr)))
        ).values.astype(np.float32)
        if "K562_log2FC_alt" in snv_chr.columns or "alt_label" in snv_chr.columns
        else np.zeros(len(snv_chr), dtype=np.float32),
        true_delta=snv_chr.get(
            "delta_log2FC", snv_chr.get("delta_label", np.zeros(len(snv_chr)))
        ).values.astype(np.float32)
        if "delta_log2FC" in snv_chr.columns or "delta_label" in snv_chr.columns
        else np.zeros(len(snv_chr), dtype=np.float32),
    )
    logger.info(f"  Saved snv_oracle.npz ({len(snv_chr)} pairs)")

    # 3. OOD designed sequences
    logger.info("Loading OOD sequences...")
    ood_path = data_path / "test_sets" / "test_ood_designed_k562.tsv"
    ood_df = pd.read_csv(ood_path, sep="\t")
    ood_seqs = ood_df["sequence"].tolist()
    logger.info(f"  {len(ood_seqs)} OOD sequences")

    logger.info("  Labeling with AG S2 oracle...")
    ood_labels = oracle.predict(ood_seqs)
    np.savez_compressed(
        out_dir / "ood_oracle.npz",
        sequences=np.array(ood_seqs, dtype=object),
        oracle_mean=ood_labels,
        true_label=ood_df["K562_log2FC"].values.astype(np.float32)
        if "K562_log2FC" in ood_df.columns
        else np.zeros(len(ood_seqs), dtype=np.float32),
    )
    logger.info(f"  Saved ood_oracle.npz ({len(ood_seqs)} seqs)")

    # 4. Random 10K (for calibration)
    logger.info("Generating 10K random sequences...")
    rng = np.random.default_rng(42)
    random_seqs = ["".join(rng.choice(list("ACGT"), size=200)) for _ in range(10000)]
    random_labels = oracle.predict(random_seqs)
    np.savez_compressed(
        out_dir / "random_10k_oracle.npz",
        sequences=np.array(random_seqs, dtype=object),
        oracle_mean=random_labels,
    )
    logger.info("  Saved random_10k_oracle.npz")

    logger.info(f"\nAll test labels saved to {out_dir}")


if __name__ == "__main__":
    main()
