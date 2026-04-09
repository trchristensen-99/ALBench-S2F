#!/usr/bin/env python
"""Build AlphaGenome embedding cache for the FULL MPRA dataset (856K sequences).

Includes all ref sequences (798K), alt alleles from SNV pairs (35K),
and OOD designed sequences (23K). Each sequence is encoded through the
frozen AlphaGenome encoder to produce (T=5, D=1536) embeddings.

Saves canonical and RC embeddings as float16 numpy arrays.

Usage:
    uv run --no-sync python scripts/build_full_oracle_cache.py \
        --output-dir outputs/oracle_full_856k/embedding_cache
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_all_sequences():
    """Load ALL measured MPRA sequences with their K562 labels."""
    data_path = REPO / "data" / "k562"

    # 1. Full MPRA dataset (ref sequences)
    df = pd.read_csv(data_path / "DATA-Table_S2__MPRA_dataset.txt", sep="\t", low_memory=False)
    ref_seqs = df["sequence"].tolist()
    ref_labels = df["K562_log2FC"].values.astype(np.float32)
    logger.info("Ref sequences: %d" % len(ref_seqs))

    # 2. Alt alleles from SNV pairs
    snv_path = data_path / "test_sets" / "test_snv_pairs_hashfrag.tsv"
    if snv_path.exists():
        snv_df = pd.read_csv(snv_path, sep="\t")
        alt_seqs = snv_df["sequence_alt"].tolist()
        if "K562_log2FC_alt" in snv_df.columns:
            alt_labels = snv_df["K562_log2FC_alt"].values.astype(np.float32)
        elif "alt_label" in snv_df.columns:
            alt_labels = snv_df["alt_label"].values.astype(np.float32)
        else:
            alt_labels = np.zeros(len(alt_seqs), dtype=np.float32)
            logger.warning("No alt labels found, using zeros")
        logger.info("Alt allele sequences: %d" % len(alt_seqs))
    else:
        alt_seqs, alt_labels = [], np.array([], dtype=np.float32)

    # 3. OOD designed sequences
    ood_path = data_path / "test_sets" / "test_ood_designed_k562.tsv"
    if ood_path.exists():
        ood_df = pd.read_csv(ood_path, sep="\t")
        ood_seqs = ood_df["sequence"].tolist()
        ood_labels = ood_df["K562_log2FC"].values.astype(np.float32)
        logger.info("OOD designed sequences: %d" % len(ood_seqs))
    else:
        ood_seqs, ood_labels = [], np.array([], dtype=np.float32)

    # Combine all
    all_seqs = ref_seqs + alt_seqs + ood_seqs
    all_labels = np.concatenate([ref_labels, alt_labels, ood_labels])
    logger.info("TOTAL: %d sequences" % len(all_seqs))

    return all_seqs, all_labels


def encode_sequences(sequences, batch_size=128):
    """Encode sequences through frozen AlphaGenome encoder.

    Initializes the AG model, extracts the encoder function, then uses
    _encode_sequences_for_ag for batched encoding with flanking/RC.
    """
    from experiments.exp1_1_scaling import _encode_sequences_for_ag, _get_ag_model_and_encoder

    logger.info("Initializing AlphaGenome encoder...")
    ag = _get_ag_model_and_encoder("k562")
    encoder_fn = ag["encoder_fn"]
    logger.info("  Encoder ready.")

    logger.info("Encoding %d sequences through AG encoder..." % len(sequences))
    t0 = time.perf_counter()

    # _encode_sequences_for_ag returns (N, T, D) float16 array (canonical only)
    # We need to call it twice: once for canonical, once for RC
    # Actually, looking at the function, it only returns canonical embeddings.
    # For the oracle training cache, we need both canonical and RC.
    # Let's encode canonical and RC separately.

    canonical = _encode_sequences_for_ag(list(sequences), "k562", encoder_fn, batch_size)
    logger.info("  Canonical done: %s" % (canonical.shape,))

    # For RC: reverse-complement the sequences and encode again
    def _rc_seq(seq):
        comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
        return "".join(comp.get(c, "N") for c in reversed(seq.upper()))

    rc_sequences = [_rc_seq(s) for s in sequences]
    rc = _encode_sequences_for_ag(rc_sequences, "k562", encoder_fn, batch_size)
    logger.info("  RC done: %s" % (rc.shape,))

    elapsed = time.perf_counter() - t0
    logger.info(
        "Encoded %d sequences in %.1f min (%.0f seq/s)"
        % (len(sequences), elapsed / 60, len(sequences) / max(elapsed, 1))
    )

    return canonical.astype(np.float16), rc.astype(np.float16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    done_file = args.output_dir / ".cache_done"
    if done_file.exists():
        logger.info("Cache already built, skipping.")
        return

    # Load all sequences
    all_seqs, all_labels = load_all_sequences()

    # Save labels
    np.save(args.output_dir / "all_labels.npy", all_labels)
    logger.info("Saved labels: %s" % (all_labels.shape,))

    # Encode
    canonical, rc = encode_sequences(all_seqs, batch_size=args.batch_size)

    # Save
    np.save(args.output_dir / "train_canonical.npy", canonical)
    np.save(args.output_dir / "train_rc.npy", rc)
    logger.info(
        "Saved: canonical=%s (%.1f GB), rc=%s (%.1f GB)"
        % (canonical.shape, canonical.nbytes / 1e9, rc.shape, rc.nbytes / 1e9)
    )

    done_file.touch()
    logger.info("Done.")


if __name__ == "__main__":
    main()
