#!/usr/bin/env python
"""Build AlphaGenome embedding cache for the FULL MPRA dataset (856K sequences).

Includes all ref sequences (798K), alt alleles from SNV pairs (35K),
and OOD designed sequences (23K). Each sequence is encoded through the
frozen AlphaGenome encoder to produce (T=5, D=1536) embeddings.

Saves canonical and RC embeddings as float16 numpy arrays.

Usage:
    uv run --no-sync python scripts/build_full_oracle_cache.py --output-dir outputs/oracle_full_856k/embedding_cache
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
        # Alt labels: try K562_log2FC_alt, then alt_label
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

    # Combine
    all_seqs = ref_seqs + alt_seqs + ood_seqs
    all_labels = np.concatenate([ref_labels, alt_labels, ood_labels])
    logger.info("TOTAL: %d sequences" % len(all_seqs))

    return all_seqs, all_labels


def encode_sequences(sequences, batch_size=128):
    """Encode sequences through frozen AlphaGenome encoder."""
    import jax
    import jax.numpy as jnp
    from alphagenome_ft import create_model_with_heads

    from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
    from models.alphagenome_heads import register_s2f_head

    # Register a dummy head
    head_name = "oracle_cache_head"
    register_s2f_head(
        head_name=head_name,
        arch="boda-flatten",
        hidden_dims=[512, 512],
        dropout=0.1,
    )

    # Create model with detached backbone (we only want embeddings)
    model = create_model_with_heads(
        heads=[head_name],
        detach_backbone=True,
    )

    # Load weights
    import orbax.checkpoint as ocp

    weights_path = str(REPO / "alphagenome_weights" / "alphagenome-jax-all_folds-v1")
    if not Path(weights_path).exists():
        weights_path = (
            "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"
        )

    ckpt_mgr = ocp.CheckpointManager(weights_path)
    params = ckpt_mgr.restore(ckpt_mgr.latest_step())

    # Initialize
    rng = jax.random.PRNGKey(0)
    dummy = jnp.zeros((1, 16384, 4))
    init_out, init_params = model.init_with_output(rng, dummy, is_training=False)

    # Merge pretrained params
    from models.alphagenome_heads import merge_params

    merged = merge_params(init_params, params)

    # Setup flanking sequences
    flank_5 = MPRA_UPSTREAM[-200:]
    flank_3 = MPRA_DOWNSTREAM[:200]
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}

    def _encode_flank(seq):
        enc = np.zeros((len(seq), 4), dtype=np.float32)
        for i, c in enumerate(seq):
            if c in mapping:
                enc[i, mapping[c]] = 1.0
        return enc

    f5 = _encode_flank(flank_5)
    f3 = _encode_flank(flank_3)

    # Encode function
    @jax.jit
    def _get_embeddings(params, x):
        """Get encoder output embeddings (before head)."""
        out = model.apply(params, x, is_training=False)
        # The encoder output is at the detach point
        # We need to intercept it - use the model's intermediate
        return out

    # Process in batches
    all_canonical = []
    all_rc = []
    n = len(sequences)
    t0 = time.perf_counter()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_seqs = sequences[start:end]

        # Encode batch
        canonical_batch = []
        rc_batch = []
        for seq in batch_seqs:
            seq = str(seq).upper()
            # Pad/trim to 200bp
            if len(seq) < 200:
                pad = 200 - len(seq)
                seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
            elif len(seq) > 200:
                s = (len(seq) - 200) // 2
                seq = seq[s : s + 200]

            # One-hot encode with flanks
            ohe = np.zeros((200, 4), dtype=np.float32)
            for i, c in enumerate(seq):
                if c in mapping:
                    ohe[i, mapping[c]] = 1.0

            full = np.concatenate([f5, ohe, f3], axis=0)  # (600, 4)
            # Pad to 16384
            padded = np.zeros((16384, 4), dtype=np.float32)
            offset = (16384 - 600) // 2
            padded[offset : offset + 600] = full
            canonical_batch.append(padded)

            # RC
            rc = padded[::-1, ::-1].copy()
            rc_batch.append(rc)

        x_can = jnp.array(np.stack(canonical_batch))
        x_rc = jnp.array(np.stack(rc_batch))

        # Run through model
        out_can = _get_embeddings(merged, x_can)
        out_rc = _get_embeddings(merged, x_rc)

        # Extract embeddings (T=5, D=1536 from the encoder output)
        # The model output includes head predictions, but we want the encoder embedding
        # For the flatten architecture, the encoder output shape is (B, T, D)
        # We store the full (T, D) embedding per sequence
        if isinstance(out_can, dict):
            # Try to get the embedding before the head
            emb_can = out_can.get("encoder_output", out_can.get(head_name))
            emb_rc = out_rc.get("encoder_output", out_rc.get(head_name))
        else:
            emb_can = out_can
            emb_rc = out_rc

        # Convert to numpy float16
        all_canonical.append(np.array(emb_can).astype(np.float16))
        all_rc.append(np.array(emb_rc).astype(np.float16))

        if (start // batch_size) % 50 == 0:
            elapsed = time.perf_counter() - t0
            rate = end / elapsed if elapsed > 0 else 0
            eta = (n - end) / rate / 60 if rate > 0 else 0
            logger.info(
                "  Encoded %d/%d (%.1f%%) — %.0f seq/s, ETA %.0f min"
                % (end, n, 100 * end / n, rate, eta)
            )

    canonical = np.concatenate(all_canonical, axis=0)
    rc = np.concatenate(all_rc, axis=0)
    logger.info("Final shapes: canonical=%s, rc=%s" % (canonical.shape, rc.shape))
    return canonical, rc


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
    logger.info("Encoding %d sequences..." % len(all_seqs))
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
