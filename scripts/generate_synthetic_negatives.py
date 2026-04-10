#!/usr/bin/env python
"""Generate synthetic negative sequences for oracle bias correction.

Creates three types of synthetic negatives:
  1. Random DNA (50K): purely random 200bp sequences
  2. Dinucleotide-shuffled (1 per training sequence): preserves dinuc freq
  3. GC-matched random (50K): random seqs matching training GC distribution

Each gets a label drawn from the empirical Agarwal shuffled control distribution.

Usage:
    uv run --no-sync python scripts/generate_synthetic_negatives.py \
        --output-dir data/synthetic_negatives
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def dinucleotide_shuffle(seq: str, rng: np.random.Generator) -> str:
    """Shuffle sequence preserving dinucleotide frequencies (Altschul-Erickson)."""
    seq = seq.upper()
    # Build edge list for each dinucleotide
    from collections import defaultdict

    edges = defaultdict(list)
    for i in range(len(seq) - 1):
        edges[seq[i]].append(seq[i + 1])

    # Shuffle each edge list
    for base in edges:
        rng.shuffle(edges[base])

    # Reconstruct: Euler path through the dinucleotide graph
    result = [seq[0]]
    idx = defaultdict(int)
    for _ in range(len(seq) - 1):
        cur = result[-1]
        if idx[cur] < len(edges[cur]):
            nxt = edges[cur][idx[cur]]
            idx[cur] += 1
            result.append(nxt)
        else:
            # Fallback: random base
            result.append(rng.choice(list("ACGT")))
    return "".join(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO / "data" / "synthetic_negatives")
    parser.add_argument("--n-random", type=int, default=50000)
    parser.add_argument("--n-gc-matched", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Load Agarwal shuffled control distribution for label generation
    # Real measured: mean=-0.525, std=0.274 (ENCODE raw scale)
    # On Gosai scale: mean=-0.454, std=0.618 (from metadata transform)
    gosai_mean, gosai_std = 0.5120, 1.1981
    enc_mean, enc_std = -0.0958, 0.5322
    shuf_enc_mean, shuf_enc_std = -0.5252, 0.2740

    def enc_to_gosai(val):
        return (val - enc_mean) / enc_std * gosai_std + gosai_mean

    shuf_gosai_mean = enc_to_gosai(shuf_enc_mean)
    shuf_gosai_std = shuf_enc_std / enc_std * gosai_std
    logger.info(
        "Shuffled control distribution (Gosai scale): mean=%.3f, std=%.3f",
        shuf_gosai_mean,
        shuf_gosai_std,
    )

    def sample_negative_label():
        """Sample a label from the shuffled control distribution."""
        return rng.normal(shuf_gosai_mean, shuf_gosai_std)

    # ═══════════════════════════════════════════
    # 1. Random DNA sequences
    # ═══════════════════════════════════════════
    logger.info("Generating %d random 200bp sequences...", args.n_random)
    random_seqs = []
    random_labels = []
    for _ in range(args.n_random):
        seq = "".join(rng.choice(list("ACGT"), size=200))
        random_seqs.append(seq)
        random_labels.append(sample_negative_label())

    with open(args.output_dir / "random_negatives.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["sequence", "K562_log2FC", "category"])
        for seq, label in zip(random_seqs, random_labels):
            w.writerow([seq, f"{label:.6f}", "random_negative"])
    logger.info("  Saved random_negatives.tsv")

    # ═══════════════════════════════════════════
    # 2. Dinucleotide-shuffled training sequences
    # ═══════════════════════════════════════════
    logger.info("Generating dinucleotide-shuffled versions of training sequences...")
    from data.k562 import K562Dataset

    train_ds = K562Dataset(data_path=str(REPO / "data" / "k562"), split="train")
    train_seqs = list(train_ds.sequences)

    # Subsample to 50K for manageable size
    n_shuffle = min(50000, len(train_seqs))
    shuffle_idx = rng.choice(len(train_seqs), size=n_shuffle, replace=False)

    shuffled_seqs = []
    shuffled_labels = []
    for i in shuffle_idx:
        orig = train_seqs[i]
        shuffled = dinucleotide_shuffle(orig, rng)
        shuffled_seqs.append(shuffled)
        shuffled_labels.append(sample_negative_label())

    with open(args.output_dir / "dinuc_shuffled_negatives.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["sequence", "K562_log2FC", "category"])
        for seq, label in zip(shuffled_seqs, shuffled_labels):
            w.writerow([seq, f"{label:.6f}", "dinuc_shuffled_negative"])
    logger.info("  Saved dinuc_shuffled_negatives.tsv (%d sequences)", len(shuffled_seqs))

    # ═══════════════════════════════════════════
    # 3. GC-matched random sequences
    # ═══════════════════════════════════════════
    logger.info("Generating %d GC-matched random sequences...", args.n_gc_matched)

    # Compute GC distribution of training sequences
    gc_contents = []
    for seq in train_seqs[:50000]:
        gc = sum(1 for c in seq.upper() if c in "GC") / len(seq)
        gc_contents.append(gc)
    gc_mean = np.mean(gc_contents)
    gc_std = np.std(gc_contents)
    logger.info("  Training GC: mean=%.3f, std=%.3f", gc_mean, gc_std)

    gc_matched_seqs = []
    gc_matched_labels = []
    for _ in range(args.n_gc_matched):
        # Sample target GC content from training distribution
        target_gc = np.clip(rng.normal(gc_mean, gc_std), 0.1, 0.9)
        n_gc = int(round(target_gc * 200))
        n_at = 200 - n_gc
        bases = (
            ["G"] * (n_gc // 2)
            + ["C"] * (n_gc - n_gc // 2)
            + ["A"] * (n_at // 2)
            + ["T"] * (n_at - n_at // 2)
        )
        rng.shuffle(bases)
        seq = "".join(bases)
        gc_matched_seqs.append(seq)
        gc_matched_labels.append(sample_negative_label())

    with open(args.output_dir / "gc_matched_negatives.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["sequence", "K562_log2FC", "category"])
        for seq, label in zip(gc_matched_seqs, gc_matched_labels):
            w.writerow([seq, f"{label:.6f}", "gc_matched_negative"])
    logger.info("  Saved gc_matched_negatives.tsv")

    # ═══════════════════════════════════════════
    # 4. Combined file for easy loading
    # ═══════════════════════════════════════════
    with open(args.output_dir / "all_negatives.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["sequence", "K562_log2FC", "category"])
        for seq, label in zip(random_seqs, random_labels):
            w.writerow([seq, f"{label:.6f}", "random"])
        for seq, label in zip(shuffled_seqs, shuffled_labels):
            w.writerow([seq, f"{label:.6f}", "dinuc_shuffled"])
        for seq, label in zip(gc_matched_seqs, gc_matched_labels):
            w.writerow([seq, f"{label:.6f}", "gc_matched"])

    # Save metadata
    meta = {
        "n_random": len(random_seqs),
        "n_dinuc_shuffled": len(shuffled_seqs),
        "n_gc_matched": len(gc_matched_seqs),
        "total": len(random_seqs) + len(shuffled_seqs) + len(gc_matched_seqs),
        "label_distribution": {
            "source": "Agarwal shuffled controls (ENCODE raw → Gosai scale)",
            "gosai_mean": float(shuf_gosai_mean),
            "gosai_std": float(shuf_gosai_std),
            "encode_raw_mean": float(shuf_enc_mean),
            "encode_raw_std": float(shuf_enc_std),
        },
        "gc_stats": {"training_mean": float(gc_mean), "training_std": float(gc_std)},
        "seed": args.seed,
    }
    with open(args.output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nGenerated {meta['total']:,} synthetic negatives in {args.output_dir}")
    print(f"  Random: {len(random_seqs):,}")
    print(f"  Dinuc-shuffled: {len(shuffled_seqs):,}")
    print(f"  GC-matched: {len(gc_matched_seqs):,}")


if __name__ == "__main__":
    main()
