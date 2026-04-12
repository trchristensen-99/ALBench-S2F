#!/usr/bin/env python
"""Generate synthetic negatives with corrected Gosai-scale labels.

The original negatives used a z-score mapping (mean=-0.454, std=0.617) that
overestimates both the magnitude and spread of random DNA activity on Gosai scale.

This script uses the piecewise-linear QQ transform fitted on the inactive region
of Agarwal vs Gosai data:
  Inactive (Agarwal <= 0): Gosai = 1.296 * Agarwal + 0.522

This gives corrected Gosai-scale labels:
  Shuffled controls: mean=-0.158, std=0.355
  (vs original:      mean=-0.454, std=0.617)

We also test an intermediate transform using the global QQ linear fit:
  Global: Gosai = 2.130 * Agarwal + 0.864
  -> Shuffled controls: mean=-0.254, std=0.584

Usage:
    python scripts/generate_corrected_negatives.py
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# Agarwal shuffled control measurements (ENCODE raw scale)
AGAR_SHUF_MEAN = -0.5252
AGAR_SHUF_STD = 0.2740

# Three Agarwal->Gosai transforms to test
TRANSFORMS = {
    "piecewise": {
        "desc": "Inactive-region piecewise linear (QQ, Agar<=0)",
        "slope": 1.296,
        "intercept": 0.522,
    },
    "global_qq": {
        "desc": "Global QQ linear",
        "slope": 2.130,
        "intercept": 0.864,
    },
    "original": {
        "desc": "Original z-score mapping",
        "slope": 2.252,  # gosai_std / enc_std
        "intercept": 0.728,  # derived from original enc_to_gosai
    },
}


def dinucleotide_shuffle(seq: str, rng: np.random.Generator) -> str:
    """Shuffle sequence preserving dinucleotide frequencies."""
    seq = seq.upper()
    from collections import defaultdict as dd

    edges = dd(list)
    for i in range(len(seq) - 1):
        edges[seq[i]].append(seq[i + 1])
    for base in edges:
        rng.shuffle(edges[base])
    result = [seq[0]]
    idx = dd(int)
    for _ in range(len(seq) - 1):
        cur = result[-1]
        if idx[cur] < len(edges[cur]):
            result.append(edges[cur][idx[cur]])
            idx[cur] += 1
        else:
            result.append(rng.choice(list("ACGT")))
    return "".join(result)


def generate_negatives(
    out_dir: Path,
    transform_name: str,
    slope: float,
    intercept: float,
    n_random: int = 50_000,
    n_dinuc: int = 50_000,
    n_gc: int = 50_000,
    seed: int = 42,
):
    """Generate synthetic negatives with given transform parameters."""
    rng = np.random.default_rng(seed)

    gosai_mean = slope * AGAR_SHUF_MEAN + intercept
    gosai_std = abs(slope) * AGAR_SHUF_STD

    log.info(
        "Transform '%s': Gosai-scale labels mean=%.3f, std=%.3f",
        transform_name,
        gosai_mean,
        gosai_std,
    )

    all_seqs = []
    all_labels = []
    all_cats = []

    def sample_label():
        return rng.normal(gosai_mean, gosai_std)

    # 1. Random DNA
    log.info("  Generating %d random 200bp sequences...", n_random)
    for _ in range(n_random):
        seq = "".join(rng.choice(list("ACGT"), size=200))
        all_seqs.append(seq)
        all_labels.append(sample_label())
        all_cats.append("random_negative")

    # 2. Dinucleotide-shuffled from Gosai training data
    log.info("  Generating %d dinucleotide-shuffled sequences...", n_dinuc)
    gosai_path = REPO / "data" / "k562" / "DATA-Table_S2__MPRA_dataset.txt"
    if gosai_path.exists():
        import pandas as pd

        df = pd.read_csv(gosai_path, sep="\t", low_memory=False, usecols=["sequence"])
        train_seqs = df["sequence"].dropna().values
        indices = rng.choice(len(train_seqs), size=n_dinuc, replace=True)
        for idx in indices:
            seq = str(train_seqs[idx])[:200]
            if len(seq) >= 198:
                shuffled = dinucleotide_shuffle(seq, rng)
                all_seqs.append(shuffled)
                all_labels.append(sample_label())
                all_cats.append("dinuc_shuffled_negative")
    else:
        log.warning("Gosai data not found, generating random DNA instead of dinuc shuffle")
        for _ in range(n_dinuc):
            all_seqs.append("".join(rng.choice(list("ACGT"), size=200)))
            all_labels.append(sample_label())
            all_cats.append("random_negative")

    # 3. GC-matched random
    log.info("  Generating %d GC-matched random sequences...", n_gc)
    gc_mean, gc_std = 0.462, 0.106  # from Gosai training data
    for _ in range(n_gc):
        target_gc = np.clip(rng.normal(gc_mean, gc_std), 0.2, 0.8)
        n_gc_bases = int(200 * target_gc)
        n_at_bases = 200 - n_gc_bases
        bases = ["G"] * (n_gc_bases // 2) + ["C"] * (n_gc_bases - n_gc_bases // 2)
        bases += ["A"] * (n_at_bases // 2) + ["T"] * (n_at_bases - n_at_bases // 2)
        rng.shuffle(bases)
        all_seqs.append("".join(bases))
        all_labels.append(sample_label())
        all_cats.append("gc_matched_negative")

    # Write combined TSV
    out_path = out_dir / f"negatives_{transform_name}.tsv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["sequence", "K562_log2FC", "category"])
        for seq, label, cat in zip(all_seqs, all_labels, all_cats):
            w.writerow([seq, f"{label:.6f}", cat])

    # Also write dinuc-only TSV (for configs that use dinuc_shuffled_negatives.tsv)
    dinuc_path = out_dir / f"dinuc_shuffled_{transform_name}.tsv"
    with open(dinuc_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["sequence", "K562_log2FC", "category"])
        for seq, label, cat in zip(all_seqs, all_labels, all_cats):
            if cat == "dinuc_shuffled_negative":
                w.writerow([seq, f"{label:.6f}", cat])

    metadata = {
        "transform": transform_name,
        "slope": slope,
        "intercept": intercept,
        "gosai_label_mean": gosai_mean,
        "gosai_label_std": gosai_std,
        "agarwal_shuf_mean": AGAR_SHUF_MEAN,
        "agarwal_shuf_std": AGAR_SHUF_STD,
        "n_random": n_random,
        "n_dinuc": n_dinuc,
        "n_gc": n_gc,
        "seed": seed,
    }
    with open(out_dir / f"metadata_{transform_name}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("  Saved: %s (%d sequences)", out_path, len(all_seqs))
    return gosai_mean, gosai_std


def main():
    out_base = REPO / "data" / "synthetic_negatives_corrected"
    out_base.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERATING CORRECTED SYNTHETIC NEGATIVES")
    print("=" * 70)
    print()

    results = {}
    for name, params in TRANSFORMS.items():
        print(f"\n--- {name}: {params['desc']} ---")
        g_mean, g_std = generate_negatives(
            out_base,
            name,
            params["slope"],
            params["intercept"],
        )
        results[name] = {"mean": g_mean, "std": g_std}

    print("\n" + "=" * 70)
    print("SUMMARY OF LABEL DISTRIBUTIONS")
    print("=" * 70)
    print(f"{'Transform':20s} {'Gosai mean':>12s} {'Gosai std':>12s}")
    for name, r in results.items():
        print(f"{name:20s} {r['mean']:12.3f} {r['std']:12.3f}")

    print(f"\nAll outputs in: {out_base}")


if __name__ == "__main__":
    main()
