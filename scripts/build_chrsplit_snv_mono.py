"""Extract strictly mono chr 7+13 SNV pairs from test_snv_pairs.tsv.

Saves intermediate file with sequences + real labels, awaiting AG_S2 oracle scoring
to produce the final snv_oracle.npz drop-in replacement.

Strict mono = single ref sequence × single alt sequence per (chr, pos, ref, alt) locus
in a single sequence context (no context-expansion duplicates).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data/k562/test_sets/test_snv_pairs.tsv"
OUT = REPO / "data/k562/test_sets_ag_s2_chrsplit/snv_mono_chr7_13_intermediate.npz"


def main():
    # Group rows by variant_key (chr:pos:ref:alt); keep only those with exactly 1 row
    rows_by_key = defaultdict(list)
    with open(SRC) as f:
        header = f.readline().strip().split("\t")
        col = {c: i for i, c in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            pair_key = parts[col["pair_key"]]
            chrom = pair_key.split(":")[0]
            if chrom not in ("7", "13"):
                continue
            rows_by_key[pair_key].append(parts)

    print(f"  total distinct pair_keys on chr 7+13: {len(rows_by_key):,}")
    n_mono_keys = sum(1 for v in rows_by_key.values() if len(v) == 1)
    n_multi_keys = sum(1 for v in rows_by_key.values() if len(v) > 1)
    print(f"  strict-mono (1 row per pair_key): {n_mono_keys:,}")
    print(f"  multi-context (>1 row per pair_key): {n_multi_keys:,}")

    # Strict-mono: variants with exactly 1 row in test_snv_pairs.tsv
    mono_rows = [v[0] for v in rows_by_key.values() if len(v) == 1]
    print(f"  retained: {len(mono_rows):,}")

    pair_keys = np.array([r[col["pair_key"]] for r in mono_rows])
    ref_ids = np.array([r[col["IDs_ref"]] for r in mono_rows])
    alt_ids = np.array([r[col["IDs_alt"]] for r in mono_rows])
    ref_seqs = np.array([r[col["sequence_ref"]] for r in mono_rows])
    alt_seqs = np.array([r[col["sequence_alt"]] for r in mono_rows])
    true_ref = np.array([float(r[col["K562_log2FC_ref"]]) for r in mono_rows], dtype=np.float32)
    true_alt = np.array([float(r[col["K562_log2FC_alt"]]) for r in mono_rows], dtype=np.float32)
    true_delta = np.array([float(r[col["delta_log2FC"]]) for r in mono_rows], dtype=np.float32)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        pair_keys=pair_keys,
        ref_ids=ref_ids,
        alt_ids=alt_ids,
        ref_sequences=ref_seqs,
        alt_sequences=alt_seqs,
        true_ref_label=true_ref,
        true_alt_label=true_alt,
        true_delta=true_delta,
    )
    print(f"  saved: {OUT}")
    print(f"  shapes: ref_sequences={ref_seqs.shape}  alt_sequences={alt_seqs.shape}")


if __name__ == "__main__":
    main()
