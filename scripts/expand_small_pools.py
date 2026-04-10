#!/usr/bin/env python
"""Expand small 2M pools (dinuc_shuffle, gc_matched) by point-mutating existing sequences."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

BASES = list("ACGT")


def main():
    rng = np.random.default_rng(42)
    target = 2000000

    for strat in ["dinuc_shuffle", "gc_matched"]:
        pool_path = Path("outputs/labeled_pools_2m/k562/ag_s2/%s/pool.npz" % strat)
        if not pool_path.exists():
            print("%s: pool not found" % strat)
            continue

        d = np.load(pool_path, allow_pickle=True)
        seqs = d["sequences"]
        labels = d["labels"]
        n_have = len(seqs)

        if n_have >= target:
            print("%s: already %d sequences" % (strat, n_have))
            continue

        n_new = target - n_have
        print("%s: expanding from %d to %d (+%d via mutations)..." % (strat, n_have, target, n_new))

        # Generate mutated sequences
        new_seqs = []
        for _ in range(n_new):
            idx = rng.integers(0, n_have)
            seq = list(str(seqs[idx]))
            n_mut = rng.integers(1, 6)
            for __ in range(n_mut):
                pos = rng.integers(0, len(seq))
                seq[pos] = rng.choice(BASES)
            new_seqs.append("".join(seq))

        # Label with oracle
        print("  Labeling %d new sequences with AG S2 oracle..." % n_new)
        from experiments.exp1_1_scaling import _load_oracle

        oracle = _load_oracle("k562", oracle_type="ag_s2")
        new_labels = oracle.predict(new_seqs)
        del oracle

        # Combine and save
        all_seqs = np.concatenate([seqs, np.array(new_seqs, dtype=object)])
        all_labels = np.concatenate([labels, new_labels])
        metadata = str(d["metadata"]) + " | expanded to 2M via 1-5 point mutations"
        np.savez_compressed(
            pool_path, sequences=all_seqs, labels=all_labels, metadata=np.array(metadata)
        )
        print("  Saved: %d sequences" % len(all_seqs))

    print("Done")


if __name__ == "__main__":
    main()
