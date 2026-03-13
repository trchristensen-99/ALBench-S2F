"""GC-content matched random reservoir sampler."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from albench.reservoir.base import ReservoirSampler

logger = logging.getLogger(__name__)

# Yeast flanking sequences
_YEAST_FLANK_5 = "GCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCATTTTTTTCACATCTCG"
_YEAST_FLANK_3 = "GGTTACGGCTGTT"

_NUC_BYTES = np.frombuffer(b"ACGT", dtype=np.uint8)


def _gc_content(seq: str) -> float:
    """Compute GC fraction of a DNA string."""
    s = seq.upper()
    gc = s.count("G") + s.count("C")
    total = s.count("A") + s.count("C") + s.count("G") + s.count("T")
    return gc / max(total, 1)


class GCMatchedSampler(ReservoirSampler):
    """Generate random sequences whose GC content matches a reference distribution.

    Estimates the GC distribution from pool sequences, then generates random
    sequences by sampling per-position nucleotides with base probabilities
    calibrated to hit target GC fractions drawn from that distribution.
    """

    def __init__(
        self,
        seed: int | None = None,
        n_gc_bins: int = 50,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self.n_gc_bins = n_gc_bins

    def sample(
        self,
        candidates: list[str],
        n_samples: int,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Backward-compatible: random subset."""
        if n_samples > len(candidates):
            raise ValueError("n_samples cannot exceed number of candidates")
        return self._rng.choice(len(candidates), size=n_samples, replace=False).tolist()

    def generate(
        self,
        n_sequences: int,
        pool_sequences: list[str] | np.ndarray,
        task: str = "k562",
        batch_size: int = 50_000,
    ) -> tuple[list[str], pd.DataFrame]:
        """Generate random sequences with GC content matching the pool distribution.

        Args:
            n_sequences: Number of sequences to generate.
            pool_sequences: Reference pool to estimate GC distribution from.
            task: ``"k562"`` (200bp) or ``"yeast"`` (80bp random + flanks).
            batch_size: Generate in batches to limit memory.

        Returns:
            Tuple of (sequences, metadata_df).
        """
        seq_len = 200 if task == "k562" else 80

        # Estimate GC distribution from pool
        pool_gc = np.array([_gc_content(str(s)) for s in pool_sequences[:50_000]])
        logger.info(
            f"Pool GC: mean={pool_gc.mean():.3f}, std={pool_gc.std():.3f}, "
            f"range=[{pool_gc.min():.3f}, {pool_gc.max():.3f}]"
        )

        # Sample target GC fractions from the empirical distribution
        # (resample from observed values with noise)
        target_gc = self._rng.choice(pool_gc, size=n_sequences, replace=True)
        # Add small noise to smooth
        target_gc += self._rng.normal(0, 0.005, size=n_sequences)
        target_gc = np.clip(target_gc, 0.05, 0.95)

        sequences: list[str] = []
        actual_gc: list[float] = []

        for start in range(0, n_sequences, batch_size):
            n_batch = min(batch_size, n_sequences - start)
            batch_gc = target_gc[start : start + n_batch]

            # For each sequence, generate with position-independent base probs
            # P(G or C) = gc, so P(each of G,C) = gc/2, P(each of A,T) = (1-gc)/2
            # Prob vector: [A, C, G, T] = [(1-gc)/2, gc/2, gc/2, (1-gc)/2]
            for i in range(n_batch):
                gc = batch_gc[i]
                p_at = (1.0 - gc) / 2.0
                p_gc = gc / 2.0
                probs = np.array([p_at, p_gc, p_gc, p_at])
                indices = self._rng.choice(4, size=seq_len, p=probs).astype(np.uint8)
                core = _NUC_BYTES[indices].tobytes().decode("ascii")

                if task == "yeast":
                    full_seq = _YEAST_FLANK_5 + core + _YEAST_FLANK_3
                else:
                    full_seq = core

                sequences.append(full_seq)
                actual_gc.append(_gc_content(core))

        meta = pd.DataFrame(
            {
                "seq_idx": np.arange(n_sequences, dtype=np.int64),
                "method": "gc_matched_random",
                "source": "generated",
                "target_gc": target_gc,
                "actual_gc": np.array(actual_gc, dtype=np.float32),
            }
        )
        logger.info(
            f"GC-matched: generated {n_sequences:,} sequences, mean GC={np.mean(actual_gc):.3f}"
        )
        return sequences, meta
