"""Recombination reservoir sampler — crossover of pool sequences."""

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


class RecombinationSampler(ReservoirSampler):
    """Generate chimeric sequences by recombining pairs of pool sequences.

    For each output sequence, randomly selects two parent sequences and
    performs 1-point or 2-point crossover on the mutable region. This explores
    novel combinations of functional elements present in the genomic pool.
    """

    def __init__(
        self,
        seed: int | None = None,
        crossover_mode: str = "uniform",
        n_crossover_points: int = 2,
    ) -> None:
        """Initialize sampler.

        Args:
            seed: Random seed.
            crossover_mode: ``"uniform"`` (per-position coin flip),
                ``"block"`` (n-point crossover).
            n_crossover_points: Number of crossover points for block mode.
        """
        self._rng = np.random.default_rng(seed)
        self.crossover_mode = crossover_mode
        self.n_crossover_points = n_crossover_points

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

    def _extract_region(self, seq: str, task: str) -> str:
        """Extract the mutable region from a sequence."""
        if task == "yeast":
            s = str(seq)
            if s.startswith(_YEAST_FLANK_5) and s.endswith(_YEAST_FLANK_3):
                return s[len(_YEAST_FLANK_5) : -len(_YEAST_FLANK_3)]
            return s[:80] if len(s) >= 80 else s
        return str(seq)

    def _crossover(self, parent_a: str, parent_b: str) -> str:
        """Perform crossover between two equal-length sequences."""
        n = min(len(parent_a), len(parent_b))
        a = list(parent_a[:n])
        b = list(parent_b[:n])

        if self.crossover_mode == "uniform":
            # Per-position coin flip
            mask = self._rng.random(n) < 0.5
            return "".join(b[i] if mask[i] else a[i] for i in range(n))
        else:
            # N-point crossover
            points = sorted(
                self._rng.choice(n - 1, size=min(self.n_crossover_points, n - 1), replace=False) + 1
            )
            result = list(a)
            use_b = False
            prev = 0
            for pt in points:
                if use_b:
                    result[prev:pt] = b[prev:pt]
                prev = pt
                use_b = not use_b
            if use_b:
                result[prev:] = b[prev:]
            return "".join(result)

    def generate(
        self,
        n_sequences: int,
        base_sequences: list[str] | np.ndarray,
        task: str = "k562",
    ) -> tuple[list[str], pd.DataFrame]:
        """Generate chimeric sequences by recombining pairs from the pool.

        Args:
            n_sequences: Number of chimeric sequences to produce.
            base_sequences: Pool of parent sequences.
            task: ``"k562"`` or ``"yeast"``.

        Returns:
            Tuple of (chimeric_sequences, metadata_df).
        """
        n_pool = len(base_sequences)
        parent_a_idx = self._rng.choice(n_pool, size=n_sequences, replace=True)
        parent_b_idx = self._rng.choice(n_pool, size=n_sequences, replace=True)

        sequences: list[str] = []
        for i in range(n_sequences):
            region_a = self._extract_region(base_sequences[parent_a_idx[i]], task)
            region_b = self._extract_region(base_sequences[parent_b_idx[i]], task)
            child = self._crossover(region_a, region_b)

            if task == "yeast":
                full_seq = _YEAST_FLANK_5 + child + _YEAST_FLANK_3
            else:
                full_seq = child

            sequences.append(full_seq)

        meta = pd.DataFrame(
            {
                "seq_idx": np.arange(n_sequences, dtype=np.int64),
                "method": f"recombination_{self.crossover_mode}",
                "source": "recombined",
                "parent_a_idx": parent_a_idx,
                "parent_b_idx": parent_b_idx,
            }
        )
        logger.info(
            f"Recombination ({self.crossover_mode}): generated {n_sequences:,} chimeric sequences"
        )
        return sequences, meta
