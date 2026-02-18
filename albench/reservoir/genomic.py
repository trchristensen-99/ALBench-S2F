"""Genomic reservoir sampler based on held-out chromosomes."""

from __future__ import annotations

from typing import Any

import numpy as np

from albench.reservoir.base import ReservoirSampler


class GenomicSampler(ReservoirSampler):
    """Sample candidates restricted to predefined held-out chromosomes."""

    def __init__(self, chromosomes: list[str] | None = None, seed: int | None = None) -> None:
        """Initialize genomic sampler.

        Args:
            chromosomes: Allowed chromosomes for sampling.
            seed: Random seed for reproducibility.
        """
        self.chromosomes = set(chromosomes or ["chr7", "chr13"])
        self._rng = np.random.default_rng(seed)

    def sample(
        self,
        candidates: list[str],
        n_samples: int,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Sample only from candidates on configured chromosomes."""
        if metadata is None:
            raise ValueError("metadata with chromosome information is required")
        valid_indices = [
            idx
            for idx, meta in enumerate(metadata)
            if str(meta.get("chromosome", "")) in self.chromosomes
        ]
        if n_samples > len(valid_indices):
            raise ValueError("not enough chromosome-matched candidates for requested sample size")
        choice = self._rng.choice(valid_indices, size=n_samples, replace=False)
        return choice.tolist()
