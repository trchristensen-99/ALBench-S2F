"""Partial mutagenesis reservoir sampler."""

from __future__ import annotations

from typing import Any

import numpy as np

from albench.reservoir.base import ReservoirSampler


class PartialMutagenesisSampler(ReservoirSampler):
    """Sample candidates consistent with partial mutagenesis constraints.

    This implementation expects mutation metadata when available and otherwise
    falls back to random subsampling.
    """

    def __init__(
        self,
        seed: int | None = None,
        min_mutation_fraction: float = 0.01,
        max_mutation_fraction: float = 0.25,
        mutation_fraction_key: str = "mutation_fraction",
    ) -> None:
        """Initialize sampler.

        Args:
            seed: Random seed for reproducibility.
            min_mutation_fraction: Minimum accepted mutation fraction.
            max_mutation_fraction: Maximum accepted mutation fraction.
            mutation_fraction_key: Metadata key storing mutation fraction.
        """
        if min_mutation_fraction < 0.0 or max_mutation_fraction > 1.0:
            raise ValueError("mutation fractions must be in [0, 1]")
        if min_mutation_fraction > max_mutation_fraction:
            raise ValueError("min_mutation_fraction must be <= max_mutation_fraction")
        self._rng = np.random.default_rng(seed)
        self.min_mutation_fraction = min_mutation_fraction
        self.max_mutation_fraction = max_mutation_fraction
        self.mutation_fraction_key = mutation_fraction_key

    def sample(
        self,
        candidates: list[str],
        n_samples: int,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Return indices, preferring mutation-constrained candidates when possible."""
        if n_samples > len(candidates):
            raise ValueError("n_samples cannot exceed number of candidates")

        if metadata is None:
            return self._rng.choice(len(candidates), size=n_samples, replace=False).tolist()
        if len(metadata) != len(candidates):
            raise ValueError("metadata length must match number of candidates")

        valid: list[int] = []
        for idx, meta in enumerate(metadata):
            if self.mutation_fraction_key not in meta:
                continue
            frac = float(meta[self.mutation_fraction_key])
            if self.min_mutation_fraction <= frac <= self.max_mutation_fraction:
                valid.append(idx)

        source = valid if len(valid) >= n_samples else list(range(len(candidates)))
        return self._rng.choice(source, size=n_samples, replace=False).tolist()
