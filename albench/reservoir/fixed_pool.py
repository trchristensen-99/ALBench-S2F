"""Generic fixed-pool reservoir sampler with optional metadata filters."""

from __future__ import annotations

from typing import Any

import numpy as np

from albench.reservoir.base import ReservoirSampler


class FixedPoolSampler(ReservoirSampler):
    """Randomly subsample from a fixed pool, optionally filtered by metadata."""

    def __init__(
        self,
        seed: int | None = None,
        metadata_filters: dict[str, list[str] | list[int] | list[float]] | None = None,
    ) -> None:
        """Initialize the sampler.

        Args:
            seed: Random seed for reproducibility.
            metadata_filters: Optional allow-list filters keyed by metadata field.
                Example: ``{"chromosome": ["chr1", "chr2"]}``.
        """
        self._rng = np.random.default_rng(seed)
        self.metadata_filters = metadata_filters or {}

    def _allowed(self, meta: dict[str, Any]) -> bool:
        """Return True if metadata row passes all configured filters."""
        for key, allowed_values in self.metadata_filters.items():
            if key not in meta:
                return False
            if meta[key] not in set(allowed_values):
                return False
        return True

    def sample(
        self,
        candidates: list[str],
        n_samples: int,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Return subsampled indices from the candidate pool."""
        if n_samples > len(candidates):
            raise ValueError("n_samples cannot exceed number of candidates")

        if not self.metadata_filters:
            return self._rng.choice(len(candidates), size=n_samples, replace=False).tolist()

        if metadata is None:
            raise ValueError("metadata is required when metadata_filters are configured")
        if len(metadata) != len(candidates):
            raise ValueError("metadata length must match number of candidates")

        valid_indices = [i for i, meta in enumerate(metadata) if self._allowed(meta)]
        if n_samples > len(valid_indices):
            raise ValueError("not enough filter-matching candidates for requested sample size")
        return self._rng.choice(valid_indices, size=n_samples, replace=False).tolist()
