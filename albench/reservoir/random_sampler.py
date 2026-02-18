"""Random reservoir sampler."""

from __future__ import annotations

import numpy as np

from albench.reservoir.base import ReservoirSampler


class RandomSampler(ReservoirSampler):
    """Uniform random sampler over the provided candidate set."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialize random sampler.

        Args:
            seed: Random seed for reproducibility.
        """
        self._rng = np.random.default_rng(seed)

    def sample(
        self,
        candidates: list[str],
        n_samples: int,
        metadata: list[dict[str, object]] | None = None,
    ) -> list[int]:
        """Sample indices uniformly without replacement."""
        if n_samples > len(candidates):
            raise ValueError("n_samples cannot exceed number of candidates")
        return self._rng.choice(len(candidates), size=n_samples, replace=False).tolist()
