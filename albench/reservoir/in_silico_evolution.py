"""In-silico evolution reservoir sampler."""

from __future__ import annotations

from typing import Any

import numpy as np

from albench.reservoir.base import ReservoirSampler


class InSilicoEvolutionSampler(ReservoirSampler):
    """Prefer candidates with better in-silico evolutionary fitness.

    Expected metadata keys (in priority order):
    - ``evolution_score``
    - ``fitness_delta``
    """

    def __init__(
        self,
        seed: int | None = None,
        primary_score_key: str = "evolution_score",
        secondary_score_key: str = "fitness_delta",
        fallback_random: bool = True,
    ) -> None:
        """Initialize sampler.

        Args:
            seed: Random seed.
            primary_score_key: Primary metadata key for ranking.
            secondary_score_key: Secondary key when primary is missing.
            fallback_random: Whether to fall back to random sampling when metadata
                is missing.
        """
        self._rng = np.random.default_rng(seed)
        self.primary_score_key = primary_score_key
        self.secondary_score_key = secondary_score_key
        self.fallback_random = fallback_random

    def sample(
        self,
        candidates: list[str],
        n_samples: int,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Select indices by evolutionary score with robust fallback."""
        if n_samples > len(candidates):
            raise ValueError("n_samples cannot exceed number of candidates")

        if metadata is None:
            if not self.fallback_random:
                raise ValueError("metadata is required when fallback_random is False")
            return self._rng.choice(len(candidates), size=n_samples, replace=False).tolist()
        if len(metadata) != len(candidates):
            raise ValueError("metadata length must match number of candidates")

        scores = np.asarray(
            [
                float(row.get(self.primary_score_key, row.get(self.secondary_score_key, 0.0)))
                for row in metadata
            ],
            dtype=np.float32,
        )
        if np.all(scores == 0.0) and not self.fallback_random:
            raise ValueError(
                "metadata did not provide nonzero in-silico evolution scores and fallback is disabled"
            )
        if np.all(scores == 0.0) and self.fallback_random:
            return self._rng.choice(len(candidates), size=n_samples, replace=False).tolist()

        jitter = self._rng.uniform(0.0, 1e-8, size=len(candidates)).astype(np.float32)
        ranking = np.argsort(scores + jitter)
        return ranking[-n_samples:].tolist()
