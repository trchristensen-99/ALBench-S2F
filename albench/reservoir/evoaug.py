"""EvoAug reservoir sampler."""

from __future__ import annotations

from typing import Any

import numpy as np

from albench.reservoir.base import ReservoirSampler


class EvoAugSampler(ReservoirSampler):
    """Prefer candidates with stronger augmentation utility scores.

    Expected metadata key:
    - ``evoaug_score`` (higher is better)
    """

    def __init__(
        self,
        seed: int | None = None,
        score_key: str = "evoaug_score",
        fallback_random: bool = True,
    ) -> None:
        """Initialize sampler.

        Args:
            seed: Random seed.
            score_key: Metadata key for augmentation utility score.
            fallback_random: Whether to fall back to random sampling when metadata
                is missing.
        """
        self._rng = np.random.default_rng(seed)
        self.score_key = score_key
        self.fallback_random = fallback_random

    def sample(
        self,
        candidates: list[str],
        n_samples: int,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Select indices by EvoAug score, with optional random fallback."""
        if n_samples > len(candidates):
            raise ValueError("n_samples cannot exceed number of candidates")

        if metadata is None:
            if not self.fallback_random:
                raise ValueError("metadata is required when fallback_random is False")
            return self._rng.choice(len(candidates), size=n_samples, replace=False).tolist()
        if len(metadata) != len(candidates):
            raise ValueError("metadata length must match number of candidates")

        if not any(self.score_key in row for row in metadata):
            if not self.fallback_random:
                raise ValueError(f"metadata must include key '{self.score_key}'")
            return self._rng.choice(len(candidates), size=n_samples, replace=False).tolist()

        scores = np.asarray(
            [float(row.get(self.score_key, 0.0)) for row in metadata], dtype=np.float32
        )
        jitter = self._rng.uniform(0.0, 1e-8, size=len(candidates)).astype(np.float32)
        ranking = np.argsort(scores + jitter)
        return ranking[-n_samples:].tolist()
