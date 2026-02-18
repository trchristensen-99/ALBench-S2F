"""TF motif shuffle reservoir sampler stub."""

from __future__ import annotations

from typing import Any

from albench.reservoir.base import ReservoirSampler


class TFMotifShuffleSampler(ReservoirSampler):
    """Placeholder for TF-motif shuffling candidate generation."""

    def sample(
        self,
        candidates: list[str],
        n_samples: int,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Sample indices (not yet implemented)."""
        raise NotImplementedError("TFMotifShuffleSampler is not implemented yet")
