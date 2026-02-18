"""Reservoir sampling interfaces for candidate generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ReservoirSampler(ABC):
    """Base interface for reservoir sampling strategies."""

    @abstractmethod
    def sample(
        self,
        candidates: list[str],
        n_samples: int,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Return selected candidate indices."""
