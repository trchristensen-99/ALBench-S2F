"""TF motif shuffle reservoir sampler."""

from __future__ import annotations

from typing import Any

import numpy as np

from albench.reservoir.base import ReservoirSampler


class TFMotifShuffleSampler(ReservoirSampler):
    """Prefer candidates with stronger TF motif support.

    Expects one of:
    - metadata[motif_score_key], or
    - raw motif counts from sequence content for configured motifs.
    """

    def __init__(
        self,
        motifs: list[str] | None = None,
        motif_score_key: str = "motif_score",
        seed: int | None = None,
    ) -> None:
        """Initialize sampler.

        Args:
            motifs: Motifs to count directly from sequence when metadata score
                is unavailable.
            motif_score_key: Metadata key for precomputed motif score.
            seed: Random seed.
        """
        self.motifs = [m.upper() for m in (motifs or ["TATA", "CACGTG", "GGGCGG"]) if m]
        self.motif_score_key = motif_score_key
        self._rng = np.random.default_rng(seed)

    def _sequence_motif_score(self, sequence: str) -> float:
        """Compute motif count score from sequence string."""
        s = sequence.upper()
        return float(sum(s.count(motif) for motif in self.motifs))

    def sample(
        self,
        candidates: list[str],
        n_samples: int,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Return candidate indices favoring motif-rich sequences."""
        if n_samples > len(candidates):
            raise ValueError("n_samples cannot exceed number of candidates")

        if metadata is not None and len(metadata) != len(candidates):
            raise ValueError("metadata length must match number of candidates")

        scores = np.zeros(len(candidates), dtype=np.float32)
        for i, seq in enumerate(candidates):
            if metadata is not None and self.motif_score_key in metadata[i]:
                scores[i] = float(metadata[i][self.motif_score_key])
            else:
                scores[i] = self._sequence_motif_score(seq)

        # Add tiny jitter to break ties reproducibly under fixed seed.
        jitter = self._rng.uniform(0.0, 1e-8, size=len(candidates)).astype(np.float32)
        ranking = np.argsort(scores + jitter)
        return ranking[-n_samples:].tolist()
