"""Activity-stratified genomic reservoir sampler."""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np
import pandas as pd

from albench.reservoir.base import ReservoirSampler

logger = logging.getLogger(__name__)


class Predictor(Protocol):
    """Minimal interface for student model predictions."""

    def predict(self, sequences: list[str]) -> np.ndarray: ...


class ActivityStratifiedSampler(ReservoirSampler):
    """Sample genomic sequences ensuring uniform coverage across the expression range.

    Divides predicted (or known) activity scores into equal-width bins and samples
    uniformly from each bin. This tests the hypothesis that expression range
    coverage drives student model performance.
    """

    def __init__(
        self,
        seed: int | None = None,
        n_bins: int = 10,
    ) -> None:
        """Initialize sampler.

        Args:
            seed: Random seed.
            n_bins: Number of equal-width bins to partition the score range into.
        """
        self._rng = np.random.default_rng(seed)
        self.n_bins = n_bins

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

    def generate(
        self,
        n_sequences: int,
        pool_sequences: list[str] | np.ndarray,
        pool_labels: np.ndarray | None = None,
        student_model: Predictor | None = None,
    ) -> tuple[list[str], pd.DataFrame]:
        """Sample sequences with uniform coverage across activity bins.

        Args:
            n_sequences: Number of sequences to draw.
            pool_sequences: The full pool to sample from.
            pool_labels: Known activity labels for pool sequences.
            student_model: Student model with ``predict()`` method for scoring.

        Returns:
            Tuple of (sequences, metadata_df).

        Raises:
            ValueError: If neither ``pool_labels`` nor ``student_model`` is provided.
        """
        # Obtain scores
        if student_model is not None:
            seq_list = (
                list(pool_sequences) if not isinstance(pool_sequences, list) else pool_sequences
            )
            scores = np.asarray(student_model.predict(seq_list)).ravel()
            score_source = "student_model"
        elif pool_labels is not None:
            scores = np.asarray(pool_labels).ravel()
            score_source = "pool_labels"
        else:
            raise ValueError(
                "ActivityStratifiedSampler requires either pool_labels or student_model."
            )

        n_pool = len(pool_sequences)
        if len(scores) != n_pool:
            raise ValueError(f"Score length ({len(scores)}) does not match pool size ({n_pool}).")

        # Build equal-width bins
        score_min, score_max = float(np.min(scores)), float(np.max(scores))
        bin_edges = np.linspace(score_min, score_max, self.n_bins + 1)
        # Assign each pool sequence to a bin (clip last bin to include max)
        bin_assignments = np.digitize(scores, bin_edges[1:-1])  # 0 .. n_bins-1

        # Sample uniformly across bins
        per_bin = n_sequences // self.n_bins
        remainder = n_sequences % self.n_bins

        selected_indices: list[int] = []
        selected_bins: list[int] = []

        for b in range(self.n_bins):
            bin_mask = np.where(bin_assignments == b)[0]
            n_draw = per_bin + (1 if b < remainder else 0)
            if n_draw == 0:
                continue
            if len(bin_mask) == 0:
                logger.warning(f"Bin {b} is empty; skipping {n_draw} samples.")
                continue
            replace = len(bin_mask) < n_draw
            drawn = self._rng.choice(bin_mask, size=n_draw, replace=replace)
            selected_indices.extend(drawn.tolist())
            selected_bins.extend([b] * n_draw)

        indices = np.array(selected_indices)
        sequences = [str(pool_sequences[i]) for i in indices]

        meta = pd.DataFrame(
            {
                "seq_idx": np.arange(len(sequences), dtype=np.int64),
                "method": "activity_stratified",
                "source": "pool",
                "pool_idx": indices,
                "activity_bin": np.array(selected_bins, dtype=np.int32),
                "score": scores[indices],
                "score_source": score_source,
            }
        )
        logger.info(
            f"Activity-stratified: {len(sequences):,} sequences from {self.n_bins} bins "
            f"(score range [{score_min:.3f}, {score_max:.3f}], source={score_source})"
        )
        return sequences, meta
