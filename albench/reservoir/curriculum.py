"""Curriculum-based reservoir sampling.

Generates sequences ordered by oracle prediction confidence, starting with
"easy" sequences (high-confidence predictions near the mean) and gradually
including "harder" ones (extreme or uncertain predictions).

The hypothesis: training on well-predicted sequences first builds a stable
foundation, then adding harder sequences pushes generalization.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from albench.reservoir.base import ReservoirSampler

logger = logging.getLogger(__name__)


class CurriculumSampler(ReservoirSampler):
    """Sample sequences ordered by prediction difficulty.

    Sequences are scored by how "easy" they are for the oracle to predict
    (closeness to the label mean, or low ensemble variance if available).
    The sampler returns sequences ordered from easiest to hardest.

    When used with different n_train values, smaller N gets the easiest
    sequences, larger N progressively includes harder ones.

    Parameters
    ----------
    difficulty_mode : str
        How to score difficulty:
        - ``"distance_from_mean"``: easy = close to label mean
        - ``"extreme_values"``: easy = moderate values, hard = extremes
        - ``"random_baseline"``: no curriculum (random ordering, for control)
    seed : int | None
        RNG seed.
    """

    def __init__(
        self,
        difficulty_mode: str = "distance_from_mean",
        seed: int | None = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self.difficulty_mode = difficulty_mode

    def generate(
        self,
        n_sequences: int,
        pool_sequences: list[str] | None = None,
        pool_labels: np.ndarray | None = None,
        student_model: Any = None,
        **kwargs,
    ) -> tuple[list[str], pd.DataFrame]:
        """Generate curriculum-ordered sequences from a labeled pool.

        Args:
            n_sequences: Number of sequences to return.
            pool_sequences: Full pool of candidate sequences.
            pool_labels: Oracle labels for pool_sequences.
                If None and student_model provided, uses student predictions.
            student_model: Optional model for scoring (if pool_labels unavailable).
        """
        if pool_sequences is None:
            raise ValueError("CurriculumSampler requires pool_sequences")

        n_pool = len(pool_sequences)
        if n_sequences > n_pool:
            n_sequences = n_pool

        # Get labels for difficulty scoring
        if pool_labels is not None:
            labels = np.asarray(pool_labels, dtype=np.float32)
        elif student_model is not None:
            labels = student_model.predict(pool_sequences)
        else:
            raise ValueError("Need either pool_labels or student_model for curriculum scoring")

        # Score difficulty
        if self.difficulty_mode == "distance_from_mean":
            mean_label = np.mean(labels)
            difficulty = np.abs(labels - mean_label)
        elif self.difficulty_mode == "extreme_values":
            q25 = np.percentile(labels, 25)
            q75 = np.percentile(labels, 75)
            mid = (q25 + q75) / 2
            iqr = q75 - q25
            difficulty = np.abs(labels - mid) / max(iqr, 1e-6)
        elif self.difficulty_mode == "random_baseline":
            difficulty = self._rng.random(len(labels))
        else:
            raise ValueError(f"Unknown difficulty_mode: {self.difficulty_mode}")

        # Sort by difficulty (easiest first)
        order = np.argsort(difficulty)

        # Take the easiest n_sequences
        selected_idx = order[:n_sequences]

        sequences = [pool_sequences[i] for i in selected_idx]
        meta = pd.DataFrame(
            {
                "seq_idx": selected_idx,
                "method": f"curriculum_{self.difficulty_mode}",
                "difficulty_score": difficulty[selected_idx],
                "label": labels[selected_idx],
            }
        )

        logger.info(
            f"Curriculum ({self.difficulty_mode}): selected {n_sequences:,} easiest "
            f"(difficulty range: {difficulty[selected_idx].min():.3f} - "
            f"{difficulty[selected_idx].max():.3f})"
        )

        return sequences, meta
