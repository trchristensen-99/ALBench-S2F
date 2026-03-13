"""Genomic fixed-pool reservoir sampler."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from albench.reservoir.base import ReservoirSampler

logger = logging.getLogger(__name__)


class GenomicSampler(ReservoirSampler):
    """Sample from a fixed pool of genomic sequences.

    For K562: sample from HashFrag train+pool (~320K sequences).
    For yeast: sample from the full ~6M training set.
    """

    def __init__(
        self,
        seed: int | None = None,
        replace: bool = False,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self.replace = replace

    def sample(
        self,
        candidates: list[str],
        n_samples: int,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Sample indices from candidates (backward-compatible AL loop interface)."""
        n_cand = len(candidates)
        if n_samples > n_cand and not self.replace:
            raise ValueError(f"n_samples ({n_samples}) exceeds pool ({n_cand}) and replace=False")
        return self._rng.choice(n_cand, size=n_samples, replace=self.replace).tolist()

    def generate(
        self,
        n_sequences: int,
        pool_sequences: list[str] | np.ndarray,
        pool_labels: np.ndarray | None = None,
    ) -> tuple[list[str], pd.DataFrame]:
        """Sample sequences from an existing genomic pool.

        Args:
            n_sequences: Number of sequences to draw.
            pool_sequences: The full pool to sample from.
            pool_labels: Original labels (for metadata only — oracle will re-label).

        Returns:
            Tuple of (sequences, metadata_df).
        """
        n_pool = len(pool_sequences)
        replace = self.replace or n_sequences > n_pool

        if replace and n_sequences > n_pool:
            logger.warning(
                f"Requested {n_sequences:,} sequences but pool has {n_pool:,}. "
                f"Sampling with replacement."
            )

        indices = self._rng.choice(n_pool, size=n_sequences, replace=replace)
        sequences = [str(pool_sequences[i]) for i in indices]

        meta_dict: dict[str, Any] = {
            "seq_idx": np.arange(n_sequences),
            "method": "genomic_pool",
            "source": "pool",
            "pool_idx": indices,
        }
        if pool_labels is not None:
            meta_dict["original_label"] = pool_labels[indices]

        return sequences, pd.DataFrame(meta_dict)
