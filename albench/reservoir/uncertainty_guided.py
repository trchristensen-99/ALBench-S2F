"""Oracle-uncertainty-guided reservoir sampling.

Generates sequences from a base pool, scores them by oracle ensemble
disagreement (variance across ensemble members), and preferentially
selects sequences where the oracle is most uncertain.

This is conceptually similar to Exp 1.2 acquisition functions, but
applied at the reservoir level — the "acquisition" happens before
any student is trained.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from albench.reservoir.base import ReservoirSampler

logger = logging.getLogger(__name__)


class UncertaintyGuidedSampler(ReservoirSampler):
    """Sample sequences where the oracle is most uncertain.

    Uses oracle ensemble variance (if available) or prediction magnitude
    as a proxy for difficulty/informativeness.

    Parameters
    ----------
    strategy : str
        How to use uncertainty:
        - ``"most_uncertain"``: select highest-variance sequences
        - ``"balanced"``: mix of uncertain + confident sequences
        - ``"anti_curriculum"``: reverse curriculum (hardest first)
    uncertainty_fraction : float
        For ``"balanced"``, fraction of sequences from high-uncertainty pool.
    seed : int | None
        RNG seed.
    """

    def __init__(
        self,
        strategy: str = "most_uncertain",
        uncertainty_fraction: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self.strategy = strategy
        self.uncertainty_fraction = uncertainty_fraction

    def generate(
        self,
        n_sequences: int,
        pool_sequences: list[str] | None = None,
        pool_labels: np.ndarray | None = None,
        oracle_std: np.ndarray | None = None,
        student_model: Any = None,
        **kwargs,
    ) -> tuple[list[str], pd.DataFrame]:
        """Generate uncertainty-guided sequences.

        Args:
            n_sequences: Number of sequences to return.
            pool_sequences: Candidate sequences.
            pool_labels: Oracle mean predictions.
            oracle_std: Oracle ensemble standard deviation (best signal).
            student_model: Optional — if oracle_std unavailable, uses student
                uncertainty as proxy.
        """
        if pool_sequences is None:
            raise ValueError("UncertaintyGuidedSampler requires pool_sequences")

        n_pool = len(pool_sequences)
        if n_sequences > n_pool:
            n_sequences = n_pool

        # Get uncertainty scores
        if oracle_std is not None:
            uncertainty = np.asarray(oracle_std, dtype=np.float32)
        elif student_model is not None and hasattr(student_model, "uncertainty"):
            uncertainty = student_model.uncertainty(pool_sequences)
        elif pool_labels is not None:
            # Proxy: distance from median (extreme predictions are harder)
            labels = np.asarray(pool_labels, dtype=np.float32)
            median = np.median(labels)
            uncertainty = np.abs(labels - median)
        else:
            raise ValueError("Need oracle_std, student_model, or pool_labels")

        if self.strategy == "most_uncertain":
            order = np.argsort(-uncertainty)  # highest uncertainty first
            selected_idx = order[:n_sequences]

        elif self.strategy == "balanced":
            # Mix: top uncertain + random confident
            n_uncertain = int(n_sequences * self.uncertainty_fraction)
            n_confident = n_sequences - n_uncertain

            order = np.argsort(-uncertainty)
            uncertain_idx = order[:n_uncertain]

            remaining = order[n_uncertain:]
            confident_idx = self._rng.choice(
                remaining, size=min(n_confident, len(remaining)), replace=False
            )
            selected_idx = np.concatenate([uncertain_idx, confident_idx])

        elif self.strategy == "anti_curriculum":
            order = np.argsort(-uncertainty)
            selected_idx = order[:n_sequences]

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        sequences = [pool_sequences[i] for i in selected_idx]
        meta = pd.DataFrame(
            {
                "seq_idx": selected_idx.astype(int),
                "method": f"uncertainty_guided_{self.strategy}",
                "uncertainty_score": uncertainty[selected_idx],
            }
        )

        logger.info(
            f"UncertaintyGuided ({self.strategy}): selected {n_sequences:,} "
            f"(uncertainty range: {uncertainty[selected_idx].min():.3f} - "
            f"{uncertainty[selected_idx].max():.3f})"
        )

        return sequences, meta
