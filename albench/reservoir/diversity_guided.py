"""Diversity-guided reservoir sampling.

Selects a maximally diverse subset of sequences from a genomic-derived
candidate pool via farthest-first traversal (k-center greedy) over a cheap
k-mer frequency embedding — no model or oracle scoring required.

This is the diversity counterpart to ``uncertainty_guided``: instead of
picking the sequences the oracle is most unsure about, it picks a subset that
spreads out as far as possible in composition space, so the student sees a
broad slice of the sequence manifold. The selection is deterministic given a
seed (only the first center is random; every subsequent pick is argmax).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from albench.reservoir.base import ReservoirSampler

logger = logging.getLogger(__name__)

# Pre-computed lookup: nucleotide byte (ASCII) → base index 0-3; 255 = non-ACGT.
_BASE_LUT = np.full(256, 255, dtype=np.uint8)
for _i, _b in enumerate(b"ACGT"):
    _BASE_LUT[_b] = _i


class DiversityGuidedSampler(ReservoirSampler):
    """Select a compositionally diverse subset via farthest-first traversal.

    Embeds each candidate as an L2-normalized k-mer frequency vector, then
    runs k-center greedy: seed one center, and repeatedly add the pool
    sequence farthest (in Euclidean distance) from the set of already-chosen
    centers. This maximizes coverage of composition space rather than density.

    Parameters
    ----------
    k : int
        k-mer length for the frequency embedding (default 4 → 256-dim vector).
        6 gives a finer 4096-dim embedding at higher cost.
    seed : int | None
        RNG seed — only used to pick the initial center (rest is deterministic).
    """

    def __init__(
        self,
        k: int = 4,
        seed: int | None = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self.k = k

    def sample(
        self,
        candidates: list[str],
        n_samples: int,
        metadata: list[dict[str, object]] | None = None,
    ) -> list[int]:
        """Return farthest-first-selected candidate indices (AL-loop interface)."""
        _, meta = self.generate(n_samples, pool_sequences=candidates)
        return meta["seq_idx"].tolist()

    def _kmer_embedding(self, sequences: list[str]) -> np.ndarray:
        """L2-normalized k-mer frequency vectors, shape (n_seqs, 4**k).

        Non-ACGT characters break any k-mer window that contains them (that
        window is skipped), so ambiguous bases never contribute a phantom count.
        """
        k = self.k
        dim = 4**k
        # Base-4 positional weights: kmer index = sum(base_i * 4**(k-1-i)).
        weights = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)

        embedding = np.zeros((len(sequences), dim), dtype=np.float32)
        for row, seq in enumerate(sequences):
            codes = _BASE_LUT[np.frombuffer(seq.upper().encode("ascii"), dtype=np.uint8)]
            if len(codes) < k:
                continue
            # Sliding windows of length k → (n_windows, k) base codes.
            windows = np.lib.stride_tricks.sliding_window_view(codes, k)
            valid = ~(windows == 255).any(axis=1)
            if not valid.any():
                continue
            kmer_idx = windows[valid].astype(np.int64) @ weights
            counts = np.bincount(kmer_idx, minlength=dim).astype(np.float32)
            norm = np.linalg.norm(counts)
            if norm > 0:
                embedding[row] = counts / norm
        return embedding

    def generate(
        self,
        n_sequences: int,
        pool_sequences: list[str] | None = None,
        pool_labels: np.ndarray | None = None,
        **kwargs,
    ) -> tuple[list[str], pd.DataFrame]:
        """Select ``n_sequences`` diverse candidates from ``pool_sequences``.

        Args:
            n_sequences: Number of sequences to return.
            pool_sequences: Candidate sequences to select from.
            pool_labels: Ignored (oracle re-labels downstream) — kept for a
                signature consistent with the other pool-based samplers.
        """
        if pool_sequences is None:
            raise ValueError("DiversityGuidedSampler requires pool_sequences")

        n_pool = len(pool_sequences)
        if n_sequences >= n_pool:
            # Pool too small: take everything (order preserved), sample with
            # replacement only for the overflow — matches genomic sampler policy.
            if n_sequences > n_pool:
                logger.warning(
                    f"Requested {n_sequences:,} but pool has {n_pool:,}. "
                    f"Returning all + sampling remainder with replacement."
                )
                extra = self._rng.choice(n_pool, size=n_sequences - n_pool, replace=True)
                selected_idx = np.concatenate([np.arange(n_pool), extra])
            else:
                selected_idx = np.arange(n_pool)
            sequences = [str(pool_sequences[i]) for i in selected_idx]
            meta = pd.DataFrame(
                {
                    "seq_idx": selected_idx.astype(int),
                    "method": "diversity_guided",
                    "min_dist": np.zeros(len(selected_idx), dtype=np.float32),
                }
            )
            return sequences, meta

        embedding = self._kmer_embedding(list(pool_sequences))

        # Farthest-first traversal (k-center greedy). `min_dist[i]` tracks the
        # distance from pool sequence i to the nearest already-chosen center;
        # each step adds the argmax and refreshes distances against it.
        selected_idx = np.empty(n_sequences, dtype=np.int64)
        first = int(self._rng.integers(n_pool))
        selected_idx[0] = first
        min_dist = np.linalg.norm(embedding - embedding[first], axis=1)
        min_dist[first] = -1.0  # never re-pick a chosen center

        for step in range(1, n_sequences):
            nxt = int(np.argmax(min_dist))
            selected_idx[step] = nxt
            new_dist = np.linalg.norm(embedding - embedding[nxt], axis=1)
            min_dist = np.minimum(min_dist, new_dist)
            min_dist[nxt] = -1.0

        chosen_min_dist = np.linalg.norm(embedding - embedding[selected_idx[0]], axis=1)[
            selected_idx
        ]
        sequences = [str(pool_sequences[i]) for i in selected_idx]
        meta = pd.DataFrame(
            {
                "seq_idx": selected_idx.astype(int),
                "method": "diversity_guided",
                "min_dist": chosen_min_dist.astype(np.float32),
            }
        )

        logger.info(
            f"DiversityGuided (k={self.k}): selected {n_sequences:,} of "
            f"{n_pool:,} via farthest-first (embedding dim={4**self.k})"
        )

        return sequences, meta
