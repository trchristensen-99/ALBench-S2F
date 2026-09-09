"""GC-content matched random reservoir sampler."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from albench.reservoir.base import ReservoirSampler

logger = logging.getLogger(__name__)

# Yeast flanking sequences
_YEAST_FLANK_5 = "GCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCATTTTTTTCACATCTCG"
_YEAST_FLANK_3 = "GGTTACGGCTGTT"

_NUC_BYTES = np.frombuffer(b"ACGT", dtype=np.uint8)


def _gc_content(seq: str) -> float:
    """Compute GC fraction of a DNA string."""
    s = seq.upper()
    gc = s.count("G") + s.count("C")
    total = s.count("A") + s.count("C") + s.count("G") + s.count("T")
    return gc / max(total, 1)


class GCMatchedSampler(ReservoirSampler):
    """Generate random sequences whose GC content matches a reference distribution.

    Estimates the GC distribution from pool sequences as a histogram with
    ``n_gc_bins`` bins, draws target GC fractions from that histogram, then
    generates random sequences by sampling per-position nucleotides with base
    probabilities calibrated to hit each target.

    ``n_gc_bins`` controls how faithfully the pool's GC distribution is
    reproduced: a coarse histogram smooths it (each bin becomes uniform), a fine
    one tracks its shape including any multi-modality. It is a real knob, so it
    is worth sweeping alongside the other reservoir parameters.
    """

    def __init__(
        self,
        seed: int | None = None,
        n_gc_bins: int = 50,
    ) -> None:
        if n_gc_bins < 1:
            raise ValueError(f"n_gc_bins must be >= 1, got {n_gc_bins}")
        self._rng = np.random.default_rng(seed)
        self.n_gc_bins = n_gc_bins

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
        task: str = "k562",
        batch_size: int = 50_000,
    ) -> tuple[list[str], pd.DataFrame]:
        """Generate random sequences with GC content matching the pool distribution.

        Args:
            n_sequences: Number of sequences to generate.
            pool_sequences: Reference pool to estimate GC distribution from.
            task: ``"k562"`` (200bp) or ``"yeast"`` (80bp random + flanks).
            batch_size: Generate in batches to limit memory.

        Returns:
            Tuple of (sequences, metadata_df).
        """
        seq_len = 200 if task == "k562" else 80

        # Estimate GC distribution from pool
        pool_gc = np.array([_gc_content(str(s)) for s in pool_sequences[:50_000]])
        logger.info(
            f"Pool GC: mean={pool_gc.mean():.3f}, std={pool_gc.std():.3f}, "
            f"range=[{pool_gc.min():.3f}, {pool_gc.max():.3f}]"
        )

        # Sample target GC fractions from an `n_gc_bins`-bin histogram of the pool:
        # pick a bin with probability proportional to its pool count, then draw
        # uniformly within that bin. The bin width sets the smoothing, so
        # `n_gc_bins` is what controls fidelity to the pool distribution.
        edges = np.histogram_bin_edges(pool_gc, bins=self.n_gc_bins, range=(0.0, 1.0))
        counts, _ = np.histogram(pool_gc, bins=edges)
        if counts.sum() == 0:  # degenerate pool
            raise ValueError("Could not estimate a GC histogram from pool_sequences.")
        probs = counts / counts.sum()
        chosen_bin = self._rng.choice(len(counts), size=n_sequences, replace=True, p=probs)
        lo = edges[chosen_bin]
        hi = edges[chosen_bin + 1]
        target_gc = lo + self._rng.random(n_sequences) * (hi - lo)
        target_gc = np.clip(target_gc, 0.05, 0.95)
        logger.info(
            f"GC histogram: {self.n_gc_bins} bins, {int((counts > 0).sum())} occupied, "
            f"bin width={edges[1] - edges[0]:.4f}"
        )

        sequences: list[str] = []
        actual_gc_arrays: list[np.ndarray] = []

        for start in range(0, n_sequences, batch_size):
            n_batch = min(batch_size, n_sequences - start)
            batch_gc = target_gc[start : start + n_batch]

            # Per-position base probs are position-independent within a sequence but
            # differ between sequences, so draw by inverse-CDF on a uniform matrix
            # rather than looping rng.choice per sequence (which dominated runtime at
            # the 1M+ scale these reservoirs are generated at).
            # [A, C, G, T] = [(1-gc)/2, gc/2, gc/2, (1-gc)/2]
            p_at = (1.0 - batch_gc) / 2.0
            p_gc = batch_gc / 2.0
            # cumulative boundaries after A, after C, after G  -> shape (n_batch, 3)
            cum = np.stack([p_at, p_at + p_gc, p_at + 2.0 * p_gc], axis=1)
            u = self._rng.random((n_batch, seq_len))
            idx = (u[:, :, None] >= cum[:, None, :]).sum(axis=2).astype(np.uint8)

            cores = _NUC_BYTES[idx]  # (n_batch, seq_len) uint8 ASCII
            actual_gc_arrays.append(
                ((cores == ord("G")) | (cores == ord("C"))).sum(axis=1) / seq_len
            )

            core_bytes = cores.tobytes()
            for i in range(n_batch):
                core = core_bytes[i * seq_len : (i + 1) * seq_len].decode("ascii")
                if task == "yeast":
                    sequences.append(_YEAST_FLANK_5 + core + _YEAST_FLANK_3)
                else:
                    sequences.append(core)

        actual_gc = np.concatenate(actual_gc_arrays) if actual_gc_arrays else np.zeros(0)

        meta = pd.DataFrame(
            {
                "seq_idx": np.arange(n_sequences, dtype=np.int64),
                "method": "gc_matched_random",
                "source": "generated",
                "target_gc": target_gc,
                "actual_gc": actual_gc.astype(np.float32),
            }
        )
        logger.info(
            f"GC-matched: generated {n_sequences:,} sequences, mean GC={np.mean(actual_gc):.3f}"
        )
        return sequences, meta
