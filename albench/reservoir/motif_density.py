"""Motif-density-controlled random reservoir sampler."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from albench.reservoir.base import ReservoirSampler
from albench.reservoir.motif_planted import K562_MOTIFS, YEAST_MOTIFS

logger = logging.getLogger(__name__)

# Yeast flanking sequences
_YEAST_FLANK_5 = "GCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCATTTTTTTCACATCTCG"
_YEAST_FLANK_3 = "GGTTACGGCTGTT"

_NUC_BYTES = np.frombuffer(b"ACGT", dtype=np.uint8)


class MotifDensitySampler(ReservoirSampler):
    """Generate random sequences with a controlled number of planted motifs.

    Unlike :class:`MotifPlantedSampler` which samples a *range* of motif counts
    (min_motifs to max_motifs), this sampler plants exactly ``n_motifs`` motifs
    per sequence, enabling systematic study of how motif density affects model
    performance.
    """

    def __init__(
        self,
        seed: int | None = None,
        n_motifs: int = 3,
        motif_set: str = "auto",
    ) -> None:
        """Initialize sampler.

        Args:
            seed: Random seed.
            n_motifs: Exact number of motifs to plant per sequence.
            motif_set: ``"auto"`` (task-dependent), ``"k562"``, or ``"yeast"``.
        """
        self._rng = np.random.default_rng(seed)
        self.n_motifs = n_motifs
        self.motif_set = motif_set

    def _get_motifs(self, task: str) -> list[str]:
        if self.motif_set == "k562" or (self.motif_set == "auto" and task == "k562"):
            return K562_MOTIFS
        return YEAST_MOTIFS

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
        task: str = "k562",
    ) -> tuple[list[str], pd.DataFrame]:
        """Generate random sequences with exactly ``n_motifs`` planted motifs.

        Args:
            n_sequences: Number of sequences to generate.
            task: ``"k562"`` (200bp) or ``"yeast"`` (80bp random core + flanks).

        Returns:
            Tuple of (sequences, metadata_df).
        """
        seq_len = 200 if task == "k562" else 80
        motifs = self._get_motifs(task)

        sequences: list[str] = []
        planted_motifs_list: list[str] = []
        planted_positions_list: list[str] = []
        n_planted_list: list[int] = []

        for _ in range(n_sequences):
            # Generate random background
            indices = self._rng.integers(0, 4, size=seq_len, dtype=np.uint8)
            core = list(_NUC_BYTES[indices].tobytes().decode("ascii"))

            # Select exactly n_motifs motifs (with replacement from library)
            chosen_motifs = self._rng.choice(motifs, size=self.n_motifs, replace=True).tolist()

            # Plant motifs at non-overlapping positions
            occupied: set[int] = set()
            planted: list[str] = []
            positions: list[int] = []

            for motif in chosen_motifs:
                motif_len = len(motif)
                if motif_len >= seq_len:
                    continue

                # Try up to 20 times to find a non-overlapping position
                for _ in range(20):
                    pos = int(self._rng.integers(0, seq_len - motif_len))
                    positions_needed = set(range(pos, pos + motif_len))
                    if not positions_needed & occupied:
                        for j, c in enumerate(motif):
                            core[pos + j] = c
                        occupied |= positions_needed
                        planted.append(motif)
                        positions.append(pos)
                        break

            core_str = "".join(core)
            if task == "yeast":
                full_seq = _YEAST_FLANK_5 + core_str + _YEAST_FLANK_3
            else:
                full_seq = core_str

            sequences.append(full_seq)
            planted_motifs_list.append(",".join(planted) if planted else "none")
            planted_positions_list.append(
                ",".join(str(p) for p in positions) if positions else "none"
            )
            n_planted_list.append(len(planted))

        meta = pd.DataFrame(
            {
                "seq_idx": np.arange(n_sequences, dtype=np.int64),
                "method": "motif_density_controlled",
                "source": "generated",
                "n_motifs_target": self.n_motifs,
                "planted_motifs": planted_motifs_list,
                "planted_positions": planted_positions_list,
                "n_motifs_planted": np.array(n_planted_list, dtype=np.int32),
            }
        )
        logger.info(
            f"Motif-density: {n_sequences:,} sequences, target={self.n_motifs}, "
            f"mean planted={np.mean(n_planted_list):.1f}"
        )
        return sequences, meta
