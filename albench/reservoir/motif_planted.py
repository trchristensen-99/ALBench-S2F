"""Motif-planted random reservoir sampler."""

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

# Known TF binding motifs relevant to K562 and yeast expression
K562_MOTIFS = [
    "TATAAA",  # TATA box
    "CACGTG",  # E-box (MYC, MAX)
    "GGGCGG",  # SP1
    "CCAAT",  # NF-Y / CCAAT-box
    "GATA",  # GATA factors (important in K562 erythroid)
    "AGATAA",  # GATA1 consensus
    "CTCF",  # CTCF (partial)
    "TGACGTCA",  # AP-1 / CRE
    "TGAGTCA",  # AP-1
]

YEAST_MOTIFS = [
    "TATAAA",  # TATA box
    "GCGATGAG",  # UAS (upstream activating sequence)
    "CACGTG",  # E-box
    "CCAAT",  # CCAAT-box
    "ACCCG",  # GCN4
    "TGACTC",  # AP-1-like
]


class MotifPlantedSampler(ReservoirSampler):
    """Generate random sequences with known TF motifs planted at random positions.

    Creates random background sequences and inserts 1-3 known TF binding motifs
    at random, non-overlapping positions. This tests whether models can learn
    from sequences that contain realistic regulatory grammar in random contexts.
    """

    def __init__(
        self,
        seed: int | None = None,
        min_motifs: int = 1,
        max_motifs: int = 3,
        motif_set: str = "auto",
    ) -> None:
        """Initialize sampler.

        Args:
            seed: Random seed.
            min_motifs: Minimum motifs to plant per sequence.
            max_motifs: Maximum motifs to plant per sequence.
            motif_set: ``"auto"`` (task-dependent), ``"k562"``, or ``"yeast"``.
        """
        self._rng = np.random.default_rng(seed)
        self.min_motifs = min_motifs
        self.max_motifs = max_motifs
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
        """Generate random sequences with planted TF motifs.

        Args:
            n_sequences: Number of sequences to generate.
            task: ``"k562"`` (200bp) or ``"yeast"`` (80bp random + flanks).

        Returns:
            Tuple of (sequences, metadata_df).
        """
        seq_len = 200 if task == "k562" else 80
        motifs = self._get_motifs(task)

        sequences: list[str] = []
        planted_motifs_list: list[str] = []
        n_planted_list: list[int] = []

        for _ in range(n_sequences):
            # Generate random background
            indices = self._rng.integers(0, 4, size=seq_len, dtype=np.uint8)
            core = list(_NUC_BYTES[indices].tobytes().decode("ascii"))

            # Choose how many motifs to plant
            n_plant = self._rng.integers(self.min_motifs, self.max_motifs + 1)
            chosen_motifs = self._rng.choice(motifs, size=n_plant, replace=True).tolist()

            # Plant motifs at non-overlapping positions
            occupied = set()
            planted = []
            for motif in chosen_motifs:
                motif_len = len(motif)
                if motif_len >= seq_len:
                    continue

                # Try up to 20 times to find a non-overlapping position
                for _ in range(20):
                    pos = self._rng.integers(0, seq_len - motif_len)
                    positions_needed = set(range(pos, pos + motif_len))
                    if not positions_needed & occupied:
                        for j, c in enumerate(motif):
                            core[pos + j] = c
                        occupied |= positions_needed
                        planted.append(motif)
                        break

            core_str = "".join(core)
            if task == "yeast":
                full_seq = _YEAST_FLANK_5 + core_str + _YEAST_FLANK_3
            else:
                full_seq = core_str

            sequences.append(full_seq)
            planted_motifs_list.append(",".join(planted) if planted else "none")
            n_planted_list.append(len(planted))

        meta = pd.DataFrame(
            {
                "seq_idx": np.arange(n_sequences, dtype=np.int64),
                "method": "motif_planted_random",
                "source": "generated",
                "planted_motifs": planted_motifs_list,
                "n_motifs_planted": np.array(n_planted_list, dtype=np.int32),
            }
        )
        logger.info(
            f"Motif-planted: {n_sequences:,} sequences, mean motifs={np.mean(n_planted_list):.1f}"
        )
        return sequences, meta
