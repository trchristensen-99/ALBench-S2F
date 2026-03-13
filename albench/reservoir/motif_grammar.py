"""Motif grammar reservoir sampler — systematic motif arrangement exploration.

Generates sequences by placing known TF binding motifs in random backgrounds
with controlled spacing, orientation, and positional placement. This enables
analysis of how motif grammar (arrangement rules) affects model learning.

Each generated sequence records its full configuration:
- Which motifs were placed
- Orientation of each motif (forward / reverse complement)
- Spacing between consecutive motifs (bp)
- Absolute position of the motif block within the sequence

This metadata enables post-hoc analysis of which arrangements are most
informative for model training.
"""

from __future__ import annotations

import logging
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from albench.reservoir.base import ReservoirSampler

logger = logging.getLogger(__name__)

# Yeast flanking sequences
_YEAST_FLANK_5 = "GCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCATTTTTTTCACATCTCG"
_YEAST_FLANK_3 = "GGTTACGGCTGTT"

_NUC_BYTES = np.frombuffer(b"ACGT", dtype=np.uint8)
_RC_MAP = str.maketrans("ACGT", "TGCA")

# Well-characterized TF binding motifs
K562_MOTIF_LIBRARY = {
    "TATA": "TATAAA",
    "SP1": "GGGCGG",
    "EBOX": "CACGTG",
    "CCAAT": "CCAAT",
    "GATA1": "AGATAA",
    "AP1": "TGAGTCA",
    "CRE": "TGACGTCA",
    "NFkB": "GGGACTTTCC",
    "ETS": "GGAA",
    "CTCF": "CCGCGNGGNGGCAG",  # N = any base
}

YEAST_MOTIF_LIBRARY = {
    "TATA": "TATAAA",
    "GCN4": "TGACTCA",
    "RAP1": "ACACCCATACATTT",
    "ABF1": "TCRNNNNNNACG",  # IUPAC codes
    "MCB": "ACGCGT",
    "SCB": "CACGAAA",
    "EBOX": "CACGTG",
    "CCAAT": "CCAAT",
}

# Biologically realistic spacing priors (bp between motif ends)
# Based on typical promoter architecture
SPACING_PRIORS = {
    "tight": {"min": 0, "max": 10, "mean": 5, "std": 2},
    "proximal": {"min": 5, "max": 30, "mean": 15, "std": 5},
    "distal": {"min": 20, "max": 80, "mean": 40, "std": 15},
    "uniform": {"min": 0, "max": 80},
}


def _reverse_complement(seq: str) -> str:
    return seq.upper().translate(_RC_MAP)[::-1]


def _resolve_iupac(seq: str, rng: np.random.Generator) -> str:
    """Resolve IUPAC ambiguity codes to concrete nucleotides."""
    iupac = {
        "R": "AG",
        "Y": "CT",
        "S": "GC",
        "W": "AT",
        "K": "GT",
        "M": "AC",
        "B": "CGT",
        "D": "AGT",
        "H": "ACT",
        "V": "ACG",
        "N": "ACGT",
    }
    out = []
    for c in seq.upper():
        if c in iupac:
            out.append(rng.choice(list(iupac[c])))
        else:
            out.append(c)
    return "".join(out)


class MotifGrammarSampler(ReservoirSampler):
    """Generate sequences with controlled motif arrangements.

    For each sequence:
    1. Select 1-K motifs from the library
    2. Assign orientation (fwd/rc) to each motif
    3. Sample inter-motif spacings from a distribution
    4. Place the motif block at a random position in a random background
    5. Record the full configuration for downstream analysis

    The spacing distribution can be:
    - ``"uniform"``: uniform over [0, max_spacing]
    - ``"tight"``: biased toward close spacing (5 ± 2 bp)
    - ``"proximal"``: moderate spacing (15 ± 5 bp)
    - ``"distal"``: wide spacing (40 ± 15 bp)
    - ``"all"``: sample from all spacing distributions equally
    """

    def __init__(
        self,
        seed: int | None = None,
        min_motifs: int = 1,
        max_motifs: int = 3,
        spacing_mode: str = "all",
        orientation_mode: str = "random",
        motif_set: str = "auto",
        position_mode: str = "random",
    ) -> None:
        """Initialize sampler.

        Args:
            seed: Random seed.
            min_motifs: Minimum motifs per sequence.
            max_motifs: Maximum motifs per sequence.
            spacing_mode: How to sample inter-motif spacing. One of
                ``"uniform"``, ``"tight"``, ``"proximal"``, ``"distal"``, ``"all"``.
            orientation_mode: ``"random"`` (coin flip), ``"forward_only"``,
                ``"reverse_only"``, ``"all_combos"`` (enumerate all 2^K combos).
            motif_set: ``"auto"`` (task-dependent), ``"k562"``, ``"yeast"``,
                or ``"custom"``.
            position_mode: ``"random"`` (uniform position) or ``"center"``
                (centered in sequence).
        """
        self._rng = np.random.default_rng(seed)
        self.min_motifs = min_motifs
        self.max_motifs = max_motifs
        self.spacing_mode = spacing_mode
        self.orientation_mode = orientation_mode
        self.motif_set = motif_set
        self.position_mode = position_mode

    def _get_motif_library(self, task: str) -> dict[str, str]:
        if self.motif_set == "k562" or (self.motif_set == "auto" and task == "k562"):
            return K562_MOTIF_LIBRARY
        return YEAST_MOTIF_LIBRARY

    def _sample_spacing(self) -> int:
        """Sample one inter-motif spacing (bp)."""
        if self.spacing_mode == "all":
            mode = self._rng.choice(["tight", "proximal", "distal", "uniform"])
        else:
            mode = self.spacing_mode

        prior = SPACING_PRIORS[mode]
        if mode == "uniform":
            return int(self._rng.integers(prior["min"], prior["max"] + 1))
        else:
            val = self._rng.normal(prior["mean"], prior["std"])
            return int(np.clip(val, prior["min"], prior["max"]))

    def _sample_orientations(self, n_motifs: int) -> list[str]:
        """Sample orientation for each motif."""
        if self.orientation_mode == "forward_only":
            return ["fwd"] * n_motifs
        elif self.orientation_mode == "reverse_only":
            return ["rc"] * n_motifs
        else:
            # Random coin flip per motif
            return [self._rng.choice(["fwd", "rc"]) for _ in range(n_motifs)]

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
        """Generate sequences with controlled motif arrangements.

        Args:
            n_sequences: Number of sequences to generate.
            task: ``"k562"`` (200bp) or ``"yeast"`` (80bp random + flanks).

        Returns:
            Tuple of (sequences, metadata_df) where metadata includes
            motif names, orientations, spacings, and positions.
        """
        seq_len = 200 if task == "k562" else 80
        motif_lib = self._get_motif_library(task)
        motif_names = list(motif_lib.keys())

        sequences: list[str] = []
        meta_records: list[dict[str, Any]] = []

        for idx in range(n_sequences):
            # Choose motifs
            n_motifs = int(self._rng.integers(self.min_motifs, self.max_motifs + 1))
            chosen_names = self._rng.choice(motif_names, size=n_motifs, replace=True).tolist()
            orientations = self._sample_orientations(n_motifs)

            # Resolve motif sequences
            motif_seqs = []
            for name, orient in zip(chosen_names, orientations):
                raw = _resolve_iupac(motif_lib[name], self._rng)
                if orient == "rc":
                    raw = _reverse_complement(raw)
                motif_seqs.append(raw)

            # Sample spacings between consecutive motifs
            spacings = [self._sample_spacing() for _ in range(max(0, n_motifs - 1))]

            # Build the motif block
            block_parts = [motif_seqs[0]]
            for i in range(1, n_motifs):
                # Spacer: random nucleotides
                sp_len = spacings[i - 1]
                spacer_idx = self._rng.integers(0, 4, size=sp_len, dtype=np.uint8)
                spacer = _NUC_BYTES[spacer_idx].tobytes().decode("ascii") if sp_len > 0 else ""
                block_parts.append(spacer)
                block_parts.append(motif_seqs[i])
            block = "".join(block_parts)

            # If block is too long for the sequence, truncate motifs
            if len(block) > seq_len:
                # Fall back to single motif
                block = motif_seqs[0][:seq_len]
                chosen_names = chosen_names[:1]
                orientations = orientations[:1]
                spacings = []

            # Generate random background
            bg_idx = self._rng.integers(0, 4, size=seq_len, dtype=np.uint8)
            background = list(_NUC_BYTES[bg_idx].tobytes().decode("ascii"))

            # Place block in background
            if self.position_mode == "center":
                start = max(0, (seq_len - len(block)) // 2)
            else:
                max_start = max(0, seq_len - len(block))
                start = int(self._rng.integers(0, max_start + 1)) if max_start > 0 else 0

            for j, c in enumerate(block):
                if start + j < seq_len:
                    background[start + j] = c

            core = "".join(background)
            if task == "yeast":
                full_seq = _YEAST_FLANK_5 + core + _YEAST_FLANK_3
            else:
                full_seq = core
            sequences.append(full_seq)

            meta_records.append(
                {
                    "seq_idx": idx,
                    "method": "motif_grammar",
                    "source": "generated",
                    "motif_names": ",".join(chosen_names),
                    "motif_orientations": ",".join(orientations),
                    "motif_spacings": ",".join(str(s) for s in spacings),
                    "block_position": start,
                    "block_length": len(block),
                    "n_motifs": len(chosen_names),
                    "spacing_mode": self.spacing_mode,
                }
            )

        meta = pd.DataFrame(meta_records)
        logger.info(
            f"Motif grammar: {n_sequences:,} sequences, "
            f"mean motifs={meta['n_motifs'].mean():.1f}, "
            f"spacing_mode={self.spacing_mode}"
        )
        return sequences, meta

    def generate_systematic(
        self,
        motif_pair: tuple[str, str],
        task: str = "k562",
        spacings: list[int] | None = None,
        n_replicates: int = 10,
    ) -> tuple[list[str], pd.DataFrame]:
        """Generate a systematic grid over spacing × orientation for a motif pair.

        Useful for analyzing how specific motif arrangements affect predictions.

        Args:
            motif_pair: Two motif names from the library.
            task: ``"k562"`` or ``"yeast"``.
            spacings: List of spacings to test (default: 0, 5, 10, 15, 20, 30, 50).
            n_replicates: Number of random background replicates per configuration.

        Returns:
            Tuple of (sequences, metadata_df) with one entry per
            (spacing, orientation_combo, replicate).
        """
        if spacings is None:
            spacings = [0, 5, 10, 15, 20, 30, 50]

        seq_len = 200 if task == "k562" else 80
        motif_lib = self._get_motif_library(task)
        orientations_grid = list(product(["fwd", "rc"], repeat=2))

        sequences: list[str] = []
        meta_records: list[dict[str, Any]] = []
        idx = 0

        for spacing in spacings:
            for orient_a, orient_b in orientations_grid:
                # Resolve motifs
                raw_a = _resolve_iupac(motif_lib[motif_pair[0]], self._rng)
                raw_b = _resolve_iupac(motif_lib[motif_pair[1]], self._rng)
                if orient_a == "rc":
                    raw_a = _reverse_complement(raw_a)
                if orient_b == "rc":
                    raw_b = _reverse_complement(raw_b)

                # Build block
                spacer_len = spacing
                block_len = len(raw_a) + spacer_len + len(raw_b)
                if block_len > seq_len:
                    continue

                for rep in range(n_replicates):
                    bg_idx = self._rng.integers(0, 4, size=seq_len, dtype=np.uint8)
                    background = list(_NUC_BYTES[bg_idx].tobytes().decode("ascii"))

                    # Random spacer
                    spacer_idx = self._rng.integers(0, 4, size=spacer_len, dtype=np.uint8)
                    spacer = (
                        _NUC_BYTES[spacer_idx].tobytes().decode("ascii") if spacer_len > 0 else ""
                    )
                    block = raw_a + spacer + raw_b

                    max_start = max(0, seq_len - len(block))
                    start = int(self._rng.integers(0, max_start + 1)) if max_start > 0 else 0

                    for j, c in enumerate(block):
                        if start + j < seq_len:
                            background[start + j] = c

                    core = "".join(background)
                    if task == "yeast":
                        full_seq = _YEAST_FLANK_5 + core + _YEAST_FLANK_3
                    else:
                        full_seq = core
                    sequences.append(full_seq)

                    meta_records.append(
                        {
                            "seq_idx": idx,
                            "method": "motif_grammar_systematic",
                            "source": "generated",
                            "motif_a": motif_pair[0],
                            "motif_b": motif_pair[1],
                            "orient_a": orient_a,
                            "orient_b": orient_b,
                            "spacing": spacing,
                            "replicate": rep,
                            "block_position": start,
                        }
                    )
                    idx += 1

        meta = pd.DataFrame(meta_records)
        n_configs = len(spacings) * len(orientations_grid)
        logger.info(
            f"Motif grammar systematic: {len(sequences)} sequences "
            f"({n_configs} configs × {n_replicates} replicates)"
        )
        return sequences, meta
