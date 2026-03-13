"""Random reservoir sampler and sequence generator."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from albench.reservoir.base import ReservoirSampler

logger = logging.getLogger(__name__)

# Yeast flanking sequences (from de Boer et al.)
_YEAST_FLANK_5 = "GCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCATTTTTTTCACATCTCG"
_YEAST_FLANK_3 = "GGTTACGGCTGTT"

# Pre-computed lookup: integer 0-3 → nucleotide byte
_NUC_BYTES = np.frombuffer(b"ACGT", dtype=np.uint8)


def _ints_to_seqs(indices: np.ndarray) -> list[str]:
    """Vectorized conversion of (N, L) integer array to list of DNA strings.

    ~10x faster than per-row join for large N.
    """
    flat = _NUC_BYTES[indices.ravel()]
    n, seq_len = indices.shape
    byte_arr = flat.reshape(n, seq_len)
    return [row.tobytes().decode("ascii") for row in byte_arr]


class RandomSampler(ReservoirSampler):
    """Uniform random sampler and sequence generator.

    As a sampler (``sample()``): selects indices uniformly from candidates.
    As a generator (``generate()``): creates new random DNA sequences.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def sample(
        self,
        candidates: list[str],
        n_samples: int,
        metadata: list[dict[str, object]] | None = None,
    ) -> list[int]:
        """Sample indices uniformly without replacement."""
        if n_samples > len(candidates):
            raise ValueError("n_samples cannot exceed number of candidates")
        return self._rng.choice(len(candidates), size=n_samples, replace=False).tolist()

    def generate(
        self,
        n_sequences: int,
        task: str,
        method: str = "uniform",
        reference_sequences: list[str] | np.ndarray | None = None,
        batch_size: int = 100_000,
    ) -> tuple[list[str], pd.DataFrame]:
        """Generate random DNA sequences.

        Args:
            n_sequences: Number of sequences to generate.
            task: ``"k562"`` (200bp) or ``"yeast"`` (80bp random + flanks = 150bp).
            method: ``"uniform"`` for i.i.d. nucleotides, ``"dinuc_shuffle"`` for
                dinucleotide-preserving shuffles of reference sequences.
            reference_sequences: Required when method is ``"dinuc_shuffle"``.
            batch_size: Generate in batches to limit memory.

        Returns:
            Tuple of (sequences, metadata_df).
        """
        if method == "dinuc_shuffle":
            return self._generate_dinuc_shuffle(n_sequences, task, reference_sequences)

        seq_len = 200 if task == "k562" else 80
        sequences: list[str] = []

        for start in range(0, n_sequences, batch_size):
            n_batch = min(batch_size, n_sequences - start)
            indices = self._rng.integers(0, 4, size=(n_batch, seq_len), dtype=np.uint8)
            batch = _ints_to_seqs(indices)
            if task == "yeast":
                batch = [_YEAST_FLANK_5 + s + _YEAST_FLANK_3 for s in batch]
            sequences.extend(batch)

        meta = pd.DataFrame(
            {
                "seq_idx": np.arange(n_sequences, dtype=np.int64),
                "method": "uniform_random",
                "source": "generated",
            }
        )
        return sequences, meta

    def _generate_dinuc_shuffle(
        self,
        n_sequences: int,
        task: str,
        reference_sequences: list[str] | np.ndarray | None,
    ) -> tuple[list[str], pd.DataFrame]:
        """Dinucleotide-preserving shuffle of reference sequences."""
        if reference_sequences is None or len(reference_sequences) == 0:
            raise ValueError("reference_sequences required for dinuc_shuffle")

        n_ref = len(reference_sequences)
        parent_idx = self._rng.choice(n_ref, size=n_sequences, replace=n_sequences > n_ref)

        sequences: list[str] = []
        for pi in parent_idx:
            seq = str(reference_sequences[pi])
            if task == "yeast":
                if seq.startswith(_YEAST_FLANK_5) and seq.endswith(_YEAST_FLANK_3):
                    region = seq[len(_YEAST_FLANK_5) : -len(_YEAST_FLANK_3)]
                else:
                    region = seq[:80] if len(seq) >= 80 else seq
                shuffled = _dinuc_shuffle(region, self._rng)
                sequences.append(_YEAST_FLANK_5 + shuffled + _YEAST_FLANK_3)
            else:
                sequences.append(_dinuc_shuffle(seq, self._rng))

        meta = pd.DataFrame(
            {
                "seq_idx": np.arange(n_sequences, dtype=np.int64),
                "method": "dinuc_shuffle",
                "source": "shuffled",
                "parent_idx": parent_idx,
            }
        )
        return sequences, meta


def _dinuc_shuffle(seq: str, rng: np.random.Generator) -> str:
    """Altschul-Erickson dinucleotide-preserving shuffle via Euler path."""
    if len(seq) <= 2:
        return seq

    seq = seq.upper()
    nucs = sorted(set(seq))
    nuc_to_idx = {n: i for i, n in enumerate(nucs)}
    n = len(nucs)

    edges: list[list[int]] = [[] for _ in range(n)]
    for i in range(len(seq) - 1):
        a, b = nuc_to_idx[seq[i]], nuc_to_idx[seq[i + 1]]
        edges[a].append(b)

    for i in range(n):
        rng.shuffle(edges[i])

    start = nuc_to_idx[seq[0]]
    stack = [start]
    path: list[int] = []
    edge_pos = [0] * n

    while stack:
        v = stack[-1]
        if edge_pos[v] < len(edges[v]):
            u = edges[v][edge_pos[v]]
            edge_pos[v] += 1
            stack.append(u)
        else:
            path.append(stack.pop())

    path.reverse()
    idx_to_nuc = {i: c for c, i in nuc_to_idx.items()}
    return "".join(idx_to_nuc[p] for p in path)
