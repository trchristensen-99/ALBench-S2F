"""EvoAug structural variation reservoir sampler.

Generates sequences by applying biologically realistic structural mutations
to pool sequences: deletions, insertions, inversions, translocations, and
tandem duplications. These mimic real genetic variation mechanisms beyond
simple point mutations (SNPs).

Each variation type has independent probability and size distribution,
allowing control over the complexity and magnitude of changes.
"""

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
_RC_MAP = str.maketrans("ACGT", "TGCA")


def _reverse_complement(seq: str) -> str:
    return seq.upper().translate(_RC_MAP)[::-1]


class EvoAugStructuralSampler(ReservoirSampler):
    """Generate sequences by applying structural variations to pool sequences.

    Applies a random combination of:
    - **Deletion**: remove a contiguous block (1-max_indel_size bp)
    - **Insertion**: insert random bases at a position (1-max_indel_size bp)
    - **Inversion**: reverse-complement a contiguous block in place
    - **Translocation**: swap two non-overlapping blocks
    - **Tandem duplication**: duplicate a block adjacent to its original position

    After all mutations, the sequence is trimmed or padded to the target length.
    """

    def __init__(
        self,
        seed: int | None = None,
        p_deletion: float = 0.3,
        p_insertion: float = 0.3,
        p_inversion: float = 0.2,
        p_translocation: float = 0.15,
        p_tandem_dup: float = 0.1,
        p_point_mutation: float = 0.3,
        max_indel_size: int = 20,
        max_inversion_size: int = 30,
        max_translocation_size: int = 25,
        max_dup_size: int = 15,
        point_mutation_rate: float = 0.02,
        min_events: int = 1,
        max_events: int = 3,
    ) -> None:
        """Initialize sampler.

        Args:
            seed: Random seed.
            p_deletion: Probability of applying a deletion event.
            p_insertion: Probability of applying an insertion event.
            p_inversion: Probability of applying an inversion event.
            p_translocation: Probability of applying a translocation event.
            p_tandem_dup: Probability of applying a tandem duplication.
            p_point_mutation: Probability of also applying point mutations.
            max_indel_size: Maximum size of deletions/insertions (bp).
            max_inversion_size: Maximum size of inversions (bp).
            max_translocation_size: Maximum size of translocation blocks (bp).
            max_dup_size: Maximum size of tandem duplications (bp).
            point_mutation_rate: Rate of point mutations when applied.
            min_events: Minimum structural events per sequence.
            max_events: Maximum structural events per sequence.
        """
        self._rng = np.random.default_rng(seed)
        self.p_deletion = p_deletion
        self.p_insertion = p_insertion
        self.p_inversion = p_inversion
        self.p_translocation = p_translocation
        self.p_tandem_dup = p_tandem_dup
        self.p_point_mutation = p_point_mutation
        self.max_indel_size = max_indel_size
        self.max_inversion_size = max_inversion_size
        self.max_translocation_size = max_translocation_size
        self.max_dup_size = max_dup_size
        self.point_mutation_rate = point_mutation_rate
        self.min_events = min_events
        self.max_events = max_events

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

    def _apply_deletion(self, seq: list[str]) -> tuple[list[str], dict[str, Any]]:
        """Delete a contiguous block."""
        if len(seq) <= 2:
            return seq, {"type": "deletion", "size": 0, "pos": 0}
        size = int(self._rng.integers(1, min(self.max_indel_size, len(seq) - 1) + 1))
        pos = int(self._rng.integers(0, len(seq) - size))
        result = seq[:pos] + seq[pos + size :]
        return result, {"type": "deletion", "size": size, "pos": pos}

    def _apply_insertion(self, seq: list[str]) -> tuple[list[str], dict[str, Any]]:
        """Insert random bases at a position."""
        size = int(self._rng.integers(1, self.max_indel_size + 1))
        pos = int(self._rng.integers(0, len(seq) + 1))
        insert_idx = self._rng.integers(0, 4, size=size, dtype=np.uint8)
        insert_seq = list(_NUC_BYTES[insert_idx].tobytes().decode("ascii"))
        result = seq[:pos] + insert_seq + seq[pos:]
        return result, {"type": "insertion", "size": size, "pos": pos}

    def _apply_inversion(self, seq: list[str]) -> tuple[list[str], dict[str, Any]]:
        """Reverse-complement a contiguous block in place."""
        if len(seq) <= 2:
            return seq, {"type": "inversion", "size": 0, "pos": 0}
        size = int(self._rng.integers(2, min(self.max_inversion_size, len(seq)) + 1))
        pos = int(self._rng.integers(0, len(seq) - size + 1))
        block = "".join(seq[pos : pos + size])
        rc_block = list(_reverse_complement(block))
        result = seq[:pos] + rc_block + seq[pos + size :]
        return result, {"type": "inversion", "size": size, "pos": pos}

    def _apply_translocation(self, seq: list[str]) -> tuple[list[str], dict[str, Any]]:
        """Swap two non-overlapping blocks."""
        if len(seq) < 4:
            return seq, {"type": "translocation", "size": 0, "pos_a": 0, "pos_b": 0}
        size = int(self._rng.integers(1, min(self.max_translocation_size, len(seq) // 3) + 1))
        # Pick two non-overlapping positions
        pos_a = int(self._rng.integers(0, len(seq) - 2 * size))
        pos_b = int(
            self._rng.integers(pos_a + size, min(pos_a + size + len(seq) // 2, len(seq) - size + 1))
        )
        if pos_b + size > len(seq):
            return seq, {"type": "translocation", "size": 0, "pos_a": pos_a, "pos_b": pos_b}
        block_a = seq[pos_a : pos_a + size]
        block_b = seq[pos_b : pos_b + size]
        result = list(seq)
        result[pos_a : pos_a + size] = block_b
        result[pos_b : pos_b + size] = block_a
        return result, {"type": "translocation", "size": size, "pos_a": pos_a, "pos_b": pos_b}

    def _apply_tandem_dup(self, seq: list[str]) -> tuple[list[str], dict[str, Any]]:
        """Duplicate a block adjacent to its original position."""
        if len(seq) <= 2:
            return seq, {"type": "tandem_dup", "size": 0, "pos": 0}
        size = int(self._rng.integers(1, min(self.max_dup_size, len(seq) // 2) + 1))
        pos = int(self._rng.integers(0, len(seq) - size + 1))
        block = seq[pos : pos + size]
        result = seq[: pos + size] + block + seq[pos + size :]
        return result, {"type": "tandem_dup", "size": size, "pos": pos}

    def _apply_point_mutations(self, seq: list[str]) -> tuple[list[str], int]:
        """Apply random point mutations at the configured rate."""
        n = len(seq)
        n_mut = int(self._rng.binomial(n, self.point_mutation_rate))
        if n_mut == 0:
            return seq, 0
        positions = self._rng.choice(n, size=n_mut, replace=False)
        result = list(seq)
        nuc_map = {"A": 0, "C": 1, "G": 2, "T": 3}
        for pos in positions:
            old = nuc_map.get(result[pos].upper(), 0)
            shift = int(self._rng.integers(1, 4))
            result[pos] = "ACGT"[(old + shift) % 4]
        return result, n_mut

    def _trim_or_pad(self, seq: list[str], target_len: int) -> list[str]:
        """Trim or pad sequence to target length."""
        if len(seq) == target_len:
            return seq
        if len(seq) > target_len:
            # Center-crop
            start = (len(seq) - target_len) // 2
            return seq[start : start + target_len]
        # Pad with random bases on both sides
        pad_total = target_len - len(seq)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        left_idx = self._rng.integers(0, 4, size=pad_left, dtype=np.uint8)
        right_idx = self._rng.integers(0, 4, size=pad_right, dtype=np.uint8)
        left = list(_NUC_BYTES[left_idx].tobytes().decode("ascii"))
        right = list(_NUC_BYTES[right_idx].tobytes().decode("ascii"))
        return left + seq + right

    def generate(
        self,
        n_sequences: int,
        base_sequences: list[str] | np.ndarray,
        task: str = "k562",
    ) -> tuple[list[str], pd.DataFrame]:
        """Generate structurally mutated sequences from pool.

        Args:
            n_sequences: Number of sequences to produce.
            base_sequences: Source sequences to mutate.
            task: ``"k562"`` or ``"yeast"``.

        Returns:
            Tuple of (mutated_sequences, metadata_df).
        """
        seq_len = 200 if task == "k562" else 80
        n_pool = len(base_sequences)
        parent_indices = self._rng.choice(n_pool, size=n_sequences, replace=True)

        # Event types and their probabilities
        event_types = [
            ("deletion", self.p_deletion, self._apply_deletion),
            ("insertion", self.p_insertion, self._apply_insertion),
            ("inversion", self.p_inversion, self._apply_inversion),
            ("translocation", self.p_translocation, self._apply_translocation),
            ("tandem_dup", self.p_tandem_dup, self._apply_tandem_dup),
        ]

        sequences: list[str] = []
        meta_records: list[dict[str, Any]] = []

        for idx in range(n_sequences):
            parent = str(base_sequences[parent_indices[idx]])

            # Extract mutable region
            if task == "yeast":
                if parent.startswith(_YEAST_FLANK_5) and parent.endswith(_YEAST_FLANK_3):
                    region = parent[len(_YEAST_FLANK_5) : -len(_YEAST_FLANK_3)]
                else:
                    region = parent[:80] if len(parent) >= 80 else parent
            else:
                region = parent

            seq = list(region.upper())
            events_applied: list[str] = []
            event_details: list[dict[str, Any]] = []

            # Determine number of structural events
            n_events = int(self._rng.integers(self.min_events, self.max_events + 1))

            for _ in range(n_events):
                # Sample which event to apply (weighted by probabilities)
                probs = np.array([p for _, p, _ in event_types])
                if probs.sum() == 0:
                    break
                probs = probs / probs.sum()
                choice = int(self._rng.choice(len(event_types), p=probs))
                name, _, fn = event_types[choice]
                seq, detail = fn(seq)
                events_applied.append(name)
                event_details.append(detail)

            # Optionally apply point mutations on top
            n_point = 0
            if self._rng.random() < self.p_point_mutation:
                seq, n_point = self._apply_point_mutations(seq)

            # Trim/pad to target length
            seq = self._trim_or_pad(seq, seq_len)
            core = "".join(seq)

            if task == "yeast":
                full_seq = _YEAST_FLANK_5 + core + _YEAST_FLANK_3
            else:
                full_seq = core

            sequences.append(full_seq)
            meta_records.append(
                {
                    "seq_idx": idx,
                    "method": "evoaug_structural",
                    "source": "structurally_mutated",
                    "parent_idx": int(parent_indices[idx]),
                    "n_structural_events": len(events_applied),
                    "event_types": ",".join(events_applied),
                    "n_point_mutations": n_point,
                }
            )

        meta = pd.DataFrame(meta_records)
        event_counts = meta["event_types"].str.split(",").explode().value_counts()
        logger.info(
            f"EvoAug structural: {n_sequences:,} sequences, "
            f"mean events={meta['n_structural_events'].mean():.1f}, "
            f"event distribution: {dict(event_counts)}"
        )
        return sequences, meta
