"""Partial Random Mutagenesis (PRM) reservoir sampler."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from albench.reservoir.base import ReservoirSampler

logger = logging.getLogger(__name__)

_NUCLEOTIDES = "ACGT"
_NUC_TO_IDX = {c: i for i, c in enumerate(_NUCLEOTIDES)}
_NUC_BYTES = np.frombuffer(b"ACGT", dtype=np.uint8)

# Yeast flanking sequences
_YEAST_FLANK_5 = "GCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCATTTTTTTCACATCTCG"
_YEAST_FLANK_3 = "GGTTACGGCTGTT"


def _seq_to_arr(seq: str) -> np.ndarray:
    """Convert DNA string to uint8 array (0=A, 1=C, 2=G, 3=T, 4=N)."""
    out = np.full(len(seq), 4, dtype=np.uint8)
    for i, c in enumerate(seq.upper()):
        if c in _NUC_TO_IDX:
            out[i] = _NUC_TO_IDX[c]
    return out


def _arr_to_seq(arr: np.ndarray) -> str:
    """Convert uint8 array back to DNA string."""
    lut = np.frombuffer(b"ACGTN", dtype=np.uint8)
    return lut[arr].tobytes().decode("ascii")


class PartialMutagenesisSampler(ReservoirSampler):
    """Generate sequences by introducing random point mutations.

    Vectorized for large-scale generation. For yeast, only the 80bp random
    region is mutated (flanks preserved).
    """

    def __init__(
        self,
        seed: int | None = None,
        mutation_rate_distribution: str = "fixed",
        mutation_rate: float = 0.05,
        min_rate: float = 0.01,
        max_rate: float = 0.10,
        mean_rate: float = 0.05,
        std_rate: float = 0.02,
        track_mutations: bool = True,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self.mutation_rate_distribution = mutation_rate_distribution
        self.mutation_rate = mutation_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.mean_rate = mean_rate
        self.std_rate = std_rate
        self.track_mutations = track_mutations

    def sample(
        self,
        candidates: list[str],
        n_samples: int,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Backward-compatible: randomly sample from candidates."""
        if n_samples > len(candidates):
            raise ValueError("n_samples cannot exceed number of candidates")
        return self._rng.choice(len(candidates), size=n_samples, replace=False).tolist()

    def _get_mutation_rates(self, n: int) -> np.ndarray:
        """Vectorized: get per-sequence mutation rates."""
        dist = self.mutation_rate_distribution
        if dist == "fixed":
            return np.full(n, self.mutation_rate, dtype=np.float64)
        elif dist == "uniform":
            return self._rng.uniform(self.min_rate, self.max_rate, size=n)
        elif dist == "normal":
            return np.clip(self._rng.normal(self.mean_rate, self.std_rate, size=n), 0.0, 1.0)
        elif dist == "poisson":
            # For poisson, we return rates that will be used differently
            return np.full(n, self.mutation_rate, dtype=np.float64)
        else:
            raise ValueError(f"Unknown mutation_rate_distribution: {dist}")

    def generate(
        self,
        n_sequences: int,
        base_sequences: list[str] | np.ndarray,
        task: str = "k562",
        batch_size: int = 50_000,
    ) -> tuple[list[str], pd.DataFrame]:
        """Generate mutated sequences from base sequences.

        Uses vectorized numpy operations for speed on large batches.

        Args:
            n_sequences: Number of mutated sequences to produce.
            base_sequences: Source sequences to mutagenize.
            task: ``"k562"`` or ``"yeast"``.
            batch_size: Process in batches to limit memory.

        Returns:
            Tuple of (mutated_sequences, metadata_df).
        """
        n_base = len(base_sequences)
        parent_indices = self._rng.choice(n_base, size=n_sequences, replace=True)

        all_sequences: list[str] = []
        all_n_mutations: list[int] = []
        all_rates_actual: list[float] = []
        # Only populated when track_mutations=True
        all_positions: list[list[int]] = []
        all_old_bases: list[list[str]] = []
        all_new_bases: list[list[str]] = []

        for batch_start in range(0, n_sequences, batch_size):
            batch_end = min(batch_start + batch_size, n_sequences)
            batch_pi = parent_indices[batch_start:batch_end]
            n_batch = len(batch_pi)

            # Extract regions to mutate
            regions: list[str] = []
            for pi in batch_pi:
                parent = str(base_sequences[pi])
                if task == "yeast":
                    if parent.startswith(_YEAST_FLANK_5) and parent.endswith(_YEAST_FLANK_3):
                        region = parent[len(_YEAST_FLANK_5) : -len(_YEAST_FLANK_3)]
                    else:
                        region = parent[:80] if len(parent) >= 80 else parent
                else:
                    region = parent
                regions.append(region.upper())

            # Vectorized mutation
            rates = self._get_mutation_rates(n_batch)

            for i, region in enumerate(regions):
                region_len = len(region)
                if self.mutation_rate_distribution == "poisson":
                    n_mut = min(int(self._rng.poisson(rates[i] * region_len)), region_len)
                else:
                    n_mut = min(max(0, int(round(rates[i] * region_len))), region_len)
                # Ensure at least 1 mutation: prevents round-to-zero for short
                # sequences (e.g. yeast 80bp with rate 0.005 → round(0.4) = 0).
                if n_mut == 0 and region_len > 0:
                    n_mut = 1

                if n_mut == 0:
                    mutated = region
                    positions, old_b, new_b = [], [], []
                else:
                    arr = _seq_to_arr(region)
                    positions = self._rng.choice(region_len, size=n_mut, replace=False).tolist()
                    old_b = [_NUCLEOTIDES[arr[p]] if arr[p] < 4 else "N" for p in positions]
                    # Vectorized alternative selection: shift by 1-3 positions
                    shifts = self._rng.integers(1, 4, size=n_mut, dtype=np.uint8)
                    new_vals = (arr[positions] + shifts) % 4
                    arr[positions] = new_vals
                    new_b = [_NUCLEOTIDES[v] for v in new_vals]
                    mutated = _arr_to_seq(arr)

                if task == "yeast":
                    full_seq = _YEAST_FLANK_5 + mutated + _YEAST_FLANK_3
                else:
                    full_seq = mutated

                all_sequences.append(full_seq)
                all_n_mutations.append(n_mut)
                all_rates_actual.append(n_mut / max(region_len, 1))

                if self.track_mutations:
                    all_positions.append(positions)
                    all_old_bases.append(old_b)
                    all_new_bases.append(new_b)

        meta_dict: dict[str, Any] = {
            "seq_idx": np.arange(n_sequences, dtype=np.int64),
            "method": f"prm_{self.mutation_rate_distribution}",
            "source": "mutagenized",
            "parent_idx": parent_indices,
            "n_mutations": np.array(all_n_mutations, dtype=np.int32),
            "mutation_rate_actual": np.array(all_rates_actual, dtype=np.float32),
        }
        if self.track_mutations:
            meta_dict["mutation_positions"] = all_positions
            meta_dict["original_bases"] = all_old_bases
            meta_dict["new_bases"] = all_new_bases

        meta = pd.DataFrame(meta_dict)
        mean_muts = meta["n_mutations"].mean()
        logger.info(
            f"PRM generated {n_sequences:,} sequences "
            f"(dist={self.mutation_rate_distribution}, mean_mutations={mean_muts:.1f})"
        )
        return all_sequences, meta
