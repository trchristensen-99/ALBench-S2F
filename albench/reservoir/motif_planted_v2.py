"""Motif-planted v2: GC-matched real-genomic background + denser motif planting.

Differences from `motif_planted`:
  1. Background = real genomic sequences sampled from the chr_train pool (GC-matched
     to K562 regulatory regions by construction), not pure uniform random.
  2. More motifs per sequence: 3-7 instead of 1-3 (matches typical regulatory
     element density at ~200bp).
  3. Includes reverse-complement motif variants (e.g. TGATTT plants alongside TATAAA's
     paradigm direction). Doubles effective motif set.
  4. Optional `preserve_native_motifs=True`: if a motif site already exists in the
     background, keep it instead of overwriting (avoids destroying real signal when
     planting our literal motif overrides it).

For consistency with the original motif_planted: same 9 K562 motifs, same task=k562
default, same sequence length 200bp.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from albench.reservoir.base import ReservoirSampler

logger = logging.getLogger(__name__)

_NUC_BYTES = np.frombuffer(b"ACGT", dtype=np.uint8)

K562_MOTIFS = [
    "TATAAA",  # TATA box
    "CACGTG",  # E-box (MYC, MAX)
    "GGGCGG",  # SP1
    "CCAAT",  # NF-Y / CCAAT-box
    "GATA",  # GATA factors
    "AGATAA",  # GATA1 consensus
    "CTCFCC",  # CTCF partial (6bp)
    "TGACGTCA",  # AP-1 / CRE
    "TGAGTCA",  # AP-1
]

YEAST_MOTIFS = [
    "TATAAA",
    "GCGATGAG",
    "CACGTG",
    "CCAAT",
    "ACCCG",
    "TGACTC",
]

_REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
_DEFAULT_BG_CACHE = _REPO / "outputs/chr_split_cache/chr_train_ref_only.npz"


def _bg_cache_path() -> Path:
    """Genomic-background pool used for planting/shuffling/mutating. Override via the
    RESERVOIR_BG_CACHE env var to generate a held-out, transform-matched VAL set from
    chr19/21/X backgrounds (outputs/chr_split_cache/chr_val_ref_only.npz); defaults to
    the chr-train pool so normal train-cache generation is unchanged."""
    return Path(os.environ.get("RESERVOIR_BG_CACHE", str(_DEFAULT_BG_CACHE)))


def _rc(s: str) -> str:
    """Reverse complement (no N handling)."""
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(comp.get(b, "N") for b in s[::-1])


class MotifPlantedV2Sampler(ReservoirSampler):
    """Motif-planted v2 — GC-matched real backgrounds + denser planting.

    See module docstring for differences from MotifPlantedSampler.
    """

    def __init__(
        self,
        seed: int | None = None,
        min_motifs: int = 3,
        max_motifs: int = 7,
        motif_set: str = "auto",
        include_rc_variants: bool = True,
        preserve_native_motifs: bool = True,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self.min_motifs = min_motifs
        self.max_motifs = max_motifs
        self.motif_set = motif_set
        self.include_rc_variants = include_rc_variants
        self.preserve_native_motifs = preserve_native_motifs
        self._bg_seqs: np.ndarray | None = None  # lazy

    def _get_motifs(self, task: str) -> list[str]:
        if self.motif_set == "k562" or (self.motif_set == "auto" and task == "k562"):
            motifs = list(K562_MOTIFS)
        else:
            motifs = list(YEAST_MOTIFS)
        if self.include_rc_variants:
            motifs = list(set(motifs + [_rc(m) for m in motifs]))
        return motifs

    def _load_backgrounds(self) -> np.ndarray:
        if self._bg_seqs is None:
            z = np.load(_bg_cache_path(), allow_pickle=True)
            self._bg_seqs = np.array([str(s)[:200].ljust(200, "N") for s in z["sequences"]])
            logger.info(f"MotifPlantedV2: loaded {len(self._bg_seqs):,} genomic backgrounds")
        return self._bg_seqs

    def sample(self, candidates, n_samples, metadata=None):
        if n_samples > len(candidates):
            raise ValueError("n_samples cannot exceed number of candidates")
        return self._rng.choice(len(candidates), size=n_samples, replace=False).tolist()

    def generate(self, n_sequences: int, task: str = "k562") -> tuple[list[str], pd.DataFrame]:
        seq_len = 200 if task == "k562" else 80
        motifs = self._get_motifs(task)
        backgrounds = self._load_backgrounds() if task == "k562" else None

        sequences: list[str] = []
        n_planted_list: list[int] = []
        n_native_kept: list[int] = []
        planted_motifs_list: list[str] = []

        for i in range(n_sequences):
            # Background: real genomic if available, else uniform random
            if backgrounds is not None:
                bg_idx = self._rng.integers(0, len(backgrounds))
                core = list(backgrounds[bg_idx][:seq_len])
                # Pad if needed
                while len(core) < seq_len:
                    core.append(str(self._rng.choice(["A", "C", "G", "T"])))
            else:
                indices = self._rng.integers(0, 4, size=seq_len, dtype=np.uint8)
                core = list(_NUC_BYTES[indices].tobytes().decode("ascii"))

            # Locate native motif occurrences (if preserving)
            native = 0
            occupied = set()
            if self.preserve_native_motifs:
                core_str = "".join(core)
                for motif in motifs:
                    pos = core_str.find(motif)
                    while pos != -1:
                        occupied |= set(range(pos, pos + len(motif)))
                        native += 1
                        pos = core_str.find(motif, pos + 1)

            n_plant = self._rng.integers(self.min_motifs, self.max_motifs + 1)
            chosen_motifs = self._rng.choice(motifs, size=n_plant, replace=True).tolist()
            planted_log = []
            for motif in chosen_motifs:
                motif_len = len(motif)
                if motif_len >= seq_len:
                    continue
                for _ in range(20):
                    pos = int(self._rng.integers(0, seq_len - motif_len))
                    positions_needed = set(range(pos, pos + motif_len))
                    if not positions_needed & occupied:
                        for j, c in enumerate(motif):
                            core[pos + j] = c
                        occupied |= positions_needed
                        planted_log.append(motif)
                        break

            sequences.append("".join(core))
            n_planted_list.append(len(planted_log))
            n_native_kept.append(native)
            planted_motifs_list.append(",".join(planted_log) if planted_log else "none")

        meta = pd.DataFrame(
            {
                "seq_idx": np.arange(n_sequences, dtype=np.int64),
                "method": "motif_planted_v2_gc_matched",
                "source": "generated",
                "planted_motifs": planted_motifs_list,
                "n_motifs_planted": np.array(n_planted_list, dtype=np.int32),
                "n_native_motifs_preserved": np.array(n_native_kept, dtype=np.int32),
            }
        )
        logger.info(
            f"MotifPlantedV2: {n_sequences:,} seqs, mean planted={np.mean(n_planted_list):.1f}, "
            f"mean native preserved={np.mean(n_native_kept):.1f}"
        )
        return sequences, meta


class MotifShuffledSampler(ReservoirSampler):
    """Motif-shuffled: take real genomic sequences, find motif occurrences, permute their positions.

    Preserves: the background bases, the set of motifs present.
    Changes: the positions of motifs (shuffled to other valid positions).

    This isolates the effect of motif POSITION/SPACING from motif PRESENCE — i.e.,
    does the model use only the motifs' identity or also their spatial arrangement?
    """

    def __init__(
        self,
        seed: int | None = None,
        motif_set: str = "auto",
        include_rc_variants: bool = True,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self.motif_set = motif_set
        self.include_rc_variants = include_rc_variants

    def _get_motifs(self, task: str) -> list[str]:
        motifs = list(
            K562_MOTIFS
            if (self.motif_set == "k562" or (self.motif_set == "auto" and task == "k562"))
            else YEAST_MOTIFS
        )
        if self.include_rc_variants:
            motifs = list(set(motifs + [_rc(m) for m in motifs]))
        return motifs

    def sample(self, candidates, n_samples, metadata=None):
        if n_samples > len(candidates):
            raise ValueError("n_samples cannot exceed number of candidates")
        return self._rng.choice(len(candidates), size=n_samples, replace=False).tolist()

    def generate(
        self, n_sequences: int, base_sequences=None, task: str = "k562"
    ) -> tuple[list[str], pd.DataFrame]:
        # `base_sequences` is passed by exp1_1_scaling.py for pool-derived strategies
        if base_sequences is None or len(base_sequences) == 0:
            z = np.load(_bg_cache_path(), allow_pickle=True)
            base_sequences = [str(s) for s in z["sequences"]]

        base_arr = np.array([str(s) for s in base_sequences])
        motifs = self._get_motifs(task)
        seq_len = 200

        sequences: list[str] = []
        n_shuffled_list: list[int] = []
        motifs_found_list: list[str] = []

        chosen_idx = self._rng.choice(len(base_arr), size=n_sequences, replace=True)
        for i in range(n_sequences):
            seq = str(base_arr[chosen_idx[i]])[:seq_len].ljust(seq_len, "N")
            # Find motif occurrences
            sites: list[tuple[int, str]] = []
            for m in motifs:
                pos = seq.find(m)
                while pos != -1:
                    sites.append((pos, m))
                    pos = seq.find(m, pos + 1)
            if not sites:
                # No motifs to shuffle — return seq unchanged
                sequences.append(seq)
                n_shuffled_list.append(0)
                motifs_found_list.append("none")
                continue

            # Sort by position; extract non-overlapping subset (greedy)
            sites.sort()
            non_overlap = []
            last_end = -1
            for pos, m in sites:
                if pos >= last_end:
                    non_overlap.append((pos, m))
                    last_end = pos + len(m)

            # Erase motifs (replace with original bg bases — i.e., the original chars at those positions
            # WOULD be the motif itself; we need to fill with random ACGT or pick neutral fill).
            # Simplest: erase to random ACGT (since motif IS the original content, can't recover original bg).
            chars = list(seq)
            for pos, m in non_overlap:
                for j in range(len(m)):
                    chars[pos + j] = str(self._rng.choice(["A", "C", "G", "T"]))
            # Find new random non-overlapping positions for each motif
            occupied = set()
            placed = []
            for pos, m in non_overlap:
                for _ in range(20):
                    new_pos = int(self._rng.integers(0, seq_len - len(m)))
                    span = set(range(new_pos, new_pos + len(m)))
                    if not span & occupied:
                        for j, c in enumerate(m):
                            chars[new_pos + j] = c
                        occupied |= span
                        placed.append(m)
                        break
            sequences.append("".join(chars))
            n_shuffled_list.append(len(placed))
            motifs_found_list.append(",".join(placed))

        meta = pd.DataFrame(
            {
                "seq_idx": np.arange(n_sequences, dtype=np.int64),
                "method": "motif_shuffled",
                "source": "shuffled_from_pool",
                "motifs": motifs_found_list,
                "n_motifs_shuffled": np.array(n_shuffled_list, dtype=np.int32),
            }
        )
        logger.info(
            f"MotifShuffled: {n_sequences:,} seqs, mean motifs shuffled={np.mean(n_shuffled_list):.1f}"
        )
        return sequences, meta


class PhylogeneticZoonomiaSampler(ReservoirSampler):
    """Phylogenetic-variation reservoir (Zoonomia-rate-matched substitutions).

    Approximation: real Zoonomia gives per-position substitution rates across
    ~240 mammalian species. We don't have those rates wired here, so this sampler
    applies position-independent substitutions at a rate matched to Zoonomia's
    average mammalian-conserved-element rate (~3-5% genome-wide; ~1-2% in
    conserved enhancers; we use 2% as a compromise).

    For the proper version: load `data/zoonomia/per_position_rates.npz` (TODO)
    and sample mutations according to per-base rates.
    """

    DEFAULT_RATE = 0.02

    def __init__(self, seed: int | None = None, mut_rate: float | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self.mut_rate = mut_rate or self.DEFAULT_RATE

    def sample(self, candidates, n_samples, metadata=None):
        if n_samples > len(candidates):
            raise ValueError("n_samples cannot exceed number of candidates")
        return self._rng.choice(len(candidates), size=n_samples, replace=False).tolist()

    def generate(
        self, n_sequences: int, base_sequences=None, task: str = "k562"
    ) -> tuple[list[str], pd.DataFrame]:
        if base_sequences is None or len(base_sequences) == 0:
            z = np.load(_bg_cache_path(), allow_pickle=True)
            base_sequences = [str(s) for s in z["sequences"]]
        base_arr = np.array([str(s) for s in base_sequences])
        seq_len = 200

        sequences: list[str] = []
        n_mut_list: list[int] = []
        idx = self._rng.choice(len(base_arr), size=n_sequences, replace=True)
        bases = "ACGT"
        for i in range(n_sequences):
            seq = list(str(base_arr[idx[i]])[:seq_len].ljust(seq_len, "N"))
            mask = self._rng.random(seq_len) < self.mut_rate
            n_mut = 0
            for j in range(seq_len):
                if mask[j] and seq[j] in bases:
                    original = seq[j]
                    choices = [b for b in bases if b != original]
                    seq[j] = str(self._rng.choice(choices))
                    n_mut += 1
            sequences.append("".join(seq))
            n_mut_list.append(n_mut)

        meta = pd.DataFrame(
            {
                "seq_idx": np.arange(n_sequences, dtype=np.int64),
                "method": "phylogenetic_zoonomia_rate",
                "source": "pool_with_zoonomia_mutations",
                "n_mutations": np.array(n_mut_list, dtype=np.int32),
                "mut_rate": self.mut_rate,
            }
        )
        logger.info(
            f"PhylogeneticZoonomia: {n_sequences:,} seqs, mean mutations={np.mean(n_mut_list):.1f} (rate={self.mut_rate:.2%})"
        )
        return sequences, meta
