"""Real orthologous CRE sequences from the Zoonomia 241-mammal alignment.

WHY THIS REPLACES THE EARLIER ZOONOMIA ARM. The previous sampler took HUMAN cCREs
and mutated them at per-position rates derived from the alignment. That uses the
alignment as a source of conservation SCORES, so the arm was capped at however many
human cCREs exist (199,373) and every sequence was simulated, not observed. What was
actually wanted is the orthologous sequences themselves: for each human cCRE window,
the aligned sequence in each of the other 240 mammals.

CAPACITY, MEASURED rather than assumed. The naive figure is 199,373 windows x 240
species = ~48M, but only ~41 species are callable in enough windows at the default
min_called_frac=0.95 -- the rest are too gappy in cCRE regions to yield clean 200bp
orthologs. So the realistic pool is ~199,373 x 41 = ~8.2M sequences. Still ~40x the
old arm, still real evolved sequence rather than a mutation model, and still the
answer to "real human CREs do not scale, but real MAMMALIAN CREs do".

min_called_frac trades species count against sequence quality and is worth sweeping:
relaxing it to 0.8 admits more distant species at the cost of more gap-adjacent
sequence.

EVOLUTIONARY DISTANCE IS A PARAMETER, not a fixed choice. Species are ranked by
observed identity to human over a sample of windows, then binned. `distance` selects
a tier: sequences from close relatives are nearly human and test little; distant ones
approach novel sequence. Sweeping the tier is how we find where "real but diverged"
stops helping.

CAVEAT to state in any result: these are real sequences but NOT human, so the arm
tests cross-species generalisation as much as scale.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from albench.reservoir.base import ReservoirSampler

logger = logging.getLogger(__name__)

# uint8 encoding in the HDF5: 0=N/gap, 1=A, 2=C, 3=G, 4=T
CODE_TO_BASE = np.array(list("NACGT"), dtype="<U1")


class ZoonomiaOrthologSampler(ReservoirSampler):
    """Draw orthologous CRE windows from non-human mammals."""

    def __init__(
        self,
        seed: int | None = None,
        h5_path: str | None = None,
        distance: str = "all",
        min_called_frac: float = 0.95,
        max_species_per_window: int | None = None,
        exclude_chroms: tuple[str, ...] = ("chr7", "chr13"),
        n_rank_windows: int = 200,
    ) -> None:
        """Initialize sampler.

        Args:
            seed: Random seed.
            h5_path: Zoonomia alignment HDF5. Resolved via albench.paths if None.
            distance: ``near`` | ``mid`` | ``far`` | ``all`` -- which evolutionary
                tier to draw from, by observed identity to human.
            min_called_frac: Reject a window/species whose called (non-N, non-gap)
                fraction falls below this. A mostly-uncalled window is not an
                ortholog, it is a gap.
            max_species_per_window: Cap species drawn per window, so the pool is not
                dominated by a few deeply-sampled loci.
            exclude_chroms: Held-out chromosomes, never drawn from.
            n_rank_windows: Windows sampled to rank species by identity to human.
        """
        self._rng = np.random.default_rng(seed)
        self.h5_path = h5_path
        self.distance = distance
        self.min_called_frac = min_called_frac
        self.max_species_per_window = max_species_per_window
        self.exclude_chroms = tuple(exclude_chroms)
        self.n_rank_windows = n_rank_windows
        self._identity: np.ndarray | None = None
        self._species: list[str] | None = None

    # ---- internals -------------------------------------------------------
    def _open(self):
        import h5py

        path = self.h5_path
        if path is None:
            from albench.paths import resolve

            path = str(resolve("zoonomia_alignment"))
        return h5py.File(path, "r")

    def _windows(self, f, seq_len: int) -> list[tuple[str, int]]:
        """Candidate (chrom, start) windows on non-excluded chromosomes."""
        out = []
        for chrom in f:
            if not chrom.startswith("chr") or chrom in self.exclude_chroms:
                continue
            L = f[chrom]["seq"].shape[1]
            out.append((chrom, L))
        return out

    def _rank_species(self, f, chrom_lens, seq_len: int) -> np.ndarray:
        """Mean identity to human per species, over a sample of windows."""
        if self._identity is not None:
            return self._identity
        # Accumulate matches and called positions SEPARATELY, and count only windows
        # where that species is adequately called. Averaging per-window identities
        # over all windows lets a species with almost nothing called contribute
        # near-zero identity values, which pushed species BELOW the 0.25 expected by
        # chance and put alignment gaps -- not distant relatives -- in the far tier.
        matches = np.zeros(241, dtype=np.int64)
        called_n = np.zeros(241, dtype=np.int64)
        windows_ok = np.zeros(241, dtype=np.int64)
        seen = 0
        for _ in range(self.n_rank_windows):
            chrom, L = chrom_lens[self._rng.integers(len(chrom_lens))]
            start = int(self._rng.integers(0, L - seq_len))
            block = f[chrom]["seq"][:, start : start + seq_len]
            human = block[0]
            if (human == 0).mean() > 1 - self.min_called_frac:
                continue
            called = (block != 0) & (human != 0)
            frac = called.sum(1) / seq_len
            ok = frac >= self.min_called_frac
            matches += np.where(ok, ((block == human) & called).sum(1), 0)
            called_n += np.where(ok, called.sum(1), 0)
            windows_ok += ok.astype(np.int64)
            seen += 1
        if seen == 0:
            raise RuntimeError("could not sample any callable window to rank species")
        with np.errstate(invalid="ignore", divide="ignore"):
            ident = np.where(called_n > 0, matches / np.maximum(called_n, 1), np.nan)
        # A species callable in too few windows cannot be ranked; exclude rather than
        # assign it a fabricated identity.
        min_windows = max(1, int(0.25 * seen))
        ident[windows_ok < min_windows] = np.nan
        n_rankable = int(np.isfinite(ident[1:]).sum())
        logger.info(
            "ranked %d of 240 non-human species on %d windows "
            "(others callable in <%d windows)", n_rankable, seen, min_windows,
        )
        if n_rankable < 10:
            raise RuntimeError(
                f"only {n_rankable} species are callable often enough to rank; "
                f"lower min_called_frac or raise n_rank_windows"
            )
        self._identity = ident
        return self._identity

    def _tier_mask(self, identity: np.ndarray) -> np.ndarray:
        """Species indices in the requested evolutionary tier (human excluded)."""
        idx = np.arange(1, len(identity))           # never row 0 (human)
        idx = idx[np.isfinite(identity[idx])]       # drop unrankable species
        vals = identity[idx]
        if self.distance == "all":
            return idx
        order = idx[np.argsort(-vals)]              # most human-like first
        third = max(1, len(order) // 3)
        return {"near": order[:third], "mid": order[third : 2 * third], "far": order[2 * third :]}[
            self.distance
        ]

    # ---- public API ------------------------------------------------------
    def sample(
        self,
        candidates: list[str],
        n_samples: int,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Uniform choice among supplied candidates.

        This arm GENERATES rather than selects -- the interesting entry point is
        generate(). sample() exists to satisfy the base class and is a plain uniform
        draw, so a caller that hands it candidates gets no hidden behaviour.
        """
        if n_samples > len(candidates):
            raise ValueError("n_samples cannot exceed number of candidates")
        return self._rng.choice(len(candidates), size=n_samples, replace=False).tolist()

    def generate(
        self, n_sequences: int, task: str = "k562", seq_len: int | None = None
    ) -> tuple[list[str], pd.DataFrame]:
        """Draw ``n_sequences`` orthologous windows."""
        L = seq_len or (200 if task == "k562" else 150)
        seqs: list[str] = []
        meta: list[dict[str, Any]] = []
        with self._open() as f:
            species = [s.decode() if isinstance(s, bytes) else str(s) for s in f["species"][:]]
            self._species = species
            chrom_lens = self._windows(f, L)
            if not chrom_lens:
                raise RuntimeError("no usable chromosomes after exclusions")
            identity = self._rank_species(f, chrom_lens, L)
            tier = self._tier_mask(identity)
            logger.info(
                "zoonomia orthologs: tier=%s -> %d of %d species "
                "(identity to human %.3f-%.3f)",
                self.distance, len(tier), len(species),
                float(identity[tier].min()), float(identity[tier].max()),
            )
            attempts = 0
            max_attempts = 40 * n_sequences
            while len(seqs) < n_sequences and attempts < max_attempts:
                attempts += 1
                chrom, clen = chrom_lens[self._rng.integers(len(chrom_lens))]
                start = int(self._rng.integers(0, clen - L))
                block = f[chrom]["seq"][:, start : start + L]
                order = self._rng.permutation(tier)
                if self.max_species_per_window:
                    order = order[: self.max_species_per_window]
                for si in order:
                    row = block[si]
                    if (row != 0).mean() < self.min_called_frac:
                        continue
                    s = "".join(CODE_TO_BASE[row])
                    if "N" in s:
                        continue
                    seqs.append(s)
                    meta.append(
                        {
                            "species": species[si],
                            "identity_to_human": float(identity[si]),
                            "chrom": chrom,
                            "start": start,
                            "distance_tier": self.distance,
                        }
                    )
                    if len(seqs) >= n_sequences:
                        break
        if len(seqs) < n_sequences:
            logger.warning(
                "only %d of %d requested orthologs met the called-fraction threshold "
                "(%.2f) after %d window draws",
                len(seqs), n_sequences, self.min_called_frac, attempts,
            )
        return seqs, pd.DataFrame(meta)
