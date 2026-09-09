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
        vocab_cluster_at: float | None = 0.90,
        vocab_trim_ic: float = 0.0,
        vocab_max_len: int | None = None,
        vocab_size: int | None = None,
        vocab_meme: str | None = None,
        vocab_expressed: set[str] | None = None,
        plant_mode: str = "pwm_sample",
        instance_pool: int = 256,
    ) -> None:
        """Initialize sampler.

        Args:
            seed: Random seed.
            min_motifs: Minimum motifs planted per sequence.
            max_motifs: Maximum motifs planted per sequence (inclusive).
            motif_set: ``"auto"``/``"k562"``/``"yeast"`` use the legacy hardcoded
                consensus strings. ``"jaspar"`` builds a real vocabulary from JASPAR
                PFMs via :mod:`albench.motifs.vocabulary`, which is what the
                vocabulary-source and filter arms vary.
            include_rc_variants: Also plant reverse-complement orientations.
            preserve_native_motifs: Keep motif matches already present in the
                background rather than overwriting them.
            vocab_cluster_at: JASPAR only -- collapse near-duplicate PFMs above this
                correlation. A tested factor: 0.90 gives ~242 motifs, 0.80 gives ~119.
            vocab_trim_ic: JASPAR only -- trim terminal positions below this
                per-position IC (bits). A tested factor.
            vocab_max_len: JASPAR only -- drop motifs longer than this after
                trimming. Bounds how many sites fit on one oligo.
            vocab_size: JASPAR only -- cap vocabulary size, most-informative first.
                This is the broad-coverage vs narrow-syntax knob.
            vocab_meme: Override the MEME PFM path (else env ``MOTIF_MEME_PATH``,
                else the JASPAR2022 CORE default).
            vocab_expressed: Optional TF gene symbols to keep, for the cell-type
                vocabulary arms.
            plant_mode: ``"pwm_sample"`` draws each instance from the PWM, so planted
                sites carry the variation real sites have. ``"consensus"`` stamps the
                single consensus string -- the memorisation control, since a model can
                learn a fixed string without learning binding preference.
            instance_pool: Instances pre-drawn per motif under ``pwm_sample``.
                Reservoirs are generated at the 1M+ scale, so instances are drawn from
                this pool rather than freshly per plant.
        """
        if plant_mode not in ("pwm_sample", "consensus"):
            raise ValueError(f"plant_mode must be 'pwm_sample' or 'consensus', got {plant_mode!r}")
        self._rng = np.random.default_rng(seed)
        self.min_motifs = min_motifs
        self.max_motifs = max_motifs
        self.motif_set = motif_set
        self.include_rc_variants = include_rc_variants
        self.preserve_native_motifs = preserve_native_motifs
        self.vocab_cluster_at = vocab_cluster_at
        self.vocab_trim_ic = vocab_trim_ic
        self.vocab_max_len = vocab_max_len
        self.vocab_size = vocab_size
        self.vocab_meme = vocab_meme
        self.vocab_expressed = vocab_expressed
        self.plant_mode = plant_mode
        self.instance_pool = instance_pool
        self._bg_seqs: np.ndarray | None = None  # lazy
        self._vocab: list = []  # lazy: list[Motif]
        self._inst: dict[str, list[str]] = {}  # motif name -> pre-drawn instances

    def _build_vocab(self) -> list:
        """Load and filter the JASPAR vocabulary, then pre-draw planting instances."""
        if self._vocab:
            return self._vocab
        from albench.motifs import vocabulary as V

        meme = self.vocab_meme or os.environ.get("MOTIF_MEME_PATH") or V.MEME_DEFAULT
        if not Path(meme).exists():
            raise FileNotFoundError(
                f"MEME PFM file not found: {meme}. Set vocab_meme= or the "
                f"MOTIF_MEME_PATH env var to a JASPAR .meme file."
            )
        motifs = V.build(
            meme=meme,
            human_only=True,
            cluster_at=self.vocab_cluster_at,
            max_motifs=self.vocab_size,
            expressed=self.vocab_expressed,
            trim_ic=self.vocab_trim_ic,
            max_len=self.vocab_max_len,
        )
        if not motifs:
            raise ValueError(
                "Vocabulary is empty after filtering; loosen vocab_trim_ic / "
                "vocab_max_len / vocab_cluster_at."
            )
        if self.include_rc_variants:
            motifs = motifs + [m.revcomp() for m in motifs]

        # Pre-draw instances. Under pwm_sample every plant is a different string, so
        # the model cannot memorise one literal; under consensus every plant is the
        # same string, which is exactly the control we want to compare against.
        for m in motifs:
            if self.plant_mode == "consensus":
                self._inst[m.name] = [m.consensus]
            else:
                self._inst[m.name] = m.sample(self._rng, self.instance_pool)

        self._vocab = motifs
        lens = np.array([m.length for m in motifs])
        logger.info(
            f"MotifPlantedV2 vocabulary: {len(motifs)} motifs "
            f"({len(motifs) // 2 if self.include_rc_variants else len(motifs)} unique + rc), "
            f"length min={lens.min()} median={int(np.median(lens))} max={lens.max()}, "
            f"cluster_at={self.vocab_cluster_at}, trim_ic={self.vocab_trim_ic}, "
            f"max_len={self.vocab_max_len}, plant_mode={self.plant_mode}"
        )
        return self._vocab

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
        use_vocab = self.motif_set == "jaspar"
        if use_vocab:
            vocab = self._build_vocab()
            # Native-motif detection scans for literal strings; with PWM-sampled
            # instances the consensus is the right proxy for "a site is already here".
            motifs = [m.consensus for m in vocab]
            motif_names = [m.name for m in vocab]
        else:
            motifs = self._get_motifs(task)
            motif_names = list(motifs)
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
            chosen_idx = self._rng.integers(0, len(motifs), size=n_plant)
            planted_log = []
            for mi in chosen_idx:
                mi = int(mi)
                if use_vocab:
                    # A fresh draw per plant, so the same motif appears as different
                    # strings across the reservoir.
                    pool = self._inst[motif_names[mi]]
                    motif = pool[int(self._rng.integers(0, len(pool)))]
                    label = motif_names[mi]
                else:
                    motif = motifs[mi]
                    label = motif
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
                        planted_log.append(label)
                        break

            sequences.append("".join(core))
            n_planted_list.append(len(planted_log))
            n_native_kept.append(native)
            planted_motifs_list.append(",".join(planted_log) if planted_log else "none")

        meta = pd.DataFrame(
            {
                "seq_idx": np.arange(n_sequences, dtype=np.int64),
                "method": (
                    "motif_planted_v2_jaspar" if use_vocab else "motif_planted_v2_gc_matched"
                ),
                "source": "generated",
                "vocab_size": len(motifs),
                "plant_mode": self.plant_mode if use_vocab else "consensus",
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
    """Phylogenetic-variation reservoir driven by real Zoonomia per-position rates.

    ``data/zoonomia/per_position_rates.npz`` holds, for 199,373 cCRE regions of
    200bp, the per-position substitution rate across 241 mammals together with the
    matching sequence, phyloP score and cCRE class. chr7 and chr13 are already
    excluded upstream, so the test chromosomes do not leak in.

    Using those rates rather than one flat rate is the point of this reservoir: a
    flat rate mutates a conserved TF binding site as readily as unconstrained
    flanking sequence, which is precisely the signal phylogenetic variation is
    supposed to carry. Rates must be paired with the region they were measured on,
    so sequences default to the regions in the rate file.

    ``rate_mode`` gives three arms that separate the amount of mutation from its
    distribution:

    ``flat``
        One rate everywhere (``mut_rate``). The control, and the previous behaviour.
    ``per_position_matched``
        Real per-position rates rescaled so the mean load equals ``mut_rate``.
        Isolates *where* mutations fall from *how many* there are -- the comparison
        that actually tests whether conservation-awareness helps.
    ``per_position``
        Real rates as measured (mean ~0.22 substitutions per position across 241
        mammals), i.e. the true divergence load.
    """

    DEFAULT_RATE = 0.02
    DEFAULT_RATES_PATH = _REPO / "data/zoonomia/per_position_rates.npz"

    def __init__(
        self,
        seed: int | None = None,
        mut_rate: float | None = None,
        rate_mode: str = "per_position_matched",
        rates_path: str | None = None,
        ti_tv: float = 2.0,
        exclude_chroms: tuple[str, ...] = (),
        ccre_classes: tuple[str, ...] = (),
    ) -> None:
        """Initialize sampler.

        Args:
            seed: Random seed.
            mut_rate: Mean per-position substitution rate for ``flat`` and
                ``per_position_matched``. Ignored by ``per_position``.
            rate_mode: ``"flat"``, ``"per_position_matched"`` or ``"per_position"``.
            rates_path: Override the rate file (else env ``ZOONOMIA_RATES``, else
                ``data/zoonomia/per_position_rates.npz``).
            ti_tv: Transition/transversion ratio. Mammalian substitution is
                transition-biased at roughly 2:1, so drawing uniformly among the
                three alternative bases produces the wrong mutation spectrum.
                Set to 1.0 for a uniform spectrum.
            exclude_chroms: Additional chromosomes to drop (chr7/chr13 are already
                excluded when the rate file was built).
            ccre_classes: Restrict to these cCRE classes (e.g. ``("pELS", "dELS")``).
                Empty keeps all.
        """
        if rate_mode not in ("flat", "per_position", "per_position_matched"):
            raise ValueError(
                f"rate_mode must be flat/per_position/per_position_matched, got {rate_mode!r}"
            )
        if ti_tv <= 0:
            raise ValueError(f"ti_tv must be > 0, got {ti_tv}")
        self._rng = np.random.default_rng(seed)
        self.mut_rate = mut_rate or self.DEFAULT_RATE
        self.rate_mode = rate_mode
        self.rates_path = rates_path
        self.ti_tv = ti_tv
        self.exclude_chroms = tuple(exclude_chroms)
        self.ccre_classes = tuple(ccre_classes)
        self._cache: dict | None = None

    def sample(self, candidates, n_samples, metadata=None):
        if n_samples > len(candidates):
            raise ValueError("n_samples cannot exceed number of candidates")
        return self._rng.choice(len(candidates), size=n_samples, replace=False).tolist()

    def _load_rates(self) -> dict:
        """Load Zoonomia rates and their matching sequences."""
        if self._cache is not None:
            return self._cache
        path = Path(self.rates_path or os.environ.get("ZOONOMIA_RATES") or self.DEFAULT_RATES_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"Zoonomia rate file not found: {path}. Set rates_path= or the "
                f"ZOONOMIA_RATES env var, or use rate_mode='flat'."
            )
        z = np.load(path, allow_pickle=True)
        seqs = np.array([str(s) for s in z["sequences"]])
        rates = np.asarray(z["subst_rate"], dtype=np.float32)
        chrom = np.asarray(z["chrom"]).astype(str)
        ccre = np.asarray(z["ccre_class"]).astype(str)
        phylop = np.asarray(z["phylop"], dtype=np.float32)

        keep = np.ones(len(seqs), dtype=bool)
        if self.exclude_chroms:
            keep &= ~np.isin(chrom, np.array(self.exclude_chroms))
        if self.ccre_classes:
            keep &= np.isin(ccre, np.array(self.ccre_classes))
        if not keep.any():
            raise ValueError("No Zoonomia regions survive exclude_chroms/ccre_classes filters.")

        seqs, rates, chrom, ccre, phylop = (
            seqs[keep],
            rates[keep],
            chrom[keep],
            ccre[keep],
            phylop[keep],
        )
        # Rates are per-position probabilities; guard against NaN from unaligned bases.
        rates = np.nan_to_num(rates, nan=0.0)
        excluded = np.asarray(z["excluded_chroms"]).astype(str).tolist()
        logger.info(
            f"Zoonomia: {len(seqs):,} regions x {rates.shape[1]}bp, "
            f"{int(z['n_species'])} species, mean subst_rate={rates.mean():.4f}, "
            f"chroms excluded upstream={excluded}"
            + (f" + {list(self.exclude_chroms)}" if self.exclude_chroms else "")
        )
        self._cache = dict(seqs=seqs, rates=rates, chrom=chrom, ccre=ccre, phylop=phylop)
        return self._cache

    def _substitute(self, seq_bytes: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply transition-biased substitutions at masked positions.

        ``seq_bytes`` is (n, L) uint8 ASCII; ``mask`` is (n, L) bool. Positions that
        are not A/C/G/T are left alone.
        """
        # index into ACGT; -1 for anything else (N etc.)
        code = np.full(seq_bytes.shape, -1, dtype=np.int8)
        for i, b in enumerate(b"ACGT"):
            code[seq_bytes == b] = i
        mask = mask & (code >= 0)

        # Transition partner: A<->G (0<->2), C<->T (1<->3)
        transition = np.array([2, 3, 0, 1], dtype=np.int8)
        # The two transversion partners for each base
        transversion = np.array([[1, 3], [0, 2], [1, 3], [0, 2]], dtype=np.int8)

        idx = np.where(mask)
        n_mut = idx[0].size
        if n_mut == 0:
            return seq_bytes
        base = code[idx]
        # Ti/Tv is the aggregate count ratio (~2 in mammals), so with one transition
        # partner and two transversion partners we need P(ti)/(1-P(ti)) = ti_tv,
        # i.e. P(ti) = R/(R+1). Using R/(R+2) -- the per-substitution-type
        # convention -- halves the realised ratio.
        p_ti = self.ti_tv / (self.ti_tv + 1.0)
        is_ti = self._rng.random(n_mut) < p_ti
        which_tv = self._rng.integers(0, 2, size=n_mut)
        new = np.where(is_ti, transition[base], transversion[base, which_tv])

        out = seq_bytes.copy()
        out[idx] = np.frombuffer(b"ACGT", dtype=np.uint8)[new]
        return out

    def generate(
        self, n_sequences: int, base_sequences=None, task: str = "k562"
    ) -> tuple[list[str], pd.DataFrame]:
        seq_len = 200
        use_real = self.rate_mode != "flat"

        if use_real:
            if base_sequences is not None and len(base_sequences) > 0:
                raise ValueError(
                    "rate_mode='%s' pairs each per-position rate vector with the region "
                    "it was measured on, so base_sequences cannot be supplied. Use "
                    "rate_mode='flat' to mutate arbitrary sequences." % self.rate_mode
                )
            cache = self._load_rates()
            base_arr = cache["seqs"]
            rate_arr = cache["rates"]
            chrom_arr = cache["chrom"]
            ccre_arr = cache["ccre"]
            phylop_arr = cache["phylop"]
        else:
            if base_sequences is None or len(base_sequences) == 0:
                z = np.load(_bg_cache_path(), allow_pickle=True)
                base_sequences = [str(s) for s in z["sequences"]]
            base_arr = np.array([str(s) for s in base_sequences])
            rate_arr = None
            chrom_arr = ccre_arr = phylop_arr = None

        idx = self._rng.choice(len(base_arr), size=n_sequences, replace=True)

        # Encode chosen backgrounds as an (n, L) uint8 ASCII matrix.
        padded = np.array(
            [str(base_arr[i])[:seq_len].ljust(seq_len, "N") for i in idx], dtype=f"U{seq_len}"
        )
        seq_bytes = padded.view(np.uint32).astype(np.uint8).reshape(n_sequences, seq_len)

        if use_real:
            rates = rate_arr[idx]
            if self.rate_mode == "per_position_matched":
                # Preserve the shape of the conservation profile, match the mean load.
                observed = rates.mean()
                if observed <= 0:
                    raise ValueError("Zoonomia rates are all zero; cannot rescale.")
                rates = np.clip(rates * (self.mut_rate / observed), 0.0, 1.0)
            mask = self._rng.random((n_sequences, seq_len)) < rates
        else:
            mask = self._rng.random((n_sequences, seq_len)) < self.mut_rate

        mutated = self._substitute(seq_bytes, mask)
        n_mut_list = (mutated != seq_bytes).sum(axis=1).astype(np.int32)
        sequences = [row.tobytes().decode("ascii") for row in mutated]

        cols = {
            "seq_idx": np.arange(n_sequences, dtype=np.int64),
            "method": f"phylogenetic_zoonomia_{self.rate_mode}",
            "source": "pool_with_zoonomia_mutations",
            "n_mutations": n_mut_list,
            "mut_rate": self.mut_rate if self.rate_mode != "per_position" else np.float32(np.nan),
            "rate_mode": self.rate_mode,
            "ti_tv": self.ti_tv,
        }
        if use_real:
            cols["region_idx"] = idx.astype(np.int64)
            cols["chrom"] = chrom_arr[idx]
            cols["ccre_class"] = ccre_arr[idx]
            cols["mean_phylop"] = phylop_arr[idx].mean(axis=1).astype(np.float32)
        meta = pd.DataFrame(cols)

        logger.info(
            f"PhylogeneticZoonomia[{self.rate_mode}]: {n_sequences:,} seqs, "
            f"mean mutations={n_mut_list.mean():.1f}/{seq_len}bp "
            f"({n_mut_list.mean() / seq_len:.2%} per position), ti_tv={self.ti_tv}"
        )
        return sequences, meta
