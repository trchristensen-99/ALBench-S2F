"""ENCODE accessibility reservoir sampler (cell-type shared vs specific).

Samples 200bp windows from ENCODE DNase/ATAC peaks in K562 and HepG2, using the
staged three-way partition in ``data/encode_accessibility/``:

    shared_open_both   31,475 regions   open in BOTH cell types
    k562_only          54,250 regions   open in K562 only
    hepg2_only        113,159 regions   open in HepG2 only

Those partitions are the point of this reservoir. The design asks whether a
training corpus should cover sequence that is regulatory in both cell types or
sequence that discriminates between them, and that question needs the arms
separated rather than pooled -- pooling reintroduces exactly the confound.

A DEPTH CAVEAT, stated up front because it is easy to over-read these counts.
HepG2 has 2.1x as many cell-type-specific peaks as K562. That ratio is not
evidence that HepG2 has a larger open genome; peak counts scale with sequencing
depth and with the peak caller's threshold, and the two experiments are different
accessions. So an arm sized in proportion to peak count would be comparing
sequencing depth, not biology. ``match_count=True`` subsamples the larger
partition to the smaller one's region count, which removes the count asymmetry.
It does NOT equalise the underlying signal distributions -- doing that properly
needs the per-peak signal from the narrowPeak files joined back onto the
partition intervals, which is a separate piece of work.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from albench.reservoir.base import ReservoirSampler

logger = logging.getLogger(__name__)

_REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
_DEFAULT_PEAK_DIR = _REPO / "data/encode_accessibility"

# Candidate hg38 references, first readable one wins. Override with HG38_FASTA.
_HG38_CANDIDATES = (
    "/grid/koo/home/dalin/ref/hg38.fa",
    "/grid/vakoc/home/toobian/Koo_Collab/hg38_genome.fa",
    "/grid/ngs/data/Elzar_Oxford/McCombie_Lab/hg38.fa",
)

# Held out everywhere else in the project: chr7+chr13 are the test chromosomes and
# chr19/21/X are the validation chromosomes. A training reservoir that sampled
# these would leak, and accessibility peaks are genomic, so the risk is real.
TEST_CHROMS = ("chr7", "chr13")
VAL_CHROMS = ("chr19", "chr21", "chrX")

PARTITIONS = {
    "shared_open_both": "shared_open_both.bed",
    "k562_only": "k562_only.bed",
    "hepg2_only": "hepg2_only.bed",
    "k562_all": "K562_peaks_merged.bed",
    "hepg2_all": "HepG2_peaks_merged.bed",
}


def _find_hg38() -> str:
    env = os.environ.get("HG38_FASTA")
    if env:
        if not Path(env).exists():
            raise FileNotFoundError(f"HG38_FASTA points at a missing file: {env}")
        return env
    for cand in _HG38_CANDIDATES:
        if os.access(cand, os.R_OK):
            return cand
    raise FileNotFoundError(
        "No readable hg38 FASTA found. Set the HG38_FASTA env var. Tried: "
        + ", ".join(_HG38_CANDIDATES)
    )


def _gc(seq: str) -> float:
    s = seq.upper()
    n = s.count("A") + s.count("C") + s.count("G") + s.count("T")
    return (s.count("G") + s.count("C")) / max(n, 1)


class EncodeAccessibilitySampler(ReservoirSampler):
    """Sample fixed-length windows from ENCODE accessibility peaks.

    Windows are centred on the peak midpoint and clipped to the chromosome. Peaks
    narrower than ``seq_len`` are still used -- the window simply extends into
    flanking sequence, which is what an MPRA oligo tiling that peak would contain
    anyway. Windows whose sequence contains N are dropped.
    """

    def __init__(
        self,
        seed: int | None = None,
        partition: str = "shared_open_both",
        seq_len: int = 200,
        peak_dir: str | None = None,
        fasta: str | None = None,
        exclude_chroms: tuple[str, ...] = TEST_CHROMS + VAL_CHROMS,
        primary_chroms_only: bool = True,
        match_count: int | None = None,
        min_peak_width: int = 0,
    ) -> None:
        """Initialize sampler.

        Args:
            seed: Random seed.
            partition: One of ``shared_open_both``, ``k562_only``, ``hepg2_only``,
                ``k562_all``, ``hepg2_all``.
            seq_len: Output window length.
            peak_dir: Directory holding the staged BED files.
            fasta: hg38 FASTA path (else ``HG38_FASTA``, else autodetect).
            exclude_chroms: Chromosomes to drop. Defaults to the project's test
                (chr7, chr13) AND validation (chr19, chr21, chrX) chromosomes,
                because this reservoir draws real genomic sequence.
            primary_chroms_only: Drop scaffolds, alts and chrM.
            match_count: Subsample the peak set to this many regions before
                drawing, so two cell types can be compared at equal region count
                rather than at equal sequencing depth. See the module docstring.
            min_peak_width: Drop peaks narrower than this before sampling.
        """
        if partition not in PARTITIONS:
            raise ValueError(f"partition must be one of {sorted(PARTITIONS)}, got {partition!r}")
        self._rng = np.random.default_rng(seed)
        self.partition = partition
        self.seq_len = seq_len
        self.peak_dir = Path(peak_dir) if peak_dir else _DEFAULT_PEAK_DIR
        self.fasta = fasta
        self.exclude_chroms = tuple(exclude_chroms)
        self.primary_chroms_only = primary_chroms_only
        self.match_count = match_count
        self.min_peak_width = min_peak_width
        self._peaks: pd.DataFrame | None = None

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

    def _load_peaks(self) -> pd.DataFrame:
        if self._peaks is not None:
            return self._peaks
        path = self.peak_dir / PARTITIONS[self.partition]
        if not path.exists():
            raise FileNotFoundError(f"Peak file not found: {path}")
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            usecols=[0, 1, 2],
            names=["chrom", "start", "end"],
            comment="#",
            dtype={0: str, 1: np.int64, 2: np.int64},
        )
        n0 = len(df)

        if self.primary_chroms_only:
            primary = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
            df = df[df.chrom.isin(primary)]
        if self.exclude_chroms:
            df = df[~df.chrom.isin(self.exclude_chroms)]
        if self.min_peak_width > 0:
            df = df[(df.end - df.start) >= self.min_peak_width]
        df = df.reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"No peaks left for partition={self.partition} after filtering "
                f"(excluded {self.exclude_chroms})."
            )

        if self.match_count is not None and self.match_count < len(df):
            keep = self._rng.choice(len(df), size=self.match_count, replace=False)
            df = df.iloc[np.sort(keep)].reset_index(drop=True)

        widths = (df.end - df.start).values
        logger.info(
            f"ENCODE[{self.partition}]: {len(df):,} peaks of {n0:,} after filters "
            f"(excluded {list(self.exclude_chroms)}), width median="
            f"{int(np.median(widths))} q10={int(np.quantile(widths, 0.1))} "
            f"q90={int(np.quantile(widths, 0.9))}"
        )
        self._peaks = df
        return df

    def generate(
        self,
        n_sequences: int,
        task: str = "k562",
        replace: bool | None = None,
    ) -> tuple[list[str], pd.DataFrame]:
        """Draw ``n_sequences`` windows from the selected peak partition.

        Args:
            n_sequences: Number of windows to return.
            task: Present for interface compatibility; the partition already
                determines the cell type.
            replace: Sample peaks with replacement. Defaults to False, and to True
                only if ``n_sequences`` exceeds the number of available peaks (in
                which case distinct windows still differ, since the offset within
                the peak is re-drawn).

        Returns:
            Tuple of (sequences, metadata_df).
        """
        from pyfaidx import Fasta

        peaks = self._load_peaks()
        fasta_path = self.fasta or _find_hg38()
        genome = Fasta(fasta_path, sequence_always_upper=True, as_raw=True)

        if replace is None:
            replace = n_sequences > len(peaks)
        if replace and n_sequences > len(peaks):
            logger.info(
                f"  n_sequences={n_sequences:,} exceeds {len(peaks):,} available peaks; "
                f"sampling with replacement (window offset is re-drawn per draw, so "
                f"repeated peaks give different windows)."
            )

        # Chromosome lengths are looked up per candidate window, so resolve them once
        # rather than indexing the FASTA object inside the draw loop.
        chrom_len = {c: len(genome[c]) for c in peaks.chrom.unique()}

        half = self.seq_len // 2
        seqs: list[str] = []
        rows: list[tuple] = []
        n_dropped_n = 0
        n_dropped_edge = 0
        # Over-draw so N-containing and chromosome-edge windows can be dropped
        # without falling short of n_sequences.
        attempts = 0
        max_attempts = 10
        while len(seqs) < n_sequences and attempts < max_attempts:
            attempts += 1
            need = n_sequences - len(seqs)
            use_replace = replace or need > len(peaks)
            # The over-draw itself can exceed the peak count even when `need` does
            # not, so the draw size -- not just `need` -- has to be capped when
            # drawing without replacement.
            size = min(int(need * 1.3) + 16, 10_000_000)
            if not use_replace:
                size = min(size, len(peaks))
            idx = self._rng.choice(len(peaks), size=size, replace=use_replace)
            for i in idx:
                if len(seqs) >= n_sequences:
                    break
                chrom = peaks.chrom.iat[i]
                start, end = int(peaks.start.iat[i]), int(peaks.end.iat[i])
                mid = (start + end) // 2
                w_start = mid - half
                w_end = w_start + self.seq_len
                if w_start < 0 or w_end > chrom_len[chrom]:
                    n_dropped_edge += 1
                    continue
                seq = str(genome[chrom][w_start:w_end])
                if len(seq) != self.seq_len or "N" in seq:
                    n_dropped_n += 1
                    continue
                seqs.append(seq)
                rows.append((chrom, w_start, w_end, start, end, end - start, _gc(seq)))

        if len(seqs) < n_sequences:
            raise RuntimeError(
                f"Only produced {len(seqs):,} of {n_sequences:,} windows after "
                f"{max_attempts} passes ({n_dropped_n:,} N-containing, "
                f"{n_dropped_edge:,} at chromosome edges). Widen the partition or "
                f"lower n_sequences."
            )

        arr = list(zip(*rows))
        meta = pd.DataFrame(
            {
                "seq_idx": np.arange(len(seqs), dtype=np.int64),
                "method": f"encode_accessibility_{self.partition}",
                "source": "genomic_peak",
                "chrom": list(arr[0]),
                "window_start": np.array(arr[1], dtype=np.int64),
                "window_end": np.array(arr[2], dtype=np.int64),
                "peak_start": np.array(arr[3], dtype=np.int64),
                "peak_end": np.array(arr[4], dtype=np.int64),
                "peak_width": np.array(arr[5], dtype=np.int32),
                "gc_content": np.array(arr[6], dtype=np.float32),
                "partition": self.partition,
            }
        )
        logger.info(
            f"ENCODE[{self.partition}]: {len(seqs):,} windows of {self.seq_len}bp, "
            f"mean GC={meta.gc_content.mean():.3f}, "
            f"{meta.chrom.nunique()} chromosomes, "
            f"dropped {n_dropped_n:,} N / {n_dropped_edge:,} edge"
        )
        return seqs, meta
