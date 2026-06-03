"""
K562 human cell line MPRA dataset loader.

Dataset from: Gosai et al., Nature 2023
Zenodo: https://zenodo.org/records/10698014

Following benchmark paper preprocessing:
- 200bp genomic sequences (pad shorter sequences with Ns)
- 5 channels: ACGT + reverse complement flag
- Chromosome-based train/val/test splits (val={19,21,X}, test={7,13})
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import SequenceDataset
from .utils import one_hot_encode

logger = logging.getLogger(__name__)


class K562Dataset(SequenceDataset):
    """
    K562 human MPRA dataset.

    Dataset characteristics:
    - ~367,364 regulatory sequences (reference alleles only)
    - 200bp genomic sequences
    - 5 input channels: ACGT + reverse complement flag
    - Expression values (log2 fold change)

    Data splits (chromosome-based, matching the Malinois paper):
    - test: chr7, chr13
    - val: chr19, chr21, chrX
    - train: all remaining chromosomes
    """

    SEQUENCE_LENGTH = 200  # Target sequence length (as per paper)
    NUM_CHANNELS = 5  # ACGT + reverse complement flag

    # Canonical MPRA adapter constants from alphagenome_FT_MPRA/oracle.py.
    # When include_adapters=True, sequences are stored as
    # LEFT_ADAPTER + payload + RIGHT_ADAPTER. Shift augmentation uses
    # max_shift = min(len(LEFT_ADAPTER), len(RIGHT_ADAPTER)) so payload
    # bases never cross the input window boundary.
    LEFT_ADAPTER = "AGGACCGGATCAACT"  # 15bp
    RIGHT_ADAPTER = "CATTGCGTGAACCGA"  # 15bp

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform: Optional[any] = None,
        target_transform: Optional[any] = None,
        subset_size: Optional[int] = None,
        label_column: str = "K562_log2FC",
        include_alt_alleles: bool = False,
        duplication_cutoff: Optional[float] = None,
        include_adapters: bool = False,
        val_chrs: Optional[List[str]] = None,
        test_chrs: Optional[List[str]] = None,
    ):
        """
        Initialize K562 dataset.

        Args:
            data_path: Path to data directory containing the main data file
            split: One of 'train', 'val', 'test'
            transform: Optional transform to apply to sequences
            target_transform: Optional transform to apply to labels
            subset_size: Optional number of samples to use (for downsampling experiments)
            label_column: Activity column to use as labels (default K562_log2FC)
            include_alt_alleles: If True, include both ref and alt alleles (default: False).
                The original Malinois paper trained on all 798K oligos (ref+alt).
            duplication_cutoff: If set, duplicate training sequences whose label >= cutoff.
                Follows the boda2 approach to balance the dataset toward high-activity CREs.
                Only applied when split=="train". Default 0.5 is typical.
            include_adapters: If True, prepend ``LEFT_ADAPTER`` and append
                ``RIGHT_ADAPTER`` to every payload, producing sequences of
                length ``SEQUENCE_LENGTH + len(LEFT_ADAPTER) + len(RIGHT_ADAPTER)``
                (= 230bp with the canonical 15bp adapters). Required when
                training with shift augmentation: shift moves a sliding
                window over the adapter-padded sequence so payload bases
                never cross the boundary, and adapter context is exposed
                on whichever side the window slides toward.
            val_chrs: Custom validation chromosome set (default {19,21,X}).
                Used by the chr-fold ensemble pipeline (one fold per val chr).
            test_chrs: Custom test chromosome set (default {7,13}).
        """
        self.subset_size = subset_size
        self.label_column = label_column
        self.include_alt_alleles = include_alt_alleles
        self.duplication_cutoff = duplication_cutoff
        self.include_adapters = include_adapters
        # Custom chromosome split: when set, overrides the default
        # (val={19,21,X}, test={7,13}) used by _create_chromosome_splits.
        # Used by the chr-fold ensemble pipeline: each fold passes a
        # different val_chrs (one chromosome) while keeping test_chrs={7,13}.
        self.val_chrs = [str(c).replace("chr", "") for c in val_chrs] if val_chrs else None
        self.test_chrs = [str(c).replace("chr", "") for c in test_chrs] if test_chrs else None
        super().__init__(data_path, split, transform, target_transform)

    def load_data(self) -> None:
        """Load K562 MPRA data with chromosome-based train/val/test splits."""
        data_dir = Path(self.data_path)

        # The actual filename from the Zenodo download
        file_path = data_dir / "DATA-Table_S2__MPRA_dataset.txt"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Could not find K562 data file at {file_path}. "
                f"Please run: python scripts/download_data.py --dataset k562"
            )

        logger.info(f"Loading K562 {self.split} data from {file_path}")

        # Load and filter data
        all_sequences, all_labels, all_ids = self._load_and_filter_data(file_path)

        # Chromosome-based splits — the only supported protocol
        splits = self._create_chromosome_splits(all_sequences, all_labels, all_ids)

        # Extract requested split
        self.sequences, self.labels, self.indices = splits[self.split]

        # Standardize sequences to 200bp
        self.sequences = self._standardize_to_200bp(self.sequences)

        # Optionally prepend / append the canonical MPRA adapters so shift
        # augmentation has real context to slide into. This must happen
        # AFTER standardization so we always concatenate against a 200bp
        # payload and the resulting length is deterministic.
        if self.include_adapters:
            self.sequences = np.array(
                [self.LEFT_ADAPTER + str(s) + self.RIGHT_ADAPTER for s in self.sequences]
            )

        # Duplicate high-activity sequences (boda2-style balancing)
        if self.duplication_cutoff is not None and self.split == "train":
            high_mask = self.labels >= self.duplication_cutoff
            n_high = int(np.sum(high_mask))
            if n_high > 0:
                self.sequences = np.concatenate([self.sequences, self.sequences[high_mask]])
                self.labels = np.concatenate([self.labels, self.labels[high_mask]])
                logger.info(
                    f"Duplicated {n_high:,} high-activity sequences "
                    f"(label >= {self.duplication_cutoff}), "
                    f"total now {len(self.sequences):,}"
                )

        # Apply subset size if specified (for downsampling experiments)
        if self.subset_size is not None and self.subset_size < len(self.sequences):
            rng = np.random.default_rng()
            indices = rng.choice(len(self.sequences), size=self.subset_size, replace=False)
            self.sequences = self.sequences[indices]
            self.labels = self.labels[indices]
            logger.info(
                f"Downsampled to {self.subset_size:,} sequences (random sampling, no replacement)"
            )

        if self.include_adapters:
            self.sequence_length = (
                self.SEQUENCE_LENGTH + len(self.LEFT_ADAPTER) + len(self.RIGHT_ADAPTER)
            )
        else:
            self.sequence_length = self.SEQUENCE_LENGTH

        logger.info(f"Loaded {len(self.sequences)} sequences for {self.split} split")
        if len(self.labels) > 0:
            logger.info(f"Label range: [{np.min(self.labels):.3f}, {np.max(self.labels):.3f}]")
        else:
            logger.warning(f"Split '{self.split}' is empty — no sequences matched")

    def _load_and_filter_data(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and filter K562 data to reference alleles only.

        Returns:
            Tuple of (sequences, labels, ids) for all filtered data
        """
        # Load data (tab-separated with header)
        try:
            df = pd.read_csv(file_path, sep="\t", dtype={"OL": str})
        except Exception as e:
            raise RuntimeError(f"Error loading K562 data from {file_path}: {e}")

        # Filter alleles based on include_alt_alleles flag
        # Parse ID format: chr:pos:ref:alt:allele_type:wc
        id_parts = df["IDs"].str.split(":", expand=True)
        allele_type = id_parts[4]  # R=reference, A=alternate, empty=CRE/no variant
        ref_col = id_parts[2]
        alt_col = id_parts[3]

        n_before = len(df)
        if self.include_alt_alleles:
            # Keep all alleles (ref + alt + non-variant) — matches Malinois paper (798K oligos)
            is_valid = allele_type.isin(["R", "A"]) | ((ref_col == "NA") & (alt_col == "NA"))
            df = df[is_valid].copy()
        else:
            # Keep reference alleles (R) and non-variant sequences only
            is_reference = allele_type == "R"
            is_non_variant = (ref_col == "NA") & (alt_col == "NA")
            df = df[is_reference | is_non_variant].copy()
        n_after = len(df)

        logger.info(
            f"Filtered to {n_after:,} reference alleles (excluded {n_before - n_after:,} alternate alleles)"
        )

        # Quality filters matching Malinois paper (boda2 preprocessing)
        # 1. Project filter
        if "data_project" in df.columns:
            allowed_projects = ["UKBB", "GTEX", "CRE"]
            n_pre = len(df)
            df = df[df["data_project"].isin(allowed_projects)].reset_index(drop=True)
            if len(df) < n_pre:
                logger.info(f"Project filter: {n_pre:,} -> {len(df):,}")

        # 2. Stderr quality filter (max SE across all cell types < 1.0)
        stderr_cols = [c for c in df.columns if c.endswith("_lfcSE")]
        if stderr_cols:
            n_pre = len(df)
            quality_mask = df[stderr_cols].max(axis=1) < 1.0
            df = df[quality_mask].reset_index(drop=True)
            if len(df) < n_pre:
                logger.info(f"Stderr filter (max < 1.0): {n_pre:,} -> {len(df):,}")

        # 3. Outlier removal (±6σ with +4 upper shift, matching boda2)
        activity_cols = [c for c in df.columns if c.endswith("_log2FC")]
        if activity_cols:
            means = df[activity_cols].mean().to_numpy()
            stds = df[activity_cols].std().to_numpy()
            up_cut = means + stds * 6.0 + 4.0
            down_cut = means - stds * 6.0
            n_pre = len(df)
            b_up = (df[activity_cols] < up_cut).all(axis=1)
            b_down = (df[activity_cols] > down_cut).all(axis=1)
            df = df[b_up & b_down].reset_index(drop=True)
            if len(df) < n_pre:
                logger.info(f"Outlier filter (±6σ): {n_pre:,} -> {len(df):,}")

        # Filter by sequence length (paper uses sequences >= 198bp for ~367K total)
        df["seq_len"] = df["sequence"].str.len()
        n_before_len = len(df)
        df = df[df["seq_len"] >= 198].copy()
        n_after_len = len(df)

        logger.info(
            f"Length filter (>= 198bp): {n_after_len:,} sequences (excluded {n_before_len - n_after_len:,} shorter sequences)"
        )
        df = df.drop(columns=["seq_len"])

        # Extract sequences and labels
        sequences = df["sequence"].values
        labels = df[self.label_column].values.astype(np.float32)
        ids = df["IDs"].values

        return sequences, labels, ids

    def _create_chromosome_splits(
        self, all_sequences: np.ndarray, all_labels: np.ndarray, all_ids: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Create chromosome-based splits matching the Malinois paper.

        Test: chr7, chr13
        Val: chr19, chr21, chrX
        Train: remaining chromosomes
        """
        logger.info("Creating chromosome-based splits matching the Malinois paper allocation.")

        # Extract chromosome from the IDs (format: chr:pos:ref:alt:type:wc)
        # IDs may use "chr7" or bare "7" format — normalize to bare numbers
        raw_chrs = np.array([seq_id.split(":")[0] for seq_id in all_ids])
        chrs = np.array([c.replace("chr", "") for c in raw_chrs])

        # Custom chr split (if specified at construction time) overrides defaults.
        val_chrs = set(self.val_chrs) if self.val_chrs else {"19", "21", "X"}
        test_chrs = set(self.test_chrs) if self.test_chrs else {"7", "13"}
        logger.info(f"  chr split: val={sorted(val_chrs)}  test={sorted(test_chrs)}")

        val_mask = np.isin(chrs, list(val_chrs))
        test_mask = np.isin(chrs, list(test_chrs))
        train_mask = ~(val_mask | test_mask)

        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        test_indices = np.where(test_mask)[0]

        splits = {
            "train": (
                all_sequences[train_indices],
                all_labels[train_indices],
                train_indices,
            ),
            "val": (all_sequences[val_indices], all_labels[val_indices], val_indices),
            "test": (all_sequences[test_indices], all_labels[test_indices], test_indices),
        }

        logger.info(f"Generated test  {len(test_indices):,} seqs (chr {sorted(test_chrs)})")
        logger.info(f"Generated val   {len(val_indices):,} seqs (chr {sorted(val_chrs)})")
        logger.info(f"Generated train {len(train_indices):,} seqs (all non-val/test)")

        return splits

    def _standardize_to_200bp(self, sequences: np.ndarray) -> np.ndarray:
        """
        Standardize sequences to 200bp.

        Sequences shorter than 200bp are padded equally on both ends with Ns.
        Sequences longer than 200bp are truncated (center-aligned).
        """
        processed = []

        for seq in sequences:
            curr_len = len(seq)

            if curr_len < self.SEQUENCE_LENGTH:
                # Pad equally on both ends with Ns
                pad_needed = self.SEQUENCE_LENGTH - curr_len
                left_pad = pad_needed // 2
                right_pad = pad_needed - left_pad
                padded = "N" * left_pad + seq + "N" * right_pad
                processed.append(padded)

            elif curr_len > self.SEQUENCE_LENGTH:
                # Truncate to target length (center-aligned)
                start = (curr_len - self.SEQUENCE_LENGTH) // 2
                processed.append(seq[start : start + self.SEQUENCE_LENGTH])

            else:
                processed.append(seq)

        # Verify all sequences are exactly 200bp
        for i, seq in enumerate(processed):
            if len(seq) != self.SEQUENCE_LENGTH:
                raise ValueError(
                    f"Sequence {i} length mismatch: {len(seq)} != {self.SEQUENCE_LENGTH}"
                )

        return np.array(processed)

    def encode_sequence(self, sequence: str, metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Encode a K562 sequence with 5 channels.

        Args:
            sequence: DNA sequence string (200bp)
            metadata: Optional metadata dict (not used)

        Returns:
            Encoded sequence of shape (5, 200)
            Channels:
            - 0-3: one-hot encoded ACGT
            - 4: reverse complement flag (0 for forward, 1 for reverse)
        """
        # Get one-hot encoding (4 channels)
        encoded = one_hot_encode(sequence, add_singleton_channel=False)  # Shape: (4, 200)

        # Add reverse complement channel (always 0 for forward strand during training)
        rc_channel = np.zeros((1, len(sequence)), dtype=np.float32)

        # Concatenate: (4, 200) + (1, 200) = (5, 200)
        encoded = np.concatenate([encoded, rc_channel], axis=0)

        return encoded

    def get_num_channels(self) -> int:
        """Return number of input channels (5 for K562)."""
        return self.NUM_CHANNELS
