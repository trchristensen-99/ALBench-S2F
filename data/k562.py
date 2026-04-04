"""
K562 human cell line MPRA dataset loader.

Dataset from: Gosai et al., Nature 2023
Zenodo: https://zenodo.org/records/10698014

Following benchmark paper preprocessing:
- 200bp genomic sequences (pad shorter sequences with Ns)
- 5 channels: ACGT + reverse complement flag
- hashFrag-based orthogonal train/val/test splits
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base import SequenceDataset
from .hashfrag_splits import HashFragSplitter
from .utils import one_hot_encode

logger = logging.getLogger(__name__)


class K562Dataset(SequenceDataset):
    """
    K562 human MPRA dataset.

    Dataset characteristics:
    - 367,364 regulatory sequences (reference alleles only)
    - 200bp genomic sequences
    - 5 input channels: ACGT + reverse complement flag
    - Expression values (log2 fold change)

    Data splits (following the paper):
    - train: ~320,000 sequences (all hashFrag training data; train+pool combined)
    - val: 36,737 sequences (hashFrag-based validation set)
    - test: 36,737 sequences (hashFrag-based test set)

    Note: the historical 100K "train" / 220K "pool" subdivision is no longer
    exposed.  Existing cache files with separate train_indices.npy and
    pool_indices.npy are merged transparently on load.
    """

    SEQUENCE_LENGTH = 200  # Target sequence length (as per paper)
    NUM_CHANNELS = 5  # ACGT + reverse complement flag

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform: Optional[any] = None,
        target_transform: Optional[any] = None,
        subset_size: Optional[int] = None,
        use_hashfrag: bool = True,
        hashfrag_threshold: int = 60,
        hashfrag_cache_dir: Optional[str] = None,
        use_chromosome_fallback: bool = False,
        label_column: str = "K562_log2FC",
        include_alt_alleles: bool = False,
        duplication_cutoff: Optional[float] = None,
    ):
        """
        Initialize K562 dataset.

        Args:
            data_path: Path to data directory containing the main data file
            split: One of 'train', 'val', 'test'
            transform: Optional transform to apply to sequences
            target_transform: Optional transform to apply to labels
            subset_size: Optional number of samples to use (for downsampling experiments)
            use_hashfrag: If True, use HashFrag for orthogonal splits (default: True)
            hashfrag_threshold: Smith-Waterman score threshold for homology (default: 60)
            hashfrag_cache_dir: Directory to cache HashFrag splits (default: {data_path}/hashfrag_splits)
            use_chromosome_fallback: If True, use chromosome-based splits as fallback (default: False)
            include_alt_alleles: If True, include both ref and alt alleles (default: False).
                The original Malinois paper trained on all 798K oligos (ref+alt).
            duplication_cutoff: If set, duplicate training sequences whose label >= cutoff.
                Follows the boda2 approach to balance the dataset toward high-activity CREs.
                Only applied when split=="train". Default 0.5 is typical.
        """
        self.subset_size = subset_size
        self.use_hashfrag = use_hashfrag
        self.hashfrag_threshold = hashfrag_threshold
        self.hashfrag_cache_dir = hashfrag_cache_dir
        self.use_chromosome_fallback = use_chromosome_fallback
        self.label_column = label_column
        self.include_alt_alleles = include_alt_alleles
        self.duplication_cutoff = duplication_cutoff
        super().__init__(data_path, split, transform, target_transform)

    def load_data(self) -> None:
        """
        Load K562 MPRA data with hashFrag-based train/pool/val/test splits.

        Following the paper:
        - Use hashFrag to generate orthogonal splits (80:10:10)
        - From 80% training data: 100K train + 193K pool
        - 10% validation (36,737 sequences)
        - 10% test (36,737 sequences)
        """
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

        # Get or create splits using HashFrag (or fallback to chromosome-based)
        if self.use_hashfrag:
            try:
                splits = self._get_or_create_hashfrag_splits(all_sequences, all_labels, data_dir)
            except RuntimeError as e:
                if self.use_chromosome_fallback:
                    logger.warning(f"HashFrag failed: {e}")
                    logger.warning("Falling back to chromosome-based splits")
                    splits = self._create_chromosome_splits(all_sequences, all_labels, all_ids)
                else:
                    raise
        else:
            if self.use_chromosome_fallback:
                splits = self._create_chromosome_splits(all_sequences, all_labels, all_ids)
            else:
                raise ValueError(
                    "use_hashfrag=False requires use_chromosome_fallback=True. "
                    "Please enable fallback if you want chromosome-based splits."
                )

        # Extract requested split
        self.sequences, self.labels, self.indices = splits[self.split]

        # Standardize sequences to 200bp
        self.sequences = self._standardize_to_200bp(self.sequences)

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
        # Use random sampling without replacement (seedless unless caller sets global seed)
        if self.subset_size is not None and self.subset_size < len(self.sequences):
            rng = np.random.default_rng()
            indices = rng.choice(len(self.sequences), size=self.subset_size, replace=False)
            self.sequences = self.sequences[indices]
            self.labels = self.labels[indices]
            logger.info(
                f"Downsampled to {self.subset_size:,} sequences (random sampling, no replacement)"
            )

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

    def _get_or_create_hashfrag_splits(
        self, all_sequences: np.ndarray, all_labels: np.ndarray, data_dir: Path
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get cached HashFrag splits or create new ones.

        Returns:
            Dict with 'train', 'val', 'test' keys.
            Each value: (sequences, labels, indices)
        """
        # Set cache directory
        # When no explicit override, derive the subdirectory name from filter
        # settings so that differently-filtered datasets get separate caches.
        # Quality filters (stderr < 1.0, outlier removal, project filter) are
        # always active in _load_and_filter_data(), so the old unfiltered
        # "hashfrag_splits/" cache is never used for new runs.
        if self.hashfrag_cache_dir:
            cache_dir = Path(self.hashfrag_cache_dir)
        elif self.include_alt_alleles:
            cache_dir = data_dir / "hashfrag_splits_qf_alt"
        else:
            cache_dir = data_dir / "hashfrag_splits_qf"

        required_cache_files = {
            "train": cache_dir / "train_indices.npy",
            "val": cache_dir / "val_indices.npy",
            "test": cache_dir / "test_indices.npy",
        }
        # Legacy: old caches have a separate pool_indices.npy that gets merged into train
        legacy_pool_file = cache_dir / "pool_indices.npy"

        # Check if cache exists
        if all(f.exists() for f in required_cache_files.values()):
            logger.info(f"Loading cached HashFrag splits from {cache_dir}")
            return self._load_cached_splits(
                all_sequences, all_labels, required_cache_files, legacy_pool_file
            )

        # Create new splits
        logger.info("=" * 70)
        logger.info("Creating new HashFrag splits")
        logger.info("This will take several hours for the full K562 dataset...")
        logger.info("=" * 70)

        return self._create_new_hashfrag_splits(all_sequences, all_labels, cache_dir)

    def _load_cached_splits(
        self,
        all_sequences: np.ndarray,
        all_labels: np.ndarray,
        cache_files: Dict[str, Path],
        legacy_pool_file: Optional[Path] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Load splits from cached indices.

        If a legacy pool_indices.npy exists alongside train_indices.npy, the two
        are merged so that 'train' contains all hashFrag training sequences (~320K).
        """
        splits = {}
        for split_name, cache_file in cache_files.items():
            indices = np.load(cache_file)
            splits[split_name] = (all_sequences[indices], all_labels[indices], indices)
            logger.info(f"  {split_name}: {len(indices):,} sequences")

        # Merge legacy pool into train (backward-compat with old cache layout)
        if legacy_pool_file is not None and legacy_pool_file.exists():
            pool_idx = np.load(legacy_pool_file)
            logger.info(f"  pool (legacy): {len(pool_idx):,} sequences — merging into train")
            train_idx = splits["train"][2]
            combined_idx = np.concatenate([train_idx, pool_idx])
            splits["train"] = (
                np.concatenate([splits["train"][0], all_sequences[pool_idx]]),
                np.concatenate([splits["train"][1], all_labels[pool_idx]]),
                combined_idx,
            )
            logger.info(f"  train (merged): {len(combined_idx):,} sequences")

        return splits

    def _create_new_hashfrag_splits(
        self, all_sequences: np.ndarray, all_labels: np.ndarray, cache_dir: Path
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Create new HashFrag splits and cache them."""
        # Create splitter
        splitter = HashFragSplitter(
            work_dir=str(cache_dir / "hashfrag_work"), threshold=self.hashfrag_threshold
        )

        # Create 80/10/10 splits — 'train' gets the full 80%, no pool subdivision
        raw_splits = splitter.create_splits_from_dataset(
            sequences=all_sequences,
            labels=all_labels,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            skip_revcomp=False,
        )

        splits = {
            "train": raw_splits["train"],  # full 80% (~320K)
            "val": raw_splits["val"],
            "test": raw_splits["test"],
        }

        # Cache indices
        cache_dir.mkdir(parents=True, exist_ok=True)
        for split_name, (_, _, indices) in splits.items():
            cache_file = cache_dir / f"{split_name}_indices.npy"
            np.save(cache_file, indices)
            logger.info(f"Cached {split_name}: {len(indices):,} sequences")

        logger.info("✓ HashFrag splits created and cached!")
        return splits

    def _create_chromosome_splits(
        self, all_sequences: np.ndarray, all_labels: np.ndarray, all_ids: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Create chromosome-based splits matching the Malinois paper.

        Test: chr7, chr13
        Val: chr19, chr21, chrX
        Train+Pool: remaining chromosomes
        """
        logger.info("Creating chromosome-based splits matching the Malinois paper allocation.")

        # Extract chromosome from the IDs (format: chr:pos:ref:alt:type:wc)
        # IDs may use "chr7" or bare "7" format — normalize to bare numbers
        raw_chrs = np.array([seq_id.split(":")[0] for seq_id in all_ids])
        chrs = np.array([c.replace("chr", "") for c in raw_chrs])

        val_chrs = {"19", "21", "X"}
        test_chrs = {"7", "13"}

        val_mask = np.isin(chrs, list(val_chrs))
        test_mask = np.isin(chrs, list(test_chrs))
        train_pool_mask = ~(val_mask | test_mask)

        train_pool_indices = np.where(train_pool_mask)[0]
        val_indices = np.where(val_mask)[0]
        test_indices = np.where(test_mask)[0]

        splits = {
            "train": (
                all_sequences[train_pool_indices],
                all_labels[train_pool_indices],
                train_pool_indices,
            ),
            "val": (all_sequences[val_indices], all_labels[val_indices], val_indices),
            "test": (all_sequences[test_indices], all_labels[test_indices], test_indices),
        }

        logger.info(f"Generated test  {len(test_indices):,} seqs (chr7, chr13)")
        logger.info(f"Generated val   {len(val_indices):,} seqs (chr19, chr21, chrX)")
        logger.info(f"Generated train {len(train_pool_indices):,} seqs (all non-val/test)")

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
