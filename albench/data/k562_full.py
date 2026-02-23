import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base import SequenceDataset as BaseDataset
from .utils import one_hot_encode

logger = logging.getLogger(__name__)


class K562FullDataset(BaseDataset):
    """
    K562 MPRA Dataset exactly matching the full training scheme from the Malinois paper.

    Differences from K562Dataset:
    - Analyzes the full ~800k dataset without dropping sequence variants.
    - Uses exact standard-deviation-based outlier pruning and stderr thresholds (< 1.0).
    - Hardcodes padding sequences dynamically leveraging the MPRA context flanks exactly
      replicating `FastSeqProp` upstream & downstream extensions up to 600bp.
    - Restricts allocations solely to exact chromosome bins (Val: 19,21,X | Test: 7,13).
    """

    SEQUENCE_LENGTH = 600
    NUM_CHANNELS = 5

    # Flanks pulled directly from boda2-main/boda/common/constants.py
    MPRA_UPSTREAM = "ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG"
    MPRA_DOWNSTREAM = "CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCT"

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform: Optional[any] = None,
        target_transform: Optional[any] = None,
        subset_size: Optional[int] = None,
    ):
        self.subset_size = subset_size
        super().__init__(data_path, split, transform, target_transform)

    def load_data(self) -> None:
        data_dir = Path(self.data_path)
        file_path = data_dir / "DATA-Table_S2__MPRA_dataset.txt"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Could not find K562 data file at {file_path}. "
                f"Please run: python scripts/download_data.py --dataset k562"
            )

        logger.info(f"Loading FULL Malinois-style K562 {self.split} data from {file_path}")

        # Load and precisely filter data matching boda2 DataModule
        all_sequences, all_labels, all_ids = self._load_and_filter_data(file_path)

        splits = self._create_chromosome_splits(all_sequences, all_labels, all_ids)

        # Valid split targets in this dataset structure are just train, val, and test.
        if self.split not in splits:
            raise ValueError(
                f"Split {self.split} is not valid for Full Dataset config. Use train, val, or test."
            )

        self.sequences, self.labels, self.indices = splits[self.split]

        if self.subset_size is not None and self.subset_size < len(self.sequences):
            rng = np.random.default_rng()
            indices = rng.choice(len(self.sequences), size=self.subset_size, replace=False)
            self.sequences = self.sequences[indices]
            self.labels = self.labels[indices]
            logger.info(
                f"Downsampled to {self.subset_size:,} sequences (random sampling, no replacement)"
            )

        self.sequence_length = self.SEQUENCE_LENGTH

        logger.info(f"Loaded {len(self.sequences)} total padded sequences for {self.split} split")
        logger.info(
            f"Target Label metrics range: [{np.min(self.labels):.3f}, {np.max(self.labels):.3f}]"
        )

    def _load_and_filter_data(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            df = pd.read_csv(file_path, sep="\t", low_memory=False)
        except Exception as e:
            raise RuntimeError(f"Error loading K562 dataset: {e}")

        # Filter by allowed data projects from Boda2 setup
        allowed_projects = ["UKBB", "GTEX", "CRE"]
        df = df[df["data_project"].isin(allowed_projects)].reset_index(drop=True)

        activity_columns = ["K562_log2FC", "HepG2_log2FC", "SKNSH_log2FC"]
        stderr_columns = ["K562_lfcSE", "HepG2_lfcSE", "SKNSH_lfcSE"]

        # 1. Quality Filter: max SE across all 3 cell lines must be < 1.0!
        quality_filter = df[stderr_columns].max(axis=1) < 1.0
        df = df[quality_filter].reset_index(drop=True)

        # 2. Extreme Value Clipping identical to Malinois (±6 std, shifted +4 up)
        means = df[activity_columns].mean().to_numpy()
        stds = df[activity_columns].std().to_numpy()

        std_multiple_cut = 6.0
        up_cutoff_move = 4.0

        up_cut = means + stds * std_multiple_cut + up_cutoff_move
        down_cut = means - stds * std_multiple_cut

        b_up = (df[activity_columns] < up_cut).all(axis=1)
        df = df.loc[b_up]
        b_down = (df[activity_columns] > down_cut).all(axis=1)
        df = df.loc[b_down].reset_index(drop=True)

        logger.info(f"Retained {len(df)} total dataset lines after quality bounds processing.")

        # Replicating `row_pad_sequence` exactly
        def apply_boda_padding(seq: str) -> str:
            pad_needed = self.SEQUENCE_LENGTH - len(seq)
            if pad_needed > 0:
                # Rewriting exact logic match manually for robustness:
                len_up = pad_needed // 2
                len_down = (pad_needed + 1) // 2

                up_str = self.MPRA_UPSTREAM[-len_up:] if len_up > 0 else ""
                down_str = self.MPRA_DOWNSTREAM[:len_down] if len_down > 0 else ""
                return up_str + seq + down_str
            return seq

        sequences = df["sequence"].apply(apply_boda_padding).values
        # Though we calculate extreme bounds using all 3 to establish matching index retention,
        # we only emit K562 log2FC for actual training objective.
        labels = df["K562_log2FC"].values.astype(np.float32)
        ids = df["IDs"].values

        return sequences, labels, ids

    def _create_chromosome_splits(
        self, all_sequences: np.ndarray, all_labels: np.ndarray, all_ids: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        logger.info("Executing precise full-dataset Boda2 Chromosome parsing.")

        # IDs are structured like chr:pos:ref:alt:type:wc
        chrs = np.array([str(seq_id).split(":")[0] for seq_id in all_ids])

        val_chrs = {"19", "21", "X", "chr19", "chr21", "chrX"}
        test_chrs = {"7", "13", "chr7", "chr13"}

        val_mask = np.isin(chrs, list(val_chrs))
        test_mask = np.isin(chrs, list(test_chrs))
        # Keep everything remaining inside Train precisely
        train_mask = ~(val_mask | test_mask)

        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        test_indices = np.where(test_mask)[0]

        splits = {
            "train": (all_sequences[train_indices], all_labels[train_indices], train_indices),
            "val": (all_sequences[val_indices], all_labels[val_indices], val_indices),
            "test": (all_sequences[test_indices], all_labels[test_indices], test_indices),
        }

        logger.info(f"Generated Validation Splice {len(val_indices)} items.")
        logger.info(f"Generated Testing Splice {len(test_indices)} items.")
        logger.info(f"Generated Main Training Split {len(train_indices)} items.")

        return splits

    def encode_sequence(self, sequence: str, metadata: Optional[Dict] = None) -> np.ndarray:
        encoded = one_hot_encode(sequence, add_singleton_channel=False)
        rc_channel = np.zeros((1, len(sequence)), dtype=np.float32)
        encoded = np.concatenate([encoded, rc_channel], axis=0)
        return encoded

    def get_num_channels(self) -> int:
        return self.NUM_CHANNELS
