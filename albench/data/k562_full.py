import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

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

    **Padding (default):** Each raw variable-length sequence is extended toward 600 bp using
    real flanking sequence: ``MPRA_UPSTREAM[-len_up:] + seq + MPRA_DOWNSTREAM[:len_down]``
    with len_up + len_down = 600 - len(seq). The MPRA flanks are 200 bp each; for variables
    shorter than 200 bp the result is < 600 bp, so N-padding is applied to reach 600 bp.

    **Compact window option:** Set ``store_raw=True`` and then ``set_compact_window(...)``.
    Each sample is built as left_flank + var + right_flank with fixed total length W (no N's
    in the variable part). Longer variables use less flank (slices of the 200 bp flanks).
    When ``compact_return_raw=True``, ``__getitem__`` returns (raw_seq, label) so the
    training collate can build the window with shift augmentation.
    """

    SEQUENCE_LENGTH = 600
    NUM_CHANNELS = 5

    # Flanks pulled directly from boda2-main/boda/common/constants.py (each 200 bp)
    MPRA_UPSTREAM = "ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG"
    MPRA_DOWNSTREAM = "CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCT"

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform: Optional[any] = None,
        target_transform: Optional[any] = None,
        subset_size: Optional[int] = None,
        store_raw: bool = False,
    ):
        self.subset_size = subset_size
        self.store_raw = store_raw
        self.raw_lengths: Optional[np.ndarray] = None
        self._compact_min_var_len: Optional[int] = None
        self._compact_flank_bp: int = 200
        self._compact_W: Optional[int] = None
        self._compact_return_raw: bool = False
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
        all_sequences, all_labels, all_ids, all_raw_lengths = self._load_and_filter_data(file_path)

        splits = self._create_chromosome_splits(all_sequences, all_labels, all_ids)

        # Valid split targets in this dataset structure are just train, val, and test.
        if self.split not in splits:
            raise ValueError(
                f"Split {self.split} is not valid for Full Dataset config. Use train, val, or test."
            )

        self.sequences, self.labels, self.indices = splits[self.split]
        if all_raw_lengths is not None:
            self.raw_lengths = all_raw_lengths[self.indices].copy()
        else:
            self.raw_lengths = None

        if self.subset_size is not None and self.subset_size < len(self.sequences):
            rng = np.random.default_rng()
            indices = rng.choice(len(self.sequences), size=self.subset_size, replace=False)
            self.sequences = self.sequences[indices]
            self.labels = self.labels[indices]
            if self.raw_lengths is not None:
                self.raw_lengths = self.raw_lengths[indices]
            logger.info(
                f"Downsampled to {self.subset_size:,} sequences (random sampling, no replacement)"
            )

        if self.store_raw and self._compact_W is not None:
            self.sequence_length = self._compact_W
        else:
            self.sequence_length = self.SEQUENCE_LENGTH if not self.store_raw else None

        logger.info(
            f"Loaded {len(self.sequences)} total {'raw' if self.store_raw else 'padded'} sequences for {self.split} split"
        )
        logger.info(
            f"Target Label metrics range: [{np.min(self.labels):.3f}, {np.max(self.labels):.3f}]"
        )

    def _load_and_filter_data(
        self, file_path: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
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

        raw_lengths = df["sequence"].str.len().values.astype(np.int64)

        if not self.store_raw:
            # Extend toward 600 bp with real flanks; N-pad to 600 if flanks are 200 bp and result < 600
            def apply_boda_padding(seq: str) -> str:
                pad_needed = self.SEQUENCE_LENGTH - len(seq)
                if pad_needed > 0:
                    len_up = pad_needed // 2
                    len_down = (pad_needed + 1) // 2
                    up_str = self.MPRA_UPSTREAM[-len_up:] if len_up > 0 else ""
                    down_str = self.MPRA_DOWNSTREAM[:len_down] if len_down > 0 else ""
                    out = up_str + seq + down_str
                    if len(out) < self.SEQUENCE_LENGTH:
                        n_pad = self.SEQUENCE_LENGTH - len(out)
                        left_n = n_pad // 2
                        right_n = n_pad - left_n
                        out = "N" * left_n + out + "N" * right_n
                    return out
                return seq

            sequences = df["sequence"].apply(apply_boda_padding).values
        else:
            sequences = df["sequence"].values

        labels = df["K562_log2FC"].values.astype(np.float32)
        ids = df["IDs"].values
        return sequences, labels, ids, raw_lengths if self.store_raw else None

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

    def set_compact_window(
        self,
        min_var_len: int,
        flank_bp: int = 200,
        window_bp: Optional[int] = None,
    ) -> None:
        """Configure compact window: full variable + flank slices to fixed W (no N's in var).

        Call after load_data when store_raw=True.
        - If window_bp is set (e.g. 384), W = window_bp. Else W = min_var_len + 2*flank_bp.
        """
        if not self.store_raw:
            raise RuntimeError("set_compact_window requires store_raw=True")
        self._compact_min_var_len = min_var_len
        self._compact_flank_bp = min(flank_bp, len(self.MPRA_UPSTREAM), len(self.MPRA_DOWNSTREAM))
        self._compact_W = (
            window_bp
            if window_bp is not None
            else self._compact_min_var_len + 2 * self._compact_flank_bp
        )
        self.sequence_length = self._compact_W

    def set_compact_return_raw(self, return_raw: bool = True) -> None:
        """When True, __getitem__ returns (raw_sequence_str, label) for compact collate with shift."""
        self._compact_return_raw = return_raw

    def _build_compact_sequence(self, seq: str, shift: int = 0) -> str:
        """Build left_flank + var + right_flank (full var, no N's). shift redistributes flank."""
        W = self._compact_W
        L = len(seq)
        available = W - L
        if available < 0:
            raise ValueError(f"Compact window W={W} < len(seq)={L}")
        cap = min(200, self._compact_flank_bp)
        left_len_base = min(cap, available // 2)
        # shift: positive => more right flank; left_len = left_len_base - shift, clamped
        left_len = max(max(0, available - cap), min(cap, left_len_base - shift))
        right_len = available - left_len
        left = self.MPRA_UPSTREAM[-left_len:] if left_len > 0 else ""
        right = self.MPRA_DOWNSTREAM[:right_len] if right_len > 0 else ""
        return left + seq + right

    def __getitem__(self, idx: int):
        if self.store_raw and self._compact_W is not None and self._compact_return_raw:
            return (
                self.sequences[idx],
                torch.tensor(self.labels[idx], dtype=torch.float32),
            )
        return super().__getitem__(idx)

    def encode_sequence(self, sequence: str, metadata: Optional[Dict] = None) -> np.ndarray:
        if self._compact_W is not None:
            sequence = self._build_compact_sequence(sequence, shift=0)
        encoded = one_hot_encode(sequence, add_singleton_channel=False)
        rc_channel = np.zeros((1, len(sequence)), dtype=np.float32)
        encoded = np.concatenate([encoded, rc_channel], axis=0)
        return encoded

    def get_num_channels(self) -> int:
        return self.NUM_CHANNELS


# Module-level exports for compact collate (train script imports these)
MPRA_UPSTREAM = K562FullDataset.MPRA_UPSTREAM
MPRA_DOWNSTREAM = K562FullDataset.MPRA_DOWNSTREAM
