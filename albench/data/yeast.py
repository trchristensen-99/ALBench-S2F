"""
Yeast promoter MPRA dataset loader.

Dataset from: de Boer et al., Nature Biotechnology 2024
Zenodo: https://zenodo.org/records/10633252

Following DREAM challenge preprocessing:
- 80bp random sequences padded to 150bp with plasmid context
- 6 channels: ACGT + reverse complement flag + singleton flag
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import SequenceDataset
from .utils import one_hot_encode


class YeastDataset(SequenceDataset):
    """Yeast promoter MPRA dataset."""

    FLANK_5_PRIME = "GCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCATTTTTTTCACATCTCG"  # 57bp
    FLANK_3_PRIME = "GGTTACGGCTGTT"  # 13bp

    SEQUENCE_LENGTH = 150
    RANDOM_REGION_LENGTH = 80
    NUM_CHANNELS = 6
    FIXED_TRAIN_SIZE = 100_000
    FIXED_VAL_SIZE = 20_000

    def __init__(self, data_path: str, split: str = "train", subset_size: Optional[int] = None):
        self.subset_size = subset_size
        super().__init__(data_path, split)

    @staticmethod
    def _deterministic_order(n: int, seed: int = 42) -> np.ndarray:
        """Return a deterministic random ordering with a fixed seed."""
        rng = np.random.default_rng(seed=seed)
        return rng.permutation(n)

    def load_data(self) -> None:
        """Load yeast MPRA data with fixed train/pool/val/test splits."""
        data_dir = Path(self.data_path)

        if self.split in ["train", "pool"]:
            file_path = data_dir / "train.txt"
            print(f"Loading Yeast train data from {file_path}")
            df = pd.read_csv(file_path, sep="\t", header=None, names=["sequence", "expression"])

            order = self._deterministic_order(len(df), seed=42)
            train_idx = order[: self.FIXED_TRAIN_SIZE]
            pool_idx = order[self.FIXED_TRAIN_SIZE :]

            if self.split == "train":
                df = df.iloc[train_idx].reset_index(drop=True)
                print(f"Using fixed train split: {len(df):,} sequences")
            else:
                df = df.iloc[pool_idx].reset_index(drop=True)
                print(f"Using fixed pool split: {len(df):,} sequences")

            is_singleton = (df["expression"] % 1 == 0).values.astype(np.float32)

        elif self.split == "val":
            file_path = data_dir / "val.txt"
            print(f"Loading Yeast val data from {file_path}")
            df = pd.read_csv(file_path, sep="\t", header=None, names=["sequence", "expression"])

            if len(df) > self.FIXED_VAL_SIZE:
                order = self._deterministic_order(len(df), seed=42)
                keep_idx = order[: self.FIXED_VAL_SIZE]
                df = df.iloc[keep_idx].reset_index(drop=True)
                print(f"Using fixed validation subset: {self.FIXED_VAL_SIZE:,} sequences")

            is_singleton = (df["expression"] % 1 == 0).values.astype(np.float32)

        elif self.split == "test":
            file_path = data_dir / "filtered_test_data_with_MAUDE_expression.txt"
            print(f"Loading Yeast test data from {file_path}")
            df = pd.read_csv(file_path, sep="\t", header=None, names=["sequence", "expression"])
            is_singleton = (df["expression"] % 1 == 0).values.astype(np.float32)
        else:
            raise ValueError(
                f"Invalid split: {self.split}. Expected one of: train, pool, val, test"
            )

        self.sequences = df["sequence"].values
        self.labels = df["expression"].values.astype(np.float32)
        self.is_singleton = is_singleton

        self.sequences = self._add_plasmid_context(self.sequences)

        if self.subset_size is not None and self.subset_size < len(self.sequences):
            indices = np.random.choice(len(self.sequences), size=self.subset_size, replace=False)
            self.sequences = self.sequences[indices]
            self.labels = self.labels[indices]
            self.is_singleton = self.is_singleton[indices]
            print(f"Downsampled to {self.subset_size} sequences (random sampling, no replacement)")

        self.sequence_length = self.SEQUENCE_LENGTH

        print(f"Loaded {len(self.sequences)} sequences for {self.split} split")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Label range: [{np.min(self.labels):.3f}, {np.max(self.labels):.3f}]")

    def _add_plasmid_context(self, sequences: np.ndarray) -> np.ndarray:
        """Add plasmid flanking sequences to get 150bp sequences."""
        processed = []
        partial_5_prime = self.FLANK_5_PRIME[-17:]

        for seq in sequences:
            if seq.endswith(self.FLANK_3_PRIME):
                seq = seq[: -len(self.FLANK_3_PRIME)]

            if seq.startswith(partial_5_prime):
                seq = seq[len(partial_5_prime) :]

            if len(seq) < self.RANDOM_REGION_LENGTH:
                seq = seq + "N" * (self.RANDOM_REGION_LENGTH - len(seq))
            elif len(seq) > self.RANDOM_REGION_LENGTH:
                seq = seq[: self.RANDOM_REGION_LENGTH]

            full_seq = self.FLANK_5_PRIME + seq + self.FLANK_3_PRIME

            if len(full_seq) != self.SEQUENCE_LENGTH:
                raise ValueError(
                    f"Sequence length mismatch: {len(full_seq)} != {self.SEQUENCE_LENGTH}\n"
                    f"5' flank: {len(self.FLANK_5_PRIME)}, random: {len(seq)}, 3' flank: {len(self.FLANK_3_PRIME)}"
                )

            processed.append(full_seq)

        return np.array(processed)

    def encode_sequence(self, sequence: str, metadata: Optional[Dict] = None) -> np.ndarray:
        """Encode a yeast sequence with 6 channels."""
        encoded = one_hot_encode(sequence, add_singleton_channel=False)
        rc_channel = np.zeros((1, len(sequence)), dtype=np.float32)

        if metadata is not None and "is_singleton" in metadata:
            singleton_value = metadata["is_singleton"]
        else:
            singleton_value = 0.0

        singleton_channel = np.full((1, len(sequence)), singleton_value, dtype=np.float32)
        encoded = np.concatenate([encoded, rc_channel, singleton_channel], axis=0)
        return encoded

    def __getitem__(self, idx: int) -> tuple:
        """Get a single sample."""
        import torch

        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Singleton channel is label-derived, so keep it off at inference/eval.
        singleton_value = 0.0
        if self.split == "train" and hasattr(self, "is_singleton"):
            singleton_value = float(self.is_singleton[idx])
        metadata = {"is_singleton": singleton_value}

        encoded = self.encode_sequence(sequence, metadata)
        encoded_tensor = torch.from_numpy(encoded).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return encoded_tensor, label_tensor

    def get_num_channels(self) -> int:
        """Return number of input channels (6 for yeast)."""
        return self.NUM_CHANNELS
