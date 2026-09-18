"""
Yeast promoter MPRA dataset loader.

Dataset from: de Boer et al., Nature Biotechnology 2024
Zenodo: https://zenodo.org/records/10633252

Following DREAM challenge preprocessing:
- 80bp random sequences padded to 150bp with plasmid context
- 6 channels: ACGT + reverse complement flag + singleton flag
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import SequenceDataset
from .utils import one_hot_encode

# ---------------------------------------------------------------------------
# Single source of truth for yeast plasmid context.
#
# VERIFIED against the raw DREAM file: 100.00% of 400,000 sampled rows END with
# YEAST_FLANK_3[:13] and START with YEAST_FLANK_5[37:54]. The yeast_f5/yeast_f3
# literals that used to live in experiments/exp1_1_scaling.py matched 0.00% at
# BOTH ends -- they were a different construct -- and have been removed.
#
# Rows already carry 17bp of 5' flank and 13bp of 3' flank, so anything that
# re-attaches full flanks MUST strip the overlap or it duplicates them.
YEAST_FLANK_5 = "GCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCATTTTTTTCACATC"  # 54bp
YEAST_FLANK_3 = (
    "GGTTACGGCTGTTTCTTAATTAAAAAAAGATAGAAAACATTAGGAGTGTAACACAAGACT"
    "TTCGGATCCTGAGCAGGCAAGATAAACGA"
)  # 89bp
YEAST_EMBEDDED_5 = 17  # bases of YEAST_FLANK_5 already present at the row start
YEAST_EMBEDDED_3 = 13  # bases of YEAST_FLANK_3 already present at the row end


def build_yeast_context(seq: str, window: int = 384, pad_char: str = "N") -> str:
    """Place one raw DREAM row in a `window`-bp input, using REAL plasmid context.

    The row is 5'flank[17] + random + 3'flank[13]. We re-attach only the parts of
    the full flanks that are NOT already there (+37bp upstream, +76bp downstream),
    which is all the real sequence available. If that still falls short of
    `window` the remainder is padded -- there is no more known plasmid sequence,
    so padding is unavoidable in yeast, unlike human.

    Shrinking `window` toward the true construct length is therefore the more
    meaningful lever than the choice of pad character.
    """
    seq = seq.upper()
    extra5 = YEAST_FLANK_5[:-YEAST_EMBEDDED_5]  # 37bp
    extra3 = YEAST_FLANK_3[YEAST_EMBEDDED_3:]  # 76bp
    full = extra5 + seq + extra3
    if len(full) >= window:  # centre-crop
        start = (len(full) - window) // 2
        return full[start : start + window]
    pad = window - len(full)
    left = pad // 2
    return pad_char * left + full + pad_char * (pad - left)


class YeastDataset(SequenceDataset):
    """Yeast promoter MPRA dataset."""

    FLANK_5_PRIME = "GCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCATTTTTTTCACATCTCG"  # 57bp
    FLANK_3_PRIME = "GGTTACGGCTGTT"  # 13bp

    SEQUENCE_LENGTH = 150
    # Env-overridable so the window-size screen can sweep it. The true pTpA
    # construct is only ~223bp, so 384 leaves ~42% of every input fabricated.
    ALPHAGENOME_SEQUENCE_LENGTH = int(os.environ.get("ALBENCH_YEAST_WINDOW", "384"))
    RANDOM_REGION_LENGTH = 80
    NUM_CHANNELS = 6
    FIXED_VAL_SIZE = 20_000
    ALPHAGENOME_FLANK_5_PRIME = "GCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCATTTTTTTCACATC"  # 54bp
    ALPHAGENOME_FLANK_3_PRIME = "GGTTACGGCTGTTTCTTAATTAAAAAAAGATAGAAAACATTAGGAGTGTAACACAAGACTTTCGGATCCTGAGCAGGCAAGATAAACGA"  # 89bp

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        subset_size: Optional[int] = None,
        context_mode: str = "dream150",
    ):
        self.subset_size = subset_size
        if context_mode not in {"dream150", "alphagenome384"}:
            raise ValueError(
                f"Invalid context_mode={context_mode!r}. Expected 'dream150' or 'alphagenome384'."
            )
        self.context_mode = context_mode
        super().__init__(data_path, split)

    @staticmethod
    def _deterministic_order(n: int, seed: int = 42) -> np.ndarray:
        """Return a deterministic random ordering with a fixed seed."""
        rng = np.random.default_rng(seed=seed)
        return rng.permutation(n)

    def load_data(self) -> None:
        """Load yeast MPRA data with train/val/test splits."""
        data_dir = Path(self.data_path)

        if self.split == "train":
            file_path = data_dir / "train.txt"
            print(f"Loading Yeast train data from {file_path}")
            df = pd.read_csv(file_path, sep="\t", header=None, names=["sequence", "expression"])
            print(f"Loaded {len(df):,} training sequences")

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
            raise ValueError(f"Invalid split: {self.split}. Expected one of: train, val, test")

        self.sequences = df["sequence"].values
        self.labels = df["expression"].values.astype(np.float32)
        self.is_singleton = is_singleton

        self.sequences = self._add_plasmid_context(self.sequences)
        if self.context_mode == "alphagenome384":
            self.sequences = self._add_alphagenome_context(self.sequences)

        if self.subset_size is not None and self.subset_size < len(self.sequences):
            indices = np.random.choice(len(self.sequences), size=self.subset_size, replace=False)
            self.sequences = self.sequences[indices]
            self.labels = self.labels[indices]
            self.is_singleton = self.is_singleton[indices]
            print(f"Downsampled to {self.subset_size} sequences (random sampling, no replacement)")

        if self.context_mode == "alphagenome384":
            self.sequence_length = self.ALPHAGENOME_SEQUENCE_LENGTH
        else:
            self.sequence_length = self.SEQUENCE_LENGTH

        print(f"Loaded {len(self.sequences)} sequences for {self.split} split")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Label range: [{np.min(self.labels):.3f}, {np.max(self.labels):.3f}]")

    def _add_plasmid_context(self, sequences: np.ndarray) -> np.ndarray:
        """Add plasmid flanking sequences to get 150bp sequences."""
        processed = []
        # The rows carry FLANK_5_PRIME[37:54], NOT the final 17bp -- FLANK_5_PRIME
        # ends in an extra "TCG". Slicing [-17:] meant this strip never fired.
        partial_5_prime = self.FLANK_5_PRIME[37:54]

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

    def _add_alphagenome_context(self, sequences_150bp: np.ndarray) -> np.ndarray:
        """Expand 150bp DREAM-formatted sequences to AlphaGenome 384bp context."""
        processed = []
        flank5 = self.ALPHAGENOME_FLANK_5_PRIME
        flank3 = self.ALPHAGENOME_FLANK_3_PRIME
        del flank5, flank3  # superseded by build_yeast_context

        for seq in sequences_150bp:
            expanded = build_yeast_context(seq, window=self.ALPHAGENOME_SEQUENCE_LENGTH)
            if len(expanded) != self.ALPHAGENOME_SEQUENCE_LENGTH:
                raise ValueError(
                    f"AlphaGenome-expanded sequence has length {len(expanded)} "
                    f"(expected {self.ALPHAGENOME_SEQUENCE_LENGTH})"
                )
            processed.append(expanded)

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
