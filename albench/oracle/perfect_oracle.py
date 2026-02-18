"""Perfect in-silico oracle implementation."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from albench.model import SequenceModel


class PerfectOracle(SequenceModel):
    """Oracle that returns ground-truth labels from a sequence->label map."""

    def __init__(self, labels_by_sequence: Mapping[str, float]) -> None:
        """Initialize the oracle with exact labels.

        Args:
            labels_by_sequence: Mapping from sequence string to true activity label.
        """
        self._labels = dict(labels_by_sequence)

    def predict(self, sequences: list[str]) -> np.ndarray:
        """Return exact labels for input sequences.

        Args:
            sequences: DNA sequences to query.

        Returns:
            Array of true labels with shape ``(N,)``.

        Raises:
            KeyError: If a sequence is not present in the oracle mapping.
        """
        values = [self._labels[seq] for seq in sequences]
        return np.asarray(values, dtype=np.float32)
