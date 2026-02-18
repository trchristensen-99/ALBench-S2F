"""AlphaGenome oracle placeholder."""

from __future__ import annotations

import numpy as np

from albench.model import SequenceModel


class AlphaGenomeOracle(SequenceModel):
    """Stub wrapper for future AlphaGenome oracle integration."""

    def predict(self, sequences: list[str]) -> np.ndarray:
        """Predict labels from AlphaGenome (not yet implemented)."""
        raise NotImplementedError("AlphaGenomeOracle is not implemented yet")
