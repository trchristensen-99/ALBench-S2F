"""Prior-knowledge acquisition function stub."""

from __future__ import annotations

import numpy as np

from albench.acquisition.base import AcquisitionFunction
from albench.model import SequenceModel


class PriorKnowledgeAcquisition(AcquisitionFunction):
    """Placeholder for motif/activity-coverage selection baseline."""

    def select(
        self,
        student: SequenceModel,
        candidates: list[str],
        n_select: int,
    ) -> np.ndarray:
        """Select indices (not yet implemented)."""
        raise NotImplementedError("PriorKnowledgeAcquisition is not implemented yet")
