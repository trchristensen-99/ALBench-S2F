"""Diversity-driven acquisition based on student embeddings."""

from __future__ import annotations

import numpy as np

from albench.acquisition.base import AcquisitionFunction
from albench.model import SequenceModel


class DiversityAcquisition(AcquisitionFunction):
    """Greedy farthest-point selection in embedding space."""

    def select(
        self,
        student: SequenceModel,
        candidates: list[str],
        n_select: int,
    ) -> np.ndarray:
        """Select a diverse subset of candidate indices."""
        if n_select > len(candidates):
            raise ValueError("n_select cannot exceed candidate count")
        embeddings = student.embed(candidates)
        selected: list[int] = [0]
        while len(selected) < n_select:
            dist_to_selected = np.min(
                np.linalg.norm(embeddings[:, None, :] - embeddings[selected][None, :, :], axis=2),
                axis=1,
            )
            dist_to_selected[selected] = -np.inf
            selected.append(int(np.argmax(dist_to_selected)))
        return np.asarray(selected, dtype=np.int64)
