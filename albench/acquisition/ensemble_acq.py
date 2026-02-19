"""Ensemble-based acquisition strategy."""

from __future__ import annotations

import numpy as np

from albench.acquisition.base import AcquisitionFunction
from albench.model import SequenceModel


class EnsembleAcquisition(AcquisitionFunction):
    """Rank candidates by ensemble disagreement.

    Falls back to ``student.uncertainty`` when explicit ensemble members
    are not available on the wrapper.
    """

    def select(
        self,
        student: SequenceModel,
        candidates: list[str],
        n_select: int,
    ) -> np.ndarray:
        """Select top-k candidates with largest disagreement."""
        if n_select > len(candidates):
            raise ValueError("n_select cannot exceed candidate count")

        scores: np.ndarray
        if hasattr(student, "models") and hasattr(student, "_encode_sequences"):
            models = getattr(student, "models")
            encode = getattr(student, "_encode_sequences")
            if isinstance(models, list) and len(models) > 1:
                x = encode(candidates)
                member_preds: list[np.ndarray] = []
                for model in models:
                    model.eval()
                    with np.errstate(all="ignore"):
                        pred = student._predict_member(model, x)  # type: ignore[attr-defined]
                    member_preds.append(np.asarray(pred, dtype=np.float32))
                scores = np.var(np.stack(member_preds, axis=0), axis=0)
            else:
                scores = np.asarray(student.uncertainty(candidates), dtype=np.float32)
        else:
            scores = np.asarray(student.uncertainty(candidates), dtype=np.float32)
        return np.argsort(scores)[-n_select:]
