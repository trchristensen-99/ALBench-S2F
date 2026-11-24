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


class ExpectedActivityAcquisition(AcquisitionFunction):
    """Top-k by PREDICTED ACTIVITY. Pure exploitation, and the family's representative.

    This is what someone would do without an active-learning framework at all -- "measure
    the sequences the model thinks are strongest" -- so it is the baseline the uncertainty
    rules have to beat, and it is not a control but a serious candidate: a library whose
    purpose includes designing strong enhancers has a real reason to spend budget at the
    top of the range.

    It is also the sharpest test of whether `epistemic` is doing anything. Arm B's
    epistemic arm selected a label mean of 2.38 against a pool mean of 1.21, so it is
    already selecting strongly for activity; if ranking by activity ALONE performs the
    same, then epistemic uncertainty contributed nothing beyond the activity it induces
    and the uncertainty machinery is decoration.

    Guided by the student bank, never the oracle: an oracle-ranked top-k is selecting at
    the labeller's own argmax, which is how the earlier `oracle_top` arm collapsed.
    """

    def __init__(self, seed: int | None = None, objective: str = "maximize") -> None:
        if objective not in ("maximize", "minimize", "extreme"):
            raise ValueError(f"objective must be maximize|minimize|extreme, got {objective!r}")
        self.seed = seed
        self.objective = objective

    def select(
        self,
        student: SequenceModel,
        candidates: list[str],
        n_select: int,
    ) -> np.ndarray:
        if n_select > len(candidates):
            raise ValueError("n_select cannot exceed candidate count")
        y = np.asarray(student.predict(candidates), dtype=float).ravel()
        if self.objective == "maximize":
            s = y
        elif self.objective == "minimize":
            s = -y
        else:
            # Both tails: a corpus that only buys the top of the range teaches nothing
            # about the bottom, and the bottom is half of what a model has to predict.
            s = np.abs(y - np.median(y))
        return np.argsort(s)[-n_select:]
