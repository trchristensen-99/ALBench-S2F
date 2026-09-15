"""Core model interface for sequence-to-function predictors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SequenceModel(ABC):
    """Abstract interface for oracle and student models."""

    @abstractmethod
    def predict(self, sequences: list[str]) -> np.ndarray:
        """Map input sequences to scalar predictions with shape ``(N,)``."""

    def uncertainty(self, sequences: list[str]) -> np.ndarray:
        """Return uncertainty values with shape ``(N,)``."""
        raise NotImplementedError

    def embed(self, sequences: list[str]) -> np.ndarray:
        """Return embeddings with shape ``(N, D)``."""
        raise NotImplementedError

    def epistemic_uncertainty(self, sequences: list[str]) -> np.ndarray:
        """Reducible uncertainty -- disagreement between ensemble members or MC passes.

        This is what acquisition should chase: it shrinks as we label more. Models that
        cannot separate the two components fall back to total uncertainty, and callers
        that care about the distinction should check
        :meth:`separates_uncertainty` rather than silently accepting the fallback.
        """
        return self.uncertainty(sequences)

    def aleatoric_uncertainty(self, sequences: list[str]) -> np.ndarray:
        """Irreducible uncertainty -- measurement noise in the label itself.

        In MPRA this is large for weakly-expressed sequences, whose activity is
        estimated from few reads. Acquisition that does not divide it out will spend
        its budget on sequences whose labels cannot be measured precisely no matter
        how many we order.
        """
        return np.ones(len(sequences), dtype=float)

    @property
    def separates_uncertainty(self) -> bool:
        """Whether epistemic and aleatoric are genuinely distinct for this model."""
        return False

    def fit(self, sequences: list[str], labels: np.ndarray) -> None:
        """Fit or update model parameters on labeled examples."""
        raise NotImplementedError
