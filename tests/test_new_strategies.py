"""Tests for newly implemented acquisition/reservoir strategies."""

from __future__ import annotations

import numpy as np

from albench.acquisition.prior_knowledge import PriorKnowledgeAcquisition
from albench.reservoir.partial_mutagenesis import PartialMutagenesisSampler


class _DummyStudent:
    """Deterministic student double for acquisition tests."""

    def predict(self, sequences: list[str]) -> np.ndarray:
        return np.asarray([0.1, 0.2, 0.8, 0.9][: len(sequences)], dtype=np.float32)


def test_prior_knowledge_select_count() -> None:
    """PriorKnowledgeAcquisition returns requested number of indices."""
    acq = PriorKnowledgeAcquisition(
        w_activity_prior=0.5,
        w_motif_prior=0.3,
        w_gc_prior=0.2,
        target_activity=0.85,
        target_gc=0.5,
        motifs=["TATA"],
    )
    selected = acq.select(
        _DummyStudent(), ["TATATATA", "CCCCCCCC", "GGGGGGGG", "AAAAAAAA"], n_select=2
    )
    assert selected.shape == (2,)


def test_partial_mutagenesis_sampler_prefers_filtered_metadata() -> None:
    """Sampler should prefer mutation-fraction-matching candidates."""
    sampler = PartialMutagenesisSampler(
        seed=7, min_mutation_fraction=0.05, max_mutation_fraction=0.2
    )
    candidates = ["A", "C", "G", "T"]
    metadata = [
        {"mutation_fraction": 0.01},
        {"mutation_fraction": 0.10},
        {"mutation_fraction": 0.15},
        {"mutation_fraction": 0.30},
    ]
    idx = sampler.sample(candidates=candidates, n_samples=2, metadata=metadata)
    assert set(idx).issubset({1, 2})
