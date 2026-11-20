"""Tests for newly implemented acquisition/reservoir strategies."""

from __future__ import annotations

import numpy as np

from albench.acquisition.combined import CombinedAcquisition
from albench.acquisition.prior_knowledge import PriorKnowledgeAcquisition
from albench.reservoir.evoaug import EvoAugSampler
from albench.reservoir.in_silico_evolution import InSilicoEvolutionSampler
from albench.reservoir.partial_mutagenesis import PartialMutagenesisSampler
from albench.reservoir.tf_motif_shuffle import TFMotifShuffleSampler


class _DummyStudent:
    """Deterministic student double for acquisition tests."""

    def predict(self, sequences: list[str]) -> np.ndarray:
        return np.asarray([0.1, 0.2, 0.8, 0.9][: len(sequences)], dtype=np.float32)

    def uncertainty(self, sequences: list[str]) -> np.ndarray:
        return np.asarray([0.9, 0.8, 0.2, 0.1][: len(sequences)], dtype=np.float32)

    def embed(self, sequences: list[str]) -> np.ndarray:
        return np.asarray(
            [[0.0, 0.0], [0.1, 0.1], [2.0, 2.0], [2.1, 2.1]][: len(sequences)],
            dtype=np.float32,
        )


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


def test_combined_acquisition_supports_activity_prior() -> None:
    """Combined acquisition should support activity-prior-only ranking."""
    acq = CombinedAcquisition(
        alpha=None,
        w_uncertainty=0.0,
        w_diversity=0.0,
        w_activity_prior=1.0,
        target_activity=0.9,
    )
    selected = acq.select(_DummyStudent(), ["A", "C", "G", "T"], n_select=1)
    assert selected.shape == (1,)
    assert int(selected[0]) == 3


def test_partial_mutagenesis_rates_stay_inside_the_configured_band() -> None:
    """A uniform rate distribution draws every per-sequence rate inside [min_rate, max_rate].

    The earlier version of this test asked ``sample()`` to filter candidates by a
    ``mutation_fraction`` in their metadata, through constructor arguments
    (``min_mutation_fraction``/``max_mutation_fraction``) the sampler has never had --
    ``sample()`` is a plain random draw, documented as such. It therefore tested nothing
    and had never passed. The rate band is the behaviour that argument pair was reaching
    for, and it is real.
    """
    sampler = PartialMutagenesisSampler(
        seed=7, mutation_rate_distribution="uniform", min_rate=0.05, max_rate=0.2
    )
    rates = sampler._get_mutation_rates(512)
    assert rates.shape == (512,)
    assert rates.min() >= 0.05 and rates.max() <= 0.2


def test_tf_motif_shuffle_sampler_prefers_motif_rich_sequences() -> None:
    """TF motif sampler should prioritize motif-rich candidates."""
    sampler = TFMotifShuffleSampler(seed=11, motifs=["TATA"])
    candidates = ["TATATATA", "CCCCCCCC", "GGGGGGGG", "ATATATAT"]
    idx = sampler.sample(candidates=candidates, n_samples=2, metadata=None)
    assert set(idx).issubset({0, 3})


def test_evoaug_sampler_prefers_high_scores() -> None:
    """EvoAug sampler should select candidates with highest evoaug_score."""
    sampler = EvoAugSampler(seed=5, score_key="evoaug_score", fallback_random=False)
    candidates = ["A", "C", "G", "T"]
    metadata = [
        {"evoaug_score": 0.1},
        {"evoaug_score": 0.9},
        {"evoaug_score": 0.8},
        {"evoaug_score": 0.2},
    ]
    idx = sampler.sample(candidates=candidates, n_samples=2, metadata=metadata)
    assert set(idx).issubset({1, 2})


def test_in_silico_evolution_sampler_prefers_high_scores() -> None:
    """ISE sampler should prioritize highest evolutionary scores."""
    sampler = InSilicoEvolutionSampler(seed=3, fallback_random=False)
    candidates = ["A", "C", "G", "T"]
    metadata = [
        {"evolution_score": 0.4},
        {"evolution_score": 1.2},
        {"evolution_score": 0.9},
        {"evolution_score": 0.3},
    ]
    idx = sampler.sample(candidates=candidates, n_samples=2, metadata=metadata)
    assert set(idx).issubset({1, 2})
