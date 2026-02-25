"""Tests for generic fixed-pool reservoir sampler."""

from __future__ import annotations

import pytest

from albench.reservoir.fixed_pool import FixedPoolSampler


def test_fixed_pool_sampler_basic_count() -> None:
    """Sampler returns requested number of indices without metadata filters."""
    sampler = FixedPoolSampler(seed=7)
    idx = sampler.sample(candidates=["A", "C", "G", "T"], n_samples=2)
    assert len(idx) == 2
    assert len(set(idx)) == 2


def test_fixed_pool_sampler_metadata_filter() -> None:
    """Sampler respects metadata filters."""
    sampler = FixedPoolSampler(seed=7, metadata_filters={"chromosome": ["chr1"]})
    candidates = ["A", "C", "G", "T"]
    metadata = [
        {"chromosome": "chr1"},
        {"chromosome": "chr2"},
        {"chromosome": "chr1"},
        {"chromosome": "chr3"},
    ]
    idx = sampler.sample(candidates=candidates, n_samples=2, metadata=metadata)
    assert set(idx).issubset({0, 2})


def test_fixed_pool_sampler_requires_metadata_with_filters() -> None:
    """Sampler errors if filters are configured but metadata is missing."""
    sampler = FixedPoolSampler(seed=1, metadata_filters={"chromosome": ["chr1"]})
    with pytest.raises(ValueError):
        sampler.sample(candidates=["A", "C"], n_samples=1, metadata=None)
