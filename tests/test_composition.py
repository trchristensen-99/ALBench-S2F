"""Composition metrics and the in-support vs OOD evaluation split.

The split exists so the reservoir ranking can be read without the oracle's
extreme-GC extrapolation tail; these tests pin the properties that make it
trustworthy: GC/CpG are computed correctly, the support mask is inclusive, and
splitting a known-structured signal recovers the expected per-partition result.
"""

from __future__ import annotations

import numpy as np
import pytest

from albench.composition import (
    GENOMIC_GC_HI,
    GENOMIC_GC_LO,
    cpg_obs_exp,
    gc_content,
    gc_content_batch,
    in_support_mask,
    split_support_metrics,
    support_range_from_sequences,
)


def test_gc_content_basic():
    assert gc_content("GGGG") == 1.0
    assert gc_content("AATT") == 0.0
    assert gc_content("ACGT") == 0.5
    # N is ignored, not counted as AT or GC.
    assert gc_content("GCNN") == 1.0


def test_cpg_obs_exp_depletion_vs_enrichment():
    # Alternating CG is CpG-enriched relative to composition expectation (>1).
    enriched = cpg_obs_exp("CGCGCGCGCG")
    # Same composition, CpG arrangement destroyed -> depleted (0 CpG dinucs).
    depleted = cpg_obs_exp("CCCCCGGGGG")
    assert enriched > 1.0
    assert depleted < enriched
    assert cpg_obs_exp("AAAA") == 0.0  # no C or G


def test_support_mask_is_inclusive_and_banded():
    gc = np.array([0.1, GENOMIC_GC_LO, 0.5, GENOMIC_GC_HI, 0.95])
    mask = in_support_mask(gc)
    # endpoints included, tails excluded
    assert mask.tolist() == [False, True, True, True, False]


def test_support_range_from_sequences_recovers_central_interval():
    # 100 sequences with GC marching from ~0 to ~1; central 98% trims the tails.
    seqs = []
    for i in range(100):
        n_gc = i  # 0..99 G's out of 100
        seqs.append("G" * n_gc + "A" * (100 - n_gc))
    lo, hi = support_range_from_sequences(seqs, central=0.98)
    assert lo < 0.1 and hi > 0.9  # trimmed but still spans most of the range
    assert lo < hi


def test_split_recovers_partition_specific_correlation():
    rng = np.random.default_rng(0)
    n = 4000
    gc = rng.uniform(0.0, 1.0, n)
    target = rng.normal(size=n)
    # In support: pred tracks target (high r). OOD: pred is pure noise (r~0).
    mask = in_support_mask(gc)
    pred = np.where(mask, target + rng.normal(scale=0.3, size=n), rng.normal(size=n))

    out = split_support_metrics(pred, target, gc)
    assert out["in_support"]["pearson_r"] > 0.85
    assert abs(out["ood"]["pearson_r"]) < 0.15
    # partitions cover the whole set
    assert out["in_support"]["n"] + out["ood"]["n"] == n
    assert out["all"]["n"] == n
    assert np.isclose(out["in_support"]["frac"] + out["ood"]["frac"], 1.0)


def test_split_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        split_support_metrics(np.zeros(5), np.zeros(5), np.zeros(4))


def test_gc_content_batch_matches_scalar():
    seqs = ["ACGT", "GGGG", "AAAA", "ACGTACGTGC"]
    batch = gc_content_batch(seqs)
    scalar = np.array([gc_content(s) for s in seqs])
    assert np.allclose(batch, scalar)
