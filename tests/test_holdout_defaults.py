"""Reservoir samplers that read genomic sequence must exclude the test AND validation chromosomes by default (AGENTS.md house rule)."""
from __future__ import annotations

import inspect

HELD_OUT = {"chr7", "chr13", "chr19", "chr21", "chrX"}


def test_zoonomia_default_excludes_all_held_out_chromosomes():
    from albench.reservoir.zoonomia_orthologs import ZoonomiaOrthologSampler

    default = inspect.signature(ZoonomiaOrthologSampler.__init__).parameters["exclude_chroms"].default
    assert HELD_OUT <= set(default), f"Zoonomia sampler default excludes only {default}"


def test_encode_default_excludes_all_held_out_chromosomes():
    from albench.reservoir.encode_accessibility import EncodeAccessibilitySampler

    default = inspect.signature(EncodeAccessibilitySampler.__init__).parameters["exclude_chroms"].default
    assert HELD_OUT <= set(default)
