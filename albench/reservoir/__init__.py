"""Reservoir samplers.

Only samplers that are registered in :mod:`albench.core.registry` live here. Ten unregistered
samplers were removed on 2026-11-20 (``git show 79a12a6^:albench/reservoir/<name>.py`` recovers
any of them): they had no importer outside this file, no registry entry, and no pool in
``outputs/grid_pools`` was ever built from them, so nothing in the study depended on them.
Four of them also carried their own hardcoded consensus-motif lists, which is the thing
:mod:`albench.motifs.vocabulary` exists to replace.
"""

from albench.reservoir.base import ReservoirSampler
from albench.reservoir.evoaug import EvoAugSampler
from albench.reservoir.evoaug_structural import EvoAugStructuralSampler
from albench.reservoir.fixed_pool import FixedPoolSampler
from albench.reservoir.gc_matched import GCMatchedSampler
from albench.reservoir.genomic import GenomicSampler
from albench.reservoir.in_silico_evolution import InSilicoEvolutionSampler
from albench.reservoir.in_silico_evolution_generative import (
    InSilicoEvolutionGenerativeSampler,
)
from albench.reservoir.mixed_pool import MixedPoolSampler
from albench.reservoir.partial_mutagenesis import PartialMutagenesisSampler
from albench.reservoir.random_sampler import RandomSampler
from albench.reservoir.tf_motif_shuffle import TFMotifShuffleSampler

__all__ = [
    "ReservoirSampler",
    "EvoAugSampler",
    "EvoAugStructuralSampler",
    "FixedPoolSampler",
    "GCMatchedSampler",
    "GenomicSampler",
    "InSilicoEvolutionGenerativeSampler",
    "InSilicoEvolutionSampler",
    "MixedPoolSampler",
    "PartialMutagenesisSampler",
    "RandomSampler",
    "TFMotifShuffleSampler",
]
