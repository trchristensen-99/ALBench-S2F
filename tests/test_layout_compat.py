"""Compatibility tests for requested module layout."""

from __future__ import annotations

from albench.acquisition.ensemble_acq import EnsembleAcquisition
from albench.data.loaders import K562Dataset, YeastDataset, load_fasta_sequences
from albench.data.sequence_utils import one_hot_encode
from albench.models.alphagenome_wrapper import AlphaGenomeWrapper
from albench.oracle import Oracle
from albench.reservoir.in_silico_evolution import InSilicoEvolutionSampler
from albench.student import Student


def test_top_level_interface_aliases() -> None:
    """Top-level compatibility modules should expose expected interfaces."""
    assert Oracle is not None
    assert Student is not None


def test_new_layout_modules_import() -> None:
    """New compatibility modules should be importable."""
    assert AlphaGenomeWrapper is not None
    assert InSilicoEvolutionSampler is not None
    assert EnsembleAcquisition is not None
    assert K562Dataset is not None
    assert YeastDataset is not None
    assert load_fasta_sequences is not None
    assert one_hot_encode is not None
