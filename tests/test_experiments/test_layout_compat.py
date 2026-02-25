"""Layout and compatibility tests for the refactored module structure."""

from __future__ import annotations

from albench.acquisition.ensemble_acq import EnsembleAcquisition
from albench.model import SequenceModel
from albench.reservoir.in_silico_evolution import InSilicoEvolutionSampler
from data.loaders import K562Dataset, YeastDataset, load_fasta_sequences
from data.sequence_utils import one_hot_encode
from models.alphagenome_wrapper import AlphaGenomeWrapper


def test_sequence_model_is_the_unified_interface() -> None:
    """Oracle and student roles are both fulfilled by SequenceModel."""
    assert SequenceModel is not None
    # The old Oracle/Student split is gone; both roles use SequenceModel
    assert hasattr(SequenceModel, "predict")
    assert hasattr(SequenceModel, "fit")
    assert hasattr(SequenceModel, "uncertainty")
    assert hasattr(SequenceModel, "embed")


def test_new_layout_modules_import() -> None:
    """New top-level packages (data/, models/) should be importable."""
    assert AlphaGenomeWrapper is not None
    assert InSilicoEvolutionSampler is not None
    assert EnsembleAcquisition is not None
    assert K562Dataset is not None
    assert YeastDataset is not None
    assert load_fasta_sequences is not None
    assert one_hot_encode is not None
