"""albench — lightweight, model-agnostic active-learning engine.

Public API
----------
The entire usable surface is importable directly from ``albench``::

    from albench import ALLoop, run_al_loop, SequenceModel, RunConfig, RoundResult
    from albench.task import TaskConfig
    from albench.reservoir.random_sampler import RandomSampler
    from albench.acquisition.random_acq import RandomAcquisition
"""

from __future__ import annotations

# Evaluation helpers
from albench.evaluation import compute_scaling_curve, evaluate_on_test_sets

# Loop driver — primary and functional APIs
from albench.loop import ALLoop, RoundResult, RunConfig, run_al_loop

# Core model interface
from albench.model import SequenceModel

# Task configuration
from albench.task import TaskConfig

# Sequence utilities
from albench.utils import (
    batch_encode_sequences,
    one_hot_encode,
    pad_sequence,
    reverse_complement,
    reverse_complement_one_hot,
    validate_sequence,
)

__version__ = "0.1.0"

__all__ = [
    # Primary abstractions
    "SequenceModel",
    "ALLoop",
    "RunConfig",
    "RoundResult",
    "TaskConfig",
    # Functional API
    "run_al_loop",
    # Evaluation
    "evaluate_on_test_sets",
    "compute_scaling_curve",
    # Sequence utilities
    "one_hot_encode",
    "reverse_complement",
    "reverse_complement_one_hot",
    "pad_sequence",
    "batch_encode_sequences",
    "validate_sequence",
    "__version__",
]
