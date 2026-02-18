"""Task-level configuration objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for a sequence-to-function learning task.

    Attributes:
        name: Task identifier (e.g. ``k562``, ``yeast``).
        organism: Species name.
        sequence_length: Total padded sequence length (bp).
        data_root: Root directory for dataset files.
        input_channels: Number of model input channels (5 for K562, 6 for yeast).
        task_mode: Training mode — ``k562`` (regression) or ``yeast`` (18-bin KL).
        random_region_length: Length of the random/variable region (bp).
        train_path: Path to training data file.
        pool_path: Path to unlabeled pool data (for AL selection).
        val_path: Path to validation data file.
        test_set: Mapping of test-set name to ``{sequences, labels}`` dicts.
        flanking_sequence: 5' and 3' flanking sequences for the organism.
        oracle_checkpoint: Path to oracle model checkpoint.
        sequence_filters: Filter descriptors applied during data loading.
        metadata: Additional task-specific key-value pairs.
    """

    name: str
    organism: str
    sequence_length: int
    data_root: str
    input_channels: int = 5
    task_mode: str = "k562"
    random_region_length: int | None = None
    train_path: str | None = None
    pool_path: str | None = None
    val_path: str | None = None
    test_set: dict[str, Any] = field(default_factory=dict)
    flanking_sequence: dict[str, str] = field(default_factory=dict)
    oracle_checkpoint: str | None = None
    sequence_filters: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
