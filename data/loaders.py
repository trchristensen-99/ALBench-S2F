"""Dataset loader compatibility module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from data.k562 import K562Dataset
from data.yeast import YeastDataset


def load_fasta_sequences(path: str) -> list[str]:
    """Load DNA sequences from a FASTA file."""
    sequences: list[str] = []
    current: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
                continue
            current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences


def load_hdf5_dataset(
    path: str, sequence_key: str = "sequences", label_key: str = "labels"
) -> dict[str, Any]:
    """Load sequence/label arrays from HDF5.

    Requires ``h5py`` in the active environment.
    """
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("h5py is required for HDF5 loading") from exc

    with h5py.File(path, "r") as handle:
        sequences = np.asarray(handle[sequence_key])
        labels = np.asarray(handle[label_key])
    return {"sequences": sequences, "labels": labels}


__all__ = ["K562Dataset", "YeastDataset", "load_fasta_sequences", "load_hdf5_dataset"]
