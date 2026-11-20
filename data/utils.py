"""Sequence encoding utilities for the dataset loaders.

This module used to hold a second, independent implementation of the five primitives in
:mod:`albench.utils` -- one-hot encoding, reverse complement, validation, padding and batch
encoding. The two agreed on every input tested (204 sequences covering N, lowercase, empty
and three lengths, 2026-11-20), but two copies of an encoder is the shape of bug this repo
has already been bitten by: a silent divergence in how a base is encoded would move every
model's input without changing any caller.

One implementation now, in ``albench.utils``, re-exported here under the names this package
has always used. The dependency runs this way round because ``albench`` is meant to install
stand-alone, so it may not import from ``data``.
"""

from __future__ import annotations

from albench.utils import (
    _COMPLEMENT_MAP as COMPLEMENT_MAP,
)
from albench.utils import (
    _NUCLEOTIDE_TO_INDEX as NUCLEOTIDE_TO_INDEX,
)
from albench.utils import (
    batch_encode_sequences,
    one_hot_encode,
    pad_sequence,
    reverse_complement,
    reverse_complement_one_hot,
    validate_sequence,
)

__all__ = [
    "COMPLEMENT_MAP",
    "NUCLEOTIDE_TO_INDEX",
    "batch_encode_sequences",
    "one_hot_encode",
    "pad_sequence",
    "reverse_complement",
    "reverse_complement_one_hot",
    "validate_sequence",
]
