"""General-purpose sequence utilities for albench.

These functions are self-contained — they depend only on ``numpy`` and the
Python standard library, so ``albench`` can be installed stand-alone without
any other package in this repository.
"""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_NUCLEOTIDE_TO_INDEX: Dict[str, int] = {
    "A": 0,
    "a": 0,
    "C": 1,
    "c": 1,
    "G": 2,
    "g": 2,
    "T": 3,
    "t": 3,
}

_COMPLEMENT_MAP: Dict[str, str] = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C",
    "a": "t",
    "t": "a",
    "c": "g",
    "g": "c",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def one_hot_encode(
    sequence: str,
    num_channels: int = 4,
    add_singleton_channel: bool = False,
    is_singleton: bool = False,
) -> np.ndarray:
    """One-hot encode a DNA sequence.

    Args:
        sequence: DNA sequence string (ACGT).
        num_channels: Number of channels (4 for ACGT).
        add_singleton_channel: Whether to add a singleton indicator channel.
        is_singleton: Whether this sequence is a singleton.

    Returns:
        One-hot encoded array of shape ``(num_channels, sequence_length)``.
    """
    sequence = sequence.upper()
    seq_length = len(sequence)
    n_ch = 5 if add_singleton_channel else 4
    encoded = np.zeros((n_ch, seq_length), dtype=np.float32)
    for i, nuc in enumerate(sequence):
        if nuc in _NUCLEOTIDE_TO_INDEX:
            encoded[_NUCLEOTIDE_TO_INDEX[nuc], i] = 1.0
    if add_singleton_channel:
        encoded[4, :] = 1.0 if is_singleton else 0.0
    return encoded


def reverse_complement(sequence: str) -> str:
    """Compute the reverse complement of a DNA sequence.

    Args:
        sequence: DNA sequence string.

    Returns:
        Reverse complement of the input sequence.

    Examples:
        >>> reverse_complement("ACGT")
        'ACGT'
        >>> reverse_complement("AAAA")
        'TTTT'
    """
    complement = "".join(_COMPLEMENT_MAP.get(base, base) for base in sequence)
    return complement[::-1]


def reverse_complement_one_hot(encoded: np.ndarray) -> np.ndarray:
    """Compute reverse complement of a one-hot encoded sequence.

    Args:
        encoded: One-hot encoded array of shape ``(num_channels, seq_length)``.
            First 4 channels are ACGT; optional 5th channel is metadata.

    Returns:
        Reverse complement of the encoded sequence.
    """
    rc = encoded[:, ::-1].copy()
    rc[[0, 1, 2, 3], :] = rc[[3, 2, 1, 0], :]
    return rc


def validate_sequence(sequence: str, allowed_chars: str = "ACGTN") -> bool:
    """Return ``True`` if *sequence* contains only allowed characters."""
    allowed = set(allowed_chars.upper() + allowed_chars.lower())
    return all(c in allowed for c in sequence)


def pad_sequence(
    sequence: str,
    target_length: int,
    pad_char: str = "N",
    mode: str = "both",
) -> str:
    """Pad a sequence to *target_length*.

    Args:
        sequence: DNA sequence to pad.
        target_length: Desired final length.
        pad_char: Character to use for padding (default ``'N'``).
        mode: ``'left'``, ``'right'``, or ``'both'`` (centred).

    Raises:
        ValueError: If sequence is already longer than *target_length*.
    """
    if len(sequence) >= target_length:
        if len(sequence) == target_length:
            return sequence
        raise ValueError(f"Sequence length {len(sequence)} exceeds target length {target_length}")
    pad_needed = target_length - len(sequence)
    if mode == "left":
        return pad_char * pad_needed + sequence
    elif mode == "right":
        return sequence + pad_char * pad_needed
    elif mode == "both":
        left = pad_needed // 2
        return pad_char * left + sequence + pad_char * (pad_needed - left)
    raise ValueError(f"Unknown padding mode: {mode!r}")


def batch_encode_sequences(
    sequences: List[str],
    add_singleton_channel: bool = False,
    is_singleton_flags: Union[List[bool], None] = None,
) -> np.ndarray:
    """Batch one-hot encode a list of same-length DNA sequences.

    Returns:
        Array of shape ``(N, num_channels, seq_length)``.

    Raises:
        ValueError: If sequences is empty or sequences have different lengths.
    """
    if not sequences:
        raise ValueError("Cannot encode an empty sequence list.")
    lengths = {len(s) for s in sequences}
    if len(lengths) > 1:
        raise ValueError(f"All sequences must have the same length; got {lengths}.")
    if add_singleton_channel and is_singleton_flags is None:
        is_singleton_flags = [False] * len(sequences)
    encoded_list = []
    for i, seq in enumerate(sequences):
        flag = bool(is_singleton_flags[i]) if is_singleton_flags else False
        encoded_list.append(
            one_hot_encode(seq, add_singleton_channel=add_singleton_channel, is_singleton=flag)
        )
    return np.stack(encoded_list, axis=0)
