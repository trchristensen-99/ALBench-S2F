"""One-hot encoding (spec 2.1): float32, shape (B, 4, 200), channels A, C, G, T; an N is 0.25 in all four channels."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

SEQ_LEN = 200
ALPHABET = "ACGT"
_LUT = np.full(256, 4, dtype=np.uint8)  # 4 = N / anything else
for _i, _c in enumerate(ALPHABET):
    _LUT[ord(_c)] = _i
    _LUT[ord(_c.lower())] = _i
_COMP = str.maketrans("ACGTacgtN", "TGCAtgcaN")


class SequenceError(ValueError):
    """A sequence that breaks the data contract (length or alphabet)."""


def check_sequences(seqs: Sequence[str], seq_len: int = SEQ_LEN) -> None:
    """Raise :class:`SequenceError` unless every sequence has exactly ``seq_len`` characters from A, C, G, T, N."""
    for i, s in enumerate(seqs):
        if len(s) != seq_len:
            raise SequenceError(f"sequence {i} has length {len(s)}, expected exactly {seq_len}")
    if len(seqs):
        joined = "".join(seqs).encode("ascii", errors="replace")
        arr = np.frombuffer(joined, dtype=np.uint8)
        bad = ~np.isin(arr, np.frombuffer(b"ACGTNacgtn", dtype=np.uint8))
        if bad.any():
            where = int(np.argmax(bad)) // seq_len
            raise SequenceError(f"sequence {where} contains characters outside A, C, G, T, N")


def encode(seqs: Sequence[str], seq_len: int = SEQ_LEN) -> np.ndarray:
    """(B, 4, seq_len) float32 one-hot; upper-cases; rejects any other length."""
    check_sequences(seqs, seq_len)
    if len(seqs) == 0:
        return np.zeros((0, 4, seq_len), np.float32)
    codes = _LUT[np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8)].reshape(
        len(seqs), seq_len
    )
    x = np.zeros((len(seqs), 4, seq_len), np.float32)
    for c in range(4):
        x[:, c, :] = codes == c
    x[(codes == 4)[:, None, :].repeat(4, axis=1)] = 0.25
    return x


def revcomp(seq: str) -> str:
    """Reverse complement of one sequence (N stays N)."""
    return seq.upper().translate(_COMP)[::-1]


def revcomp_onehot(x):
    """Reverse complement of one-hot (..., 4, L) numpy array or torch tensor: flip the length axis and swap A<->T, C<->G."""
    return (
        x.flip(-1)[..., [3, 2, 1, 0], :]
        if hasattr(x, "flip")
        else x[..., ::-1][..., [3, 2, 1, 0], :]
    )


def gc_fraction(seqs: Sequence[str]) -> np.ndarray:
    """GC fraction of each sequence."""
    return np.array([(s.upper().count("G") + s.upper().count("C")) / max(len(s), 1) for s in seqs])
