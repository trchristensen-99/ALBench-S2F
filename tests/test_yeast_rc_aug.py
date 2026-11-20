"""The yeast reverse-complement augmentation encodes the strand it says it does.

The DREAM-RNN encoding reserves channel 4 for a reverse-complement flag (data/yeast.py)
and nothing ever set it, so the oracle trained on one strand only. These tests pin the
two things that can silently go wrong once it is set: complementing the wrong channel
axis (ACGT reversed is the complement; reversing the length axis is the reverse), and
forgetting to raise the flag.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.train_dream_rnn_oracle import rc_batch  # noqa: E402

A, C, G, T = 0, 1, 2, 3


def _encode(seq: str) -> torch.Tensor:
    """(1, 6, L): ACGT one-hot, a zero RC flag and a zero singleton flag."""
    idx = {"A": A, "C": C, "G": G, "T": T}
    x = np.zeros((1, 6, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq):
        x[0, idx[b], i] = 1.0
    return torch.from_numpy(x)


def test_rc_batch_is_the_reverse_complement():
    x = _encode("ACGTTA")
    out = rc_batch(x)
    assert torch.equal(out[:, :4], _encode("TAACGT")[:, :4])


def test_rc_batch_sets_the_flag_and_is_an_involution_on_the_bases():
    x = _encode("ACGTTA")
    once = rc_batch(x)
    assert torch.all(once[:, 4] == 1.0)
    # the bases come back; the flag does not, by design -- it records the strand shown
    assert torch.equal(rc_batch(once)[:, :4], x[:, :4])


def test_rc_batch_leaves_the_singleton_channel_constant():
    x = _encode("ACGTTA")
    x[:, 5] = 1.0
    assert torch.all(rc_batch(x)[:, 5] == 1.0)
