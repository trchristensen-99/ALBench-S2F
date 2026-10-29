"""Give a student real plasmid flank as input, and shift its window over real bases, instead of seeing the 200-bp insert alone.

Why: the oracle labels every sequence inside the plasmid (600 bp), so a motif that spans the insert boundary, or a boundary artefact, is part of the label, while a student that sees 200 bp with zeros
beyond it cannot represent that. ``wide_onehot`` encodes insert + (F + S) real bases per side; ``WideCrop`` wraps any architecture built for ``200 + 2F`` bases, crops the central ``200 + 2F`` window at
test time and a window shifted by up to +-S real bases at training time (the same augmentation as the zero-padded shift, but on real sequence). F = 0 gives the shift-on-real-bases arm alone.
"""

from __future__ import annotations

import random

import numpy as np

from albench.arch_search.plasmid_flank import DOWNSTREAM_FULL, UPSTREAM_FULL
from albench.arch_search.train_generated import one_hot

INSERT = 200


def wide_onehot(seqs, margin: int) -> np.ndarray:
    """(N, 4, 200 + 2*margin) ACGT one-hot of ``UPSTREAM[-margin:] + insert + DOWNSTREAM[:margin]``."""
    if margin > len(UPSTREAM_FULL):
        raise ValueError(f"margin {margin} exceeds the {len(UPSTREAM_FULL)} stored flank bases")
    X = one_hot(seqs, INSERT)
    if margin == 0:
        return X
    up, dn = (one_hot([s], len(s))[0] for s in (UPSTREAM_FULL[-margin:], DOWNSTREAM_FULL[:margin]))
    n = len(X)
    return np.concatenate(
        [np.broadcast_to(up, (n, 4, margin)), X, np.broadcast_to(dn, (n, 4, margin))], axis=2
    )


def _crop_module():
    import torch.nn as nn

    class WideCrop(nn.Module):
        """Feeds ``net`` a ``200 + 2*ctx`` window of a wider input; random shift in +-shift while training, centred otherwise."""

        def __init__(self, net, ctx: int, shift: int):
            super().__init__()
            self.net, self.ctx, self.shift = net, int(ctx), int(shift)
            self.width = INSERT + 2 * self.ctx

        def forward(self, x):
            lo = (x.shape[-1] - self.width) // 2
            if self.training and self.shift > 0:
                lo += random.randint(-self.shift, self.shift)
            return self.net(x[:, :, lo : lo + self.width])

    return WideCrop


def wrap(net, ctx: int, shift: int):
    return _crop_module()(net, ctx, shift)
