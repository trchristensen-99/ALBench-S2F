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
    import torch
    import torch.nn as nn

    class WideCrop(nn.Module):
        """Feeds ``net`` a ``200 + 2*ctx`` window of a wider input.

        Training: a random shift in +-shift real bases, the same for the whole batch (default) or drawn per sample (``per_sample``), applied with probability ``prob`` (the oracle shifts +-15 on half of its
        samples). Evaluation: the centred window, or, with ``tta`` = k > 0, the mean prediction over the offsets -k, -k/2, 0, k/2, k (needs k <= shift).
        """

        def __init__(
            self,
            net,
            ctx: int,
            shift: int,
            per_sample: bool = False,
            prob: float = 1.0,
            tta: int = 0,
        ):
            super().__init__()
            self.net, self.ctx, self.shift = net, int(ctx), int(shift)
            self.per_sample, self.prob, self.tta = bool(per_sample), float(prob), int(tta)
            if self.tta > self.shift:
                raise ValueError("tta offsets must stay inside the shift margin")
            self.width = INSERT + 2 * self.ctx

        def _window(self, x, offset):
            lo = (x.shape[-1] - self.width) // 2 + offset
            return x[:, :, lo : lo + self.width]

        def forward(self, x):
            if self.training and self.shift > 0:
                if self.per_sample:
                    b = x.shape[0]
                    off = torch.randint(-self.shift, self.shift + 1, (b,), device=x.device)
                    if self.prob < 1.0:
                        off = torch.where(
                            torch.rand(b, device=x.device) < self.prob, off, torch.zeros_like(off)
                        )
                    lo = (x.shape[-1] - self.width) // 2 + off
                    idx = lo[:, None] + torch.arange(self.width, device=x.device)[None, :]
                    return self.net(x.gather(2, idx[:, None, :].expand(-1, x.shape[1], -1)))
                off = random.randint(-self.shift, self.shift) if random.random() < self.prob else 0
                return self.net(self._window(x, off))
            if not self.training and self.tta > 0:
                k = self.tta
                outs = [self.net(self._window(x, o)) for o in sorted({-k, -k // 2, 0, k // 2, k})]
                return sum(outs) / len(outs)
            return self.net(self._window(x, 0))

    return WideCrop


def wrap(net, ctx: int, shift: int, per_sample: bool = False, prob: float = 1.0, tta: int = 0):
    return _crop_module()(net, ctx, shift, per_sample, prob, tta)
