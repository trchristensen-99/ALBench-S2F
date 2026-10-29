"""Flank-context encoding and the window-shift wrapper: the central window must equal the insert, a shift must take real plasmid bases, and F = 0 must reproduce the AL_FLANK_SHIFT=1 shift."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from albench.arch_search.flank_context import INSERT, wide_onehot, wrap  # noqa: E402
from albench.arch_search.plasmid_flank import DOWNSTREAM_FULL, UPSTREAM_FULL  # noqa: E402
from albench.arch_search.train_generated import one_hot  # noqa: E402

SEQS = ["ACGT" * 50, "TTGCA" * 40, "G" * 200]


def test_constants_are_the_plasmid_next_to_the_insert():
    assert len(UPSTREAM_FULL) == len(DOWNSTREAM_FULL) == 300
    assert UPSTREAM_FULL.endswith("CTGGCCGCTTGACG") and DOWNSTREAM_FULL.startswith("CACTGCGGCTCC")


@pytest.mark.parametrize("margin", [0, 1, 20, 130])
def test_wide_onehot_layout(margin):
    X = wide_onehot(SEQS, margin)
    assert X.shape == (3, 4, INSERT + 2 * margin) and X.dtype == np.float32
    assert (X.sum(1) == 1).all()  # every position is exactly one base
    assert np.array_equal(X[:, :, margin : margin + INSERT], one_hot(SEQS, INSERT))
    if margin:
        assert np.array_equal(X[0, :, :margin], one_hot([UPSTREAM_FULL[-margin:]], margin)[0])
        assert np.array_equal(X[0, :, -margin:], one_hot([DOWNSTREAM_FULL[:margin]], margin)[0])


def test_wrapper_centre_at_eval_and_shift_on_real_bases_at_train():
    ctx, shift = 5, 3
    seen = []

    class Probe(torch.nn.Module):
        def forward(self, x):
            seen.append(x.detach().clone())
            return x.sum((1, 2))

    X = torch.from_numpy(wide_onehot(SEQS, ctx + shift))
    net = wrap(Probe(), ctx, shift).eval()
    net(X)
    assert torch.equal(seen[-1], torch.from_numpy(wide_onehot(SEQS, ctx)))
    net.train()
    offsets = set()
    for _ in range(200):
        net(X)
        w = seen[-1]
        for off in range(-shift, shift + 1):
            lo = shift + off
            if torch.equal(w, X[:, :, lo : lo + INSERT + 2 * ctx]):
                offsets.add(off)
    assert offsets == set(range(-shift, shift + 1))


def test_ctx0_wrapper_matches_flank_shift_of_the_harness(monkeypatch):
    from albench.arch_search import arch_autoresearch as aa

    if not hasattr(aa, "_flank"):
        pytest.skip("harness without AL_FLANK_SHIFT")
    monkeypatch.setenv("AL_FLANK_SHIFT", "1")
    x = torch.from_numpy(one_hot(SEQS, INSERT))
    wide = torch.from_numpy(wide_onehot(SEQS, 10))
    for s in (-10, -3, 1, 7, 10):
        monkeypatch.setattr(aa.random, "randint", lambda a, b, s=s: s)
        ref = aa.shift(x, 10)
        # harness shift s moves the content right by s; the wrapper crop at -s sees the same window
        monkeypatch.setattr(
            "albench.arch_search.flank_context.random.randint", lambda a, b, s=s: -s
        )
        got = wrap(torch.nn.Identity(), 0, 10).train()(wide)
        assert torch.equal(ref, got), s


def test_per_sample_shift_and_tta():
    ctx, shift = 4, 6
    X = torch.from_numpy(wide_onehot(SEQS * 20, ctx + shift))
    seen = []

    class Probe(torch.nn.Module):
        def forward(self, x):
            seen.append(x.detach().clone())
            return x.sum((1, 2))

    net = wrap(Probe(), ctx, shift, per_sample=True, prob=0.5).train()
    centred = torch.from_numpy(wide_onehot(SEQS * 20, ctx))
    n_same, n_shift = 0, 0
    for _ in range(40):
        net(X)
        w = seen[-1]
        same = (w == centred).all(dim=(1, 2))
        n_same += int(same.sum())
        n_shift += int((~same).sum())
        # every sample is a real window of the wide input at some offset in +-shift
        for i in range(0, w.shape[0], 7):
            assert any(
                torch.equal(w[i], X[i, :, shift + o : shift + o + INSERT + 2 * ctx])
                for o in range(-shift, shift + 1)
            )
    assert n_same > 0.4 * (n_same + n_shift) and n_shift > 0.2 * (
        n_same + n_shift
    )  # about half are unshifted (plus chance offset 0)
    tt = wrap(Probe(), ctx, shift, tta=4).eval()
    seen.clear()
    out = tt(X)
    assert len(seen) == 5 and out.shape[0] == X.shape[0]  # offsets -4, -2, 0, 2, 4
    with pytest.raises(ValueError):
        wrap(Probe(), ctx, shift, tta=8)
