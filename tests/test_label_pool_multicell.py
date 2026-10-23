"""Routing and encoding contracts of scripts/label_pool_multicell.py (no GPU, no oracle needed).

The encoder is checked against an independent re-statement of the two code paths it must match:
the K562 live loader (`_AGS2Oracle._encode`, exp1_1_scaling.py) for pad_mode=legacy, and the
training collate (train_oracle_s2_v2.py) for pad_mode=context. A silent drift in either would
shift every HepG2 label relative to the K562 labels it is subtracted from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import label_pool_multicell as m  # noqa: E402

MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def _onehot(seq: str) -> np.ndarray:
    out = np.zeros((len(seq), 4), dtype=np.float32)
    for i, c in enumerate(seq):
        if c in MAP:
            out[i, MAP[c]] = 1.0
    return out


def _legacy_reference(seq: str) -> np.ndarray:
    """Verbatim logic of _AGS2Oracle._encode: centred N pad / centre crop, 200 bp flanks."""
    from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM

    seq = seq.upper()
    if len(seq) < 200:
        pad = 200 - len(seq)
        seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
    elif len(seq) > 200:
        start = (len(seq) - 200) // 2
        seq = seq[start : start + 200]
    return np.concatenate(
        [_onehot(MPRA_UPSTREAM[-200:]), _onehot(seq), _onehot(MPRA_DOWNSTREAM[:200])]
    )


def _context_reference(seq: str) -> np.ndarray:
    """Logic of train_oracle_s2_v2.py collate(): centred 600 bp slice of UP[300]+core+DOWN[300]."""
    from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM

    core = _onehot(seq.upper()[:200])
    wide = np.concatenate([_onehot(MPRA_UPSTREAM[-300:]), core, _onehot(MPRA_DOWNSTREAM[:300])])
    start = (wide.shape[0] - 600) // 2
    return wide[start : start + 600]


def _decode(idx: np.ndarray) -> np.ndarray:
    oh = np.zeros((idx.shape[0], 4), dtype=np.float32)
    ok = idx < 4
    oh[np.where(ok)[0], idx[ok]] = 1.0
    return oh


@pytest.fixture(scope="module")
def seqs():
    rng = np.random.default_rng(0)
    out = []
    for n in (200, 200, 199, 180, 163, 201, 230, 150):
        s = "".join(rng.choice(list("ACGT"), n))
        out.append(s)
    out.append("ACGTN" * 40)  # N inside the core must stay all-zero
    return out


def test_legacy_encoding_matches_k562_live_loader(seqs):
    enc = m.encode_indices(seqs, "legacy")
    for s, row in zip(seqs, enc):
        np.testing.assert_array_equal(_decode(row), _legacy_reference(s))


def test_context_encoding_matches_training_collate(seqs):
    enc = m.encode_indices(seqs, "context")
    for s, row in zip(seqs, enc):
        np.testing.assert_array_equal(_decode(row), _context_reference(s))


def test_modes_agree_exactly_for_200bp():
    """Generated pools are all 200 bp, so legacy vs context must not change a single label there."""
    s = ["ACGT" * 50, "TTGCA" * 40]
    np.testing.assert_array_equal(m.encode_indices(s, "legacy"), m.encode_indices(s, "context"))


def test_reverse_complement_identity():
    """The device-side RC (3 - idx, reversed) must equal the one-hot [::-1, ::-1] of the K562 loader."""
    s = ["".join(np.random.default_rng(1).choice(list("ACGT"), 200))]
    b = m.encode_indices(s, "legacy")
    rc = np.where(b < 4, 3 - b, b)[:, ::-1]
    np.testing.assert_array_equal(_decode(rc[0]), _decode(b[0])[::-1, ::-1])


def test_weights_for_mask():
    w, code = m._weights_for_mask(1 << 3)  # sits in fold 3 -> its test-fold model, nothing else
    assert code == 1 and w[3] == 1.0 and w.sum() == 1.0
    w, code = m._weights_for_mask(
        (1 << 3) | (1 << 4)
    )  # copies in folds 3 and 4 -> model 3 (test 3, val 4)
    assert code == 2 and w[3] == 1.0 and w.sum() == 1.0
    w, code = m._weights_for_mask((1 << 9) | (1 << 0))  # wrap-around: model 9 has val fold 0
    assert code == 2 and w[9] == 1.0
    w, code = m._weights_for_mask((1 << 2) | (1 << 6))  # no model is clean for both copies
    assert code == 3 and np.allclose(w, 0.1)


def test_route_modes_and_unseen_rows():
    masks = np.array([0, 1 << 2, 1 << 5, 1 << 7], dtype=np.int64)
    first = np.array([-1, 10, 20, 900_000])
    W, src = m.route(masks, first, "ensemble", None)
    assert (src == 0).all() and np.allclose(W, 0.1)
    W, src = m.route(masks, first, "oof", 833_290)
    assert src.tolist() == [
        0,
        1,
        1,
        4,
    ]  # not-in-pool, OOF, OOF, pool row the cell's oracle never saw
    assert W[1, 2] == 1.0 and W[2, 5] == 1.0
    assert np.allclose(W[0], 0.1) and np.allclose(
        W[3], 0.1
    )  # unseen/synthetic -> every fold is clean
    assert np.allclose(W.sum(1), 1.0)
