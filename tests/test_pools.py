"""Pool labelling and nested subsetting."""

from __future__ import annotations

import numpy as np
import pytest

from albench.pools import (
    PoolSpec,
    draw_training_set,
    load_pool,
    nested_subsets,
    verify_nesting,
    write_pool,
)


def _pool(n=500, seed=0):
    rng = np.random.default_rng(seed)
    seqs = ["".join(rng.choice(list("ACGT"), size=20)) for _ in range(n)]
    return seqs, rng.normal(size=n).astype(np.float32)


def test_pool_id_is_stable_and_parameter_sensitive():
    a = PoolSpec("zoonomia", {"ti_tv": 2.0})
    b = PoolSpec("zoonomia", {"ti_tv": 2.0})
    c = PoolSpec("zoonomia", {"ti_tv": 4.0})
    d = PoolSpec("zoonomia", {"ti_tv": 2.0}, generation_seed=1)
    assert a.pool_id == b.pool_id
    assert a.pool_id != c.pool_id, "different params must not share a pool id"
    assert a.pool_id != d.pool_id, "different generation seeds must not share a pool id"


def test_write_and_load_roundtrip(tmp_path):
    seqs, lab = _pool()
    spec = PoolSpec("zoonomia", {"ti_tv": 2.0}, size=len(seqs))
    p = write_pool(tmp_path / "x.npz", seqs, lab, spec, oracle_id="full856k_clean")
    got = load_pool(p)
    assert got["sequences"] == seqs
    assert np.allclose(got["labels"], lab)
    assert got["oracle_id"] == "full856k_clean"
    assert got["params"] == {"ti_tv": 2.0}


def test_write_rejects_mismatched_lengths(tmp_path):
    seqs, lab = _pool()
    with pytest.raises(ValueError, match="labels"):
        write_pool(tmp_path / "x.npz", seqs, lab[:-1], PoolSpec("z"), "oid")


def test_write_rejects_nonfinite_labels(tmp_path):
    """A NaN label would poison training silently; refuse at write time."""
    seqs, lab = _pool()
    lab[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        write_pool(tmp_path / "x.npz", seqs, lab, PoolSpec("z"), "oid")


def test_load_rejects_a_pool_without_provenance(tmp_path):
    p = tmp_path / "bare.npz"
    np.savez(p, sequences=np.array(["AC"], dtype=object), oracle_labels=np.zeros(1))
    with pytest.raises(ValueError, match="missing"):
        load_pool(p)


def test_subsets_are_nested():
    sub = nested_subsets(10_000, [1_000, 3_000, 10_000], seed=0)
    verify_nesting(sub)
    assert set(sub[1_000]) < set(sub[3_000]) < set(sub[10_000])


def test_nesting_check_catches_a_violation():
    bad = {10: np.arange(10), 20: np.arange(5, 25)}
    with pytest.raises(AssertionError, match="not contained"):
        verify_nesting(bad)


def test_subset_larger_than_pool_is_refused():
    with pytest.raises(ValueError, match="exceeds the pool"):
        nested_subsets(100, [500])


def test_different_seeds_give_different_draws():
    a = set(nested_subsets(1000, [100], seed=0)[100].tolist())
    b = set(nested_subsets(1000, [100], seed=1)[100].tolist())
    assert a != b
    assert len(a & b) < len(a)  # overlap, but not identical


def test_draw_training_set_nests_across_the_curve(tmp_path):
    seqs, lab = _pool(n=1000)
    spec = PoolSpec("z", size=1000)
    pool = load_pool(write_pool(tmp_path / "p.npz", seqs, lab, spec, "oid"))
    curve = [100, 300, 1000]
    got = {}
    for size in curve:
        s, y, idx = draw_training_set(pool, size, seed=0, sizes_for_nesting=curve)
        assert len(s) == size == len(y)
        got[size] = set(idx.tolist())
    assert got[100] < got[300] < got[1000]


def test_labels_follow_their_sequences(tmp_path):
    """The subset must not shuffle labels relative to sequences."""
    seqs = [f"SEQ{i}" for i in range(200)]
    lab = np.arange(200, dtype=np.float32)  # label i encodes index i
    spec = PoolSpec("z", size=200)
    pool = load_pool(write_pool(tmp_path / "p.npz", seqs, lab, spec, "oid"))
    s, y, idx = draw_training_set(pool, 50, seed=2)
    for seq, val in zip(s, y):
        assert seq == f"SEQ{int(val)}"
