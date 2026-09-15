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


def test_generator_choice_is_not_reliably_nested():
    """Documents WHY the driver uses a permutation prefix.

    numpy's Generator.choice(replace=False) switches algorithm with the D/N ratio, so
    it is nested in some size regimes and not others. Relying on it would give
    scaling curves that are smooth at some sizes and jagged at others, which is worse
    than uniformly jagged because the artefact looks like a result.
    """
    violations = 0
    for n in (5_000, 50_000):
        for seed in (0, 42):
            prev = None
            for d in (100, 500, 2_000):
                cur = set(np.random.default_rng(seed).choice(n, size=d, replace=False).tolist())
                if prev is not None and not prev <= cur:
                    violations += 1
                prev = cur
    assert violations > 0, (
        "choice(replace=False) appears nested here; if numpy changed this, the "
        "permutation-prefix rationale in scaling_hp_search.py should be revisited"
    )


def test_permutation_prefix_is_always_nested():
    """The property the driver now relies on."""
    for n in (5_000, 50_000, 300_000):
        for seed in (0, 1, 42):
            prev = None
            for d in (100, 500, 2_000, 10_000):
                cur = set(np.random.default_rng(seed).permutation(n)[:d].tolist())
                if prev is not None:
                    assert prev <= cur, f"not nested at n={n} seed={seed} d={d}"
                prev = cur
