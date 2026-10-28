"""Package A: data contract tests (spec T1 and the encoding rules of 2.1)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from a_helpers import synth_data, write_pool

from ar2.data.encode import SequenceError, encode, revcomp, revcomp_onehot
from ar2.data.loader import LabelScaler, LeakageError, load_all, load_split, verify_no_overlap
from ar2.data.manifest import ManifestError, RoleError, make_manifest, save_manifest


def test_encode_rules():
    x = encode(["acgt" * 50, "N" * 200])
    assert x.shape == (2, 4, 200) and x.dtype == np.float32
    assert np.array_equal(x[0, :, :4].T, np.eye(4))  # upper-cased, channel order A, C, G, T
    assert np.all(x[1] == 0.25)  # an N is 0.25 in all four channels
    s = "ACGGT" * 40
    assert np.array_equal(revcomp_onehot(encode([s]))[0], encode([revcomp(s)])[0])


def test_t1_loader_rejects_wrong_length(tmp_path):
    with pytest.raises(SequenceError, match="199"):
        encode(["A" * 199])
    seqs, y, *_ = synth_data(20, 5)
    seqs[3] = seqs[3][:199]
    mp = write_pool(tmp_path, seqs, y, "train", "bad")
    with pytest.raises(SequenceError):
        load_split(mp, "train")


def test_t1_manifest_rejects_test_role_and_test_paths(tmp_path):
    seqs, y, *_ = synth_data(20, 5)
    npz = tmp_path / "x.npz"
    np.savez(npz, sequences=np.asarray(seqs, dtype=object), oracle_labels=y)
    bad = make_manifest(str(npz), "test")
    with pytest.raises(RoleError):
        save_manifest(tmp_path / "m.json", bad)
    (tmp_path / "m2.json").write_text(json.dumps(bad))
    with pytest.raises(RoleError):
        load_split(tmp_path / "m2.json")
    leaky_path = make_manifest(str(tmp_path / "test_sets" / "genomic_oracle.npz"), "val_loop")
    with pytest.raises(ManifestError, match="test"):
        save_manifest(tmp_path / "m3.json", leaky_path)
    good = write_pool(tmp_path, seqs, y, "train", "good")
    with pytest.raises(ManifestError, match="expected role"):
        load_split(good, "val_loop")


def test_t1_overlap_including_reverse_complement(tmp_path):
    tr, ytr, va, yva = synth_data(50, 30)
    verify_no_overlap(tr, va)  # clean
    with pytest.raises(LeakageError, match="1 sequences"):
        verify_no_overlap(tr, va[:-1] + [tr[7]])
    with pytest.raises(LeakageError):
        verify_no_overlap(tr, va[:-1] + [revcomp(tr[7])])  # reverse complement counts
    tm = write_pool(tmp_path, tr, ytr, "train", "tr")
    vm = write_pool(tmp_path, va[:-1] + [revcomp(tr[0])], yva, "val_loop", "va")
    with pytest.raises(LeakageError):
        load_all(tm, vm)


def test_index_file_and_prefix(tmp_path):
    seqs, y, *_ = synth_data(100, 5)
    npz = tmp_path / "big.npz"
    np.savez(npz, sequences=np.asarray(seqs, dtype=object), oracle_labels=y)
    m = make_manifest(str(npz), "train", D=40)
    assert len(load_split(m)) == 40 and load_split(m).seqs == [
        s.upper() for s in seqs[:40]
    ]  # first D rows
    np.save(tmp_path / "idx.npy", np.array([5, 3, 9]))
    m2 = make_manifest(str(npz), "val_select", index_file=str(tmp_path / "idx.npy"))
    sp = load_split(m2)
    assert sp.seqs == [seqs[5], seqs[3], seqs[9]] and sp.labels.shape == (3, 1)


def test_label_scaler_uses_train_statistics():
    y = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
    sc = LabelScaler.fit(y)
    z = sc.transform(y)
    assert abs(z.mean()) < 1e-6 and abs(z.std() - 1) < 1e-6
    assert np.allclose(sc.inverse(z), y, atol=1e-5)
