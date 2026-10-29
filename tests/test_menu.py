"""The menu of configurations and the ensemble built from it (albench.ensemble.menu)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from albench.arch_search.harness_standard import FLANK_CTX, INPUT_LEN, SHIFT_MAX
from albench.ensemble import menu as menu_mod
from albench.ensemble.menu import (
    DEFAULT_MENU,
    Menu,
    MenuEnsemble,
    MenuEntry,
    TrainedMember,
    build_net,
    encode,
    encode_train,
    evaluate,
    exploded,
    input_length,
    load_labelled,
    load_menu,
    load_test_sets,
    member_predict,
    one_hot,
    pearson,
    probe_sequences,
    stack_dir,
    train_member,
    train_menu,
)
from albench.ensemble.stack import ZStack, fit_zscored_stack


def _toy_entries() -> list[MenuEntry]:
    src = "class Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.c = nn.Conv1d(4, 8, 7, padding=3)\n        self.f = nn.Linear(8, 1)\n    def forward(self, x):\n        return self.f(F.silu(self.c(x)).mean(-1)).squeeze(-1)\n"
    tc = {
        "optimizer": "adamw",
        "lr": 0.01,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "epochs": 3,
        "lr_schedule": "onecycle",
    }
    runs = ["a", "a", "a", "b", "b", "ref"]
    vals = [0.9, 0.8, 0.7, 0.85, 0.75, None]
    return [
        MenuEntry(
            f"e{i}",
            r,
            "",
            v,
            None,
            src.replace("self.c =", f"self.c{i} = nn.Identity(); self.c =") if False else src,
            dict(tc),
        )
        for i, (r, v) in enumerate(zip(runs, vals))
    ]


# ------------------------------------------------------------------------------------------------------------- shipped menu
def test_shipped_menu_loads_and_is_consistent():
    m = load_menu("v0")
    assert len(m) == 33
    assert len({e.id for e in m.entries}) == 33
    assert all("class Net" in e.source and e.train_config["epochs"] >= 1 for e in m.entries)
    assert sum(e.search_val is None for e in m.entries) == 1  # the reference network


def test_priority_order_covers_each_usable_entry_once_and_per_size_table_is_valid():
    m = load_menu("v0")
    order = m.priority()
    usable = sorted(e.id for e in m.entries if e.id not in m.skipped)
    assert sorted(e.id for e in order) == usable and len(m.skipped) == 3
    for size in (10_000, 30_000, 100_000):
        row = m.recipe(size)
        assert 1 <= row["k"] <= len(usable) and row["seeds"] >= 1
        assert len(m.for_size(size)) == row["k"]
        assert not {e.id for e in m.for_size(size)} & set(m.skipped)
    assert m.recipe(5_000) == m.recipe(10_000)  # below the table: the smallest size's row
    assert m.recipe(60_000) == m.recipe(30_000)  # between: the nearest size below


# ------------------------------------------------------------------------------------------------------------------ ordering
def test_first_n_rules_put_the_best_of_every_run_first():
    m = Menu(_toy_entries())
    assert [e.id for e in m.ordered("val")] == ["e0", "e3", "e1", "e4", "e2", "e5"]
    # best of every run first (a: e0, b: e3, ref: e5 has no search val so it sorts last within the head), then the rest by value
    assert [e.id for e in m.ordered("first1")] == ["e0", "e3", "e5", "e1", "e4", "e2"]
    assert [e.id for e in m.ordered("first2")][:5] == ["e0", "e3", "e5", "e1", "e4"]
    with pytest.raises(ValueError):
        m.ordered("nonsense")
    assert [e.id for e in m.ordered("given")] == [f"e{i}" for i in range(6)]


def test_duplicate_ids_are_rejected():
    e = _toy_entries()
    with pytest.raises(ValueError):
        Menu([e[0], e[0]])


def test_menu_json_round_trip():
    m = load_menu("v0")
    m2 = Menu.from_json(json.loads(json.dumps(m.to_json())))
    assert [e.source_sha1 for e in m.entries] == [e.source_sha1 for e in m2.entries]


# ------------------------------------------------------------------------------------------------------------------- data
def test_one_hot_and_pearson():
    x = one_hot(["ACGT", "acgn"], length=6)
    assert (
        x.shape == (2, 4, 6)
        and x[0, :, 0].tolist() == [1, 0, 0, 0]
        and x[1, :, 3].sum() == 0
        and x[0, :, 5].sum() == 0
    )
    assert pearson(np.arange(5), 2 * np.arange(5) + 1) == pytest.approx(1.0)


def test_load_labelled_refuses_unstamped_sets(tmp_path):
    p = tmp_path / "pool.npz"
    np.savez(p, sequences=np.array(["ACGT"] * 3), oracle_labels=np.arange(3, dtype=float))
    with pytest.raises(ValueError, match="oracle_id"):
        load_labelled(p)
    assert len(load_labelled(p, allow_unstamped=True)[0]) == 3
    np.savez(
        p,
        sequences=np.array(["ACGT"] * 3),
        oracle_labels=np.arange(3, dtype=float),
        oracle_id="oracle_v3",
    )
    assert load_labelled(p)[1].tolist() == [0.0, 1.0, 2.0]


# ------------------------------------------------------------------------------------------------------------------- stack
def test_stack_can_drop_an_exploded_member_without_changing_scale():
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    v = np.stack(
        [
            y + 0.3 * rng.normal(size=300),
            y + 0.3 * rng.normal(size=300),
            y + 0.3 * rng.normal(size=300),
        ]
    )
    st = fit_zscored_stack(v, y)
    broken = v.copy()
    broken[1] = np.nan
    p = st.predict(broken, active=np.array([True, False, True]))
    assert np.isfinite(p).all() and abs(p.std() - st.predict(v).std()) < 0.2 * st.predict(v).std()


def test_screen_detects_exploding_predictions():
    assert (
        exploded(np.array([1.0, np.nan]))
        and exploded(np.array([1.0, 2e3]))
        and not exploded(np.array([-5.0, 7.0]))
    )
    probe = probe_sequences()
    assert len(probe) > 1000 and all(len(x) == 200 for x in probe)
    gc = np.mean([sum(c in "GC" for c in x) / 200 for x in probe[400:700]])
    assert gc > 0.7  # the second block is GC-rich


def test_zscored_stack_is_non_negative_and_round_trips():
    rng = np.random.default_rng(0)
    y = rng.normal(size=400)
    good = y + 0.3 * rng.normal(size=400)
    bad = -y + rng.normal(size=400)  # anti-correlated: a positive stack must give it no weight
    odd_scale = (
        50 * (y + 0.5 * rng.normal(size=400)) + 7
    )  # z-scoring keeps an odd output scale harmless
    v = np.stack([good, bad, odd_scale])
    st = fit_zscored_stack(v, y)
    assert (st.coef >= 0).all() and st.coef[1] == 0 and st.coef[0] > 0 and st.coef[2] > 0
    st2 = ZStack.from_json(json.loads(json.dumps(st.to_json())))
    assert np.allclose(st.predict(v), st2.predict(v))
    assert pearson(st.predict(v), y) > pearson(good, y)


# ------------------------------------------------------------------------------------------- tiny end-to-end on the CPU
@pytest.fixture(scope="module")
def tiny():
    pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    seqs = ["".join(rng.choice(list("ACGT"), 200)) for _ in range(320)]
    y = np.array([s.count("GC") / 5.0 for s in seqs], np.float32)  # a learnable toy label
    return seqs[:256], y[:256], seqs[256:], y[256:]


def test_train_stack_save_load_and_acquisition(tiny, tmp_path):
    from albench.acquisition.ensemble_acq import EnsembleAcquisition

    tr_s, tr_y, va_s, va_y = tiny
    ens = MenuEnsemble(_toy_entries()[:3], va_s, va_y, devices=["cpu"], log=lambda *_: None)
    ens.fit(tr_s, tr_y)
    assert len(ens.members) == 3 and ens.stack is not None and (ens.stack.coef >= 0).all()
    pred = ens.predict(va_s)
    assert pred.shape == (len(va_s),) and np.isfinite(pred).all()
    assert ens.member_predictions(va_s).shape == (3, len(va_s))
    assert ens.epistemic_uncertainty(va_s).shape == (len(va_s),)
    with pytest.raises(NotImplementedError):
        ens.embed(va_s)
    ens.fit(tr_s, tr_y)  # same training set: must not retrain
    first = ens.models[
        0
    ]  # BatchBALD passes raw sequences, EnsembleAcquisition one-hot arrays: both must work
    assert np.allclose(
        ens._predict_member(first, va_s[:4]), ens._predict_member(first, one_hot(va_s[:4]))
    )
    picked = EnsembleAcquisition().select(
        ens, va_s, 5
    )  # the repo's ensemble-disagreement acquisition runs on it unchanged
    assert len(picked) == 5
    ens.save(tmp_path / "round_0" / "student_checkpoint.pt")
    back = MenuEnsemble.load(
        tmp_path / "round_0" / "student_checkpoint",
        va_s,
        va_y,
        menu=Menu(_toy_entries()),
        devices=["cpu"],
    )
    assert np.allclose(back.predict(va_s), pred, atol=1e-4)


def test_sharded_training_then_stack(tiny, tmp_path):
    tr_s, tr_y, va_s, va_y = tiny
    m = Menu(
        _toy_entries(), {"rule": "first2", "sizes": {"10000": {"k": 3, "seeds": 2, "patience": 2}}}
    )
    xs, xv = one_hot(tr_s), one_hot(va_s)
    for k in range(2):  # two shards, as two Slurm jobs would
        train_menu(
            m, 10_000, xs, tr_y, xv, va_y, tmp_path, shard=(k, 2), device="cpu", log=lambda *_: None
        )
    assert len(list((tmp_path / "members").glob("*.pt"))) == 6  # 3 configurations x 2 seeds
    before = sorted(p.name for p in (tmp_path / "members").iterdir())
    train_menu(
        m, 10_000, xs, tr_y, xv, va_y, tmp_path, device="cpu", log=lambda *_: None
    )  # resumable: nothing new is trained
    assert sorted(p.name for p in (tmp_path / "members").iterdir()) == before
    meta = stack_dir(m, tmp_path, va_y)
    assert len(meta["members"]) == 6 and meta["seeds"] == 2 and meta["val_r_ensemble"] > 0
    back = MenuEnsemble.load(tmp_path, va_s, va_y, menu=m, devices=["cpu"])
    assert np.isfinite(back.predict(va_s)).all()


def test_test_sets_load_and_evaluate(tmp_path):
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), 200)) for _ in range(50)]
    y = np.array([s.count("G") for s in seqs], np.float32)
    for name in ("genomic", "ood", "random_32k"):
        np.savez(tmp_path / f"{name}_oracle.npz", sequences=np.array(seqs), oracle_mean=y)
    sets = load_test_sets(path=tmp_path)
    assert list(sets) == ["genomic", "ood", "random_32k"] and len(sets["ood"][0]) == 50

    class Perfect:
        def predict(self, sequences):
            return np.array([s.count("G") for s in sequences], np.float32)

    res = evaluate(Perfect(), path=tmp_path)
    assert all(abs(v - 1.0) < 1e-6 for v in res.values())


def test_menu_v1c_is_the_default_and_every_configuration_is_used():
    m = load_menu()
    assert DEFAULT_MENU == "v1c" and m.name == "menu_v1c" and len(m) == 20 and not m.skipped
    assert [e.id for e in m.priority()] == m.per_size["order"] and len(
        {e.id for e in m.entries}
    ) == 20
    for size in (10_000, 30_000, 100_000):
        assert len(m.for_size(size)) == 20 and m.recipe(size)["patience"] == 16
    assert len(m.for_size(300_000)) == 12
    assert (
        len(load_menu("v0")) == 33 and len(load_menu("menu_v0").priority()) == 30
    )  # older name forms still resolve


def test_menu_v1_stays_loadable_and_has_the_same_recipe_table_as_v1c():
    v1, v1c = load_menu("v1"), load_menu("v1c")
    assert v1.name == "menu_v1" and len(v1) == 20
    assert [e.id for e in v1.entries] == [e.id for e in v1c.entries]
    assert v1.per_size == v1c.per_size
    assert sum(a.source_sha1 != b.source_sha1 for a, b in zip(v1.entries, v1c.entries)) == 5


def test_every_v1c_configuration_builds_at_the_standard_input_length():
    pytest.importorskip("torch")
    assert INPUT_LEN == 240 == input_length()
    failed = {}
    for e in load_menu("v1c").entries:
        net, info = build_net(e, max_fwd_ms=1e9)  # default seq_len = INPUT_LEN
        if net is None:
            failed[e.id] = str(info)[:80]
    assert not failed, failed


# ------------------------------------------------------------------------------------------------- the standard input context
def _train_tiny(entry, tiny, **kw):
    tr_s, tr_y, va_s, va_y = tiny
    return train_member(entry, tr_s, tr_y, va_s, va_y, device="cpu", log=lambda *_: None, **kw)


def _short(entry, epochs=2):
    return MenuEntry(
        entry.id, entry.run, "", None, None, entry.source, {**entry.train_config, "epochs": epochs}
    )


def test_encodings_have_the_standard_widths_and_real_flank(tiny):
    from albench.arch_search.flank_context import wide_onehot
    from albench.ensemble.menu import _as_window

    seqs = tiny[0][:5]
    x, xt = encode(seqs), encode_train(seqs)
    assert x.shape == (5, 4, INPUT_LEN) and xt.shape == (5, 4, INPUT_LEN + 2 * SHIFT_MAX)
    assert np.array_equal(x, wide_onehot(seqs, FLANK_CTX))
    assert np.array_equal(x[:, :, FLANK_CTX : FLANK_CTX + 200], one_hot(seqs))
    assert (x.sum(1) == 1).all()  # flank positions are real bases, not zeros
    assert (
        np.array_equal(encode(seqs, 0), one_hot(seqs))
        and encode_train(seqs, 0, 10).shape[-1] == 200
    )
    # a bare one-hot array is re-framed with the flank, and a wider one is cut back; a fitting one is untouched
    assert np.array_equal(_as_window(one_hot(seqs), FLANK_CTX), x)
    assert np.array_equal(_as_window(xt, FLANK_CTX), x) and np.array_equal(
        _as_window(x, 0), one_hot(seqs)
    )
    assert _as_window(x, FLANK_CTX).shape == x.shape
    with pytest.raises(ValueError):
        _as_window(np.zeros((2, 4, 201), np.float32), FLANK_CTX)


def test_train_member_trains_at_240_with_the_shift_done_by_the_wrapper(tiny, monkeypatch):
    torch = pytest.importorskip("torch")
    from albench.arch_search import arch_autoresearch as aa

    seen, built = [], []

    class Spy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c, self.f = torch.nn.Conv1d(4, 8, 3, padding=1), torch.nn.Linear(8, 1)

        def forward(self, x):
            seen.append((self.training, tuple(x.shape), float(x.sum(1).min())))
            return self.f(torch.relu(self.c(x)).mean(-1)).squeeze(-1)

    def fake_build(entry, *, seq_len=INPUT_LEN, **_):
        built.append(seq_len)
        return Spy(), 77

    monkeypatch.setattr(menu_mod, "build_net", fake_build)
    monkeypatch.delenv("AL_SHIFT_MAX", raising=False)
    shift_before = aa.shift
    m = _train_tiny(_toy_entries()[0], tiny, patience=3)
    assert built == [INPUT_LEN] and m.input_context == FLANK_CTX and m.input_length == INPUT_LEN
    train_batches = [t for t in seen if t[0]]
    assert train_batches and all(t[1][1:] == (4, INPUT_LEN) for t in train_batches)
    # no zero-padded shift reached the window: every position of every training window is a real base
    assert all(t[2] == 1.0 for t in seen)
    assert aa.shift is shift_before and "AL_SHIFT_MAX" not in os.environ  # overrides undone
    assert m.val_pred.shape == (len(tiny[2]),) and np.isfinite(m.val_pred).all()
    assert not any(k.startswith("net.") for k in m.state)  # the wrapper's prefix is not stored
    assert m.info["shift_max"] == SHIFT_MAX


def test_flank_ctx_zero_reproduces_the_bare_insert_recipe(tiny, monkeypatch):
    torch = pytest.importorskip("torch")
    seen, built = [], []

    class Spy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.f = torch.nn.Linear(4, 1)

        def forward(self, x):
            seen.append(tuple(x.shape))
            return self.f(x.mean(-1)).squeeze(-1)

    def fake_build(entry, *, seq_len=INPUT_LEN, **_):
        built.append(seq_len)
        return Spy(), 5

    monkeypatch.setattr(menu_mod, "build_net", fake_build)
    m = _train_tiny(_toy_entries()[0], tiny, flank_ctx=0)
    assert built == [200] and m.input_context == 0 and {s[1:] for s in seen} == {(4, 200)}
    assert m.info["shift_max"] is None


def test_tiny_member_trains_on_cpu_and_predicts_at_240(tiny):
    torch = pytest.importorskip("torch")
    tr_s, tr_y, va_s, va_y = tiny
    m = _train_tiny(_short(_toy_entries()[0]), tiny)
    assert m is not None and m.input_context == FLANK_CTX and m.epochs >= 1
    p = member_predict(m, va_s, device="cpu")  # sequences are framed for the member
    assert p.shape == (len(va_s),) and np.isfinite(p).all()
    # bare one-hot arrays, wide arrays and sequences all give the same predictions for a context member
    assert np.allclose(p, member_predict(m, one_hot(va_s), device="cpu"), atol=1e-5)
    assert np.allclose(p, member_predict(m, encode_train(va_s), device="cpu"), atol=1e-5)
    assert np.allclose(
        p, m.val_pred, atol=1e-4
    )  # validation predictions came from the centred window
    assert m.net()(torch.zeros(2, 4, INPUT_LEN)).shape == (2,)


def test_wrapper_prefixed_checkpoints_load_like_plain_ones(tiny):
    m = _train_tiny(_short(_toy_entries()[0]), tiny)
    prefixed = TrainedMember(
        m.entry,
        {f"net.{k}": v for k, v in m.state.items()},
        m.seed,
        m.val_pred,
        m.val_r,
        m.epochs,
        m.minutes,
        m.n_params,
        input_context=m.input_context,
    )
    va_s = tiny[2]
    assert np.allclose(
        member_predict(m, va_s, device="cpu"), member_predict(prefixed, va_s, device="cpu")
    )


def test_context_ensemble_saves_its_context_and_old_metadata_loads_as_200bp(tiny, tmp_path):
    tr_s, tr_y, va_s, va_y = tiny
    entries = _toy_entries()[:3]
    new = MenuEnsemble(entries, va_s, va_y, devices=["cpu"], log=lambda *_: None)
    new.fit(tr_s, tr_y)
    assert {m.input_context for m in new.members} == {FLANK_CTX}
    assert new._encode_sequences(va_s).shape == (len(va_s), 4, INPUT_LEN)
    new.save(tmp_path / "new")
    meta = json.loads((tmp_path / "new" / "ensemble.json").read_text())
    assert meta["input_context"] == FLANK_CTX and {r["input_context"] for r in meta["members"]} == {
        FLANK_CTX
    }
    import torch

    saved = torch.load(
        tmp_path / "new" / "members" / f"{new.members[0].member_id}.pt", map_location="cpu"
    )
    assert not any(
        k.startswith("net.") for k in saved
    )  # a checkpoint is the bare network's weights
    back = MenuEnsemble.load(tmp_path / "new", va_s, va_y, menu=Menu(entries), devices=["cpu"])
    assert {m.input_context for m in back.members} == {FLANK_CTX} and back.flank_ctx == FLANK_CTX
    assert np.allclose(back.predict(va_s), new.predict(va_s), atol=1e-4)

    # an ensemble from before the standard: members trained on the bare insert, metadata without any input_context key
    old = MenuEnsemble(entries, va_s, va_y, devices=["cpu"], flank_ctx=0, log=lambda *_: None)
    old.fit(tr_s, tr_y)
    old.save(tmp_path / "old")
    f = tmp_path / "old" / "ensemble.json"
    meta = json.loads(f.read_text())
    meta.pop("input_context")
    for r in meta["members"]:
        r.pop("input_context")
        r.pop("shift_max")
    f.write_text(json.dumps(meta))
    loaded = MenuEnsemble.load(tmp_path / "old", va_s, va_y, menu=Menu(entries), devices=["cpu"])
    assert {m.input_context for m in loaded.members} == {0} and loaded.flank_ctx == 0
    assert loaded._encode_sequences(va_s).shape == (len(va_s), 4, 200)
    assert loaded.member_predictions(va_s).shape == (3, len(va_s))
    assert np.allclose(loaded.predict(va_s), old.predict(va_s), atol=1e-4)
    assert loaded.members[0].net()(torch.zeros(2, 4, 200)).shape == (2,)  # rebuilt at 200 positions


def test_mixed_context_ensemble_encodes_per_member_and_screens_with_the_right_probes(
    tiny, monkeypatch
):
    tr_s, tr_y, va_s, va_y = tiny
    entries = _toy_entries()[:2]
    kw = dict(devices=["cpu"], log=lambda *_: None)
    new = MenuEnsemble(entries, va_s, va_y, **kw)
    new.fit(tr_s, tr_y)
    old = MenuEnsemble(entries, va_s, va_y, flank_ctx=0, **kw)
    old.fit(tr_s, tr_y)
    mixed = MenuEnsemble(entries, va_s, va_y, **kw)
    mixed.members = [new.members[0], old.members[1]]
    mixed.fit_stack()
    enc = mixed._encode_sequences(va_s[:6])
    assert isinstance(enc, dict) and {c: a.shape for c, a in enc.items()} == {
        FLANK_CTX: (6, 4, INPUT_LEN),
        0: (6, 4, 200),
    }
    mp = mixed.member_predictions(va_s)
    assert mp.shape == (2, len(va_s)) and np.isfinite(mp).all()
    assert np.allclose(mp[0], member_predict(new.members[0], va_s, device="cpu"))
    assert np.allclose(mp[1], member_predict(old.members[1], va_s, device="cpu"))
    for j, m in enumerate(
        mixed.members
    ):  # the per-context encodings are routed to the right member
        assert np.allclose(mixed._predict_member(m, enc), mp[j][:6], atol=1e-5)
        assert np.allclose(mixed._predict_member(m, va_s[:6]), mp[j][:6], atol=1e-5)
    assert mixed.predict(va_s).shape == (len(va_s),)

    widths = []
    real = menu_mod.member_predict
    monkeypatch.setattr(
        menu_mod,
        "member_predict",
        lambda m, x, **k: (widths.append((m.input_context, x.shape[-1])), real(m, x, **k))[1],
    )
    assert mixed.screen() == []
    assert sorted(widths) == [(0, 200), (FLANK_CTX, INPUT_LEN)]


def test_evaluate_and_probes_work_on_a_context_ensemble(tiny, tmp_path):
    tr_s, tr_y, va_s, va_y = tiny
    ens = MenuEnsemble(_toy_entries()[:2], va_s, va_y, devices=["cpu"], log=lambda *_: None)
    ens.fit(tr_s, tr_y)
    assert ens.screened_out == [] and len(probe_sequences()) > 1000
    assert ens.member_predictions(probe_sequences()[:50]).shape == (2, 50)
    for name in ("genomic", "ood", "random_32k"):
        np.savez(tmp_path / f"{name}_oracle.npz", sequences=np.array(va_s), oracle_mean=va_y)
    res = evaluate(ens, path=tmp_path)
    assert list(res) == ["genomic", "ood", "random_32k"] and all(
        np.isfinite(v) for v in res.values()
    )
    assert res["genomic"] == pytest.approx(pearson(ens.predict(va_s), va_y))
