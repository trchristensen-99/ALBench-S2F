"""The menu of configurations and the ensemble built from it (albench.ensemble.menu)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from albench.ensemble.menu import (
    Menu,
    MenuEnsemble,
    MenuEntry,
    load_labelled,
    load_menu,
    one_hot,
    pearson,
    stack_dir,
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


def test_priority_order_covers_each_entry_once_and_per_size_table_is_valid():
    m = load_menu("v0")
    order = m.priority()
    assert sorted(e.id for e in order) == sorted(e.id for e in m.entries)
    for size in (10_000, 30_000, 100_000):
        row = m.recipe(size)
        assert 1 <= row["k"] <= len(m) and row["seeds"] >= 1
        assert len(m.for_size(size)) == row["k"]
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
