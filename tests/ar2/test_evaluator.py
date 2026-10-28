"""Package A: evaluator tests (spec T2, T3, T13, T14, the loss registry, failure statuses, determinism)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from a_helpers import TINY_CNN, make_spec, synth_data, write_candidate, write_pool

from ar2.data.encode import encode, revcomp_onehot
from ar2.evaluator import metrics as M
from ar2.evaluator.losses import build_loss, loss_names
from ar2.evaluator.stress import (
    dinuc_shuffle,
    load_stress_set,
    make_stress_set,
    save_stress_set,
    stress_check,
)
from ar2.evaluator.train import lr_factor, predict_z, run


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ev")
    tr, ytr, va, yva = synth_data()
    return {
        "tmp": tmp,
        "data": (tr, ytr, va, yva),
        "tm": write_pool(tmp, tr, ytr, "train", "train"),
        "vm": write_pool(tmp, va, yva, "val_loop", "val"),
        "cnn": write_candidate(tmp, "cnn", TINY_CNN),
    }


def _run(world, name, code_dir=None, **over):
    tmp = world["tmp"] / name
    tmp.mkdir(exist_ok=True)
    spec = make_spec(
        tmp,
        code_dir or world["cnn"],
        over.pop("tm", world["tm"]),
        over.pop("vm", world["vm"]),
        run_id=name,
        **over,
    )
    return run(spec), spec


def test_smoke_end_to_end_and_files(world):
    m, spec = _run(world, "smoke", recipe={"max_epochs": 6, "patience": 6})
    out = Path(spec["out_dir"])
    assert m["status"] == "ok" and m["val_r"] > 0.3 and m["stress"]["exploded"] is False
    for f in ("metrics.json", "curves.csv", "log.txt", "preds_val_loop.npy"):
        assert (out / f).exists(), f
    on_disk = json.loads((out / "metrics.json").read_text())
    for k in (
        "run_id",
        "status",
        "stop_reason",
        "val_r",
        "val_mse",
        "val_spearman",
        "train_r",
        "best_epoch",
        "epochs_run",
        "still_improving",
        "pred_sd_ratio",
        "mse_by_activity_quintile",
        "mse_by_gc_tercile",
        "stress",
        "throughput_seq_per_s",
        "peak_mem_gb",
        "wall_time_s",
        "n_params",
        "cost_equiv",
        "cap_s_used",
    ):
        assert k in on_disk, k
    assert len(on_disk["mse_by_activity_quintile"]) == 5 and len(on_disk["mse_by_gc_tercile"]) == 3
    assert "val_select_r" not in on_disk  # ordinary specs never touch val_select
    assert np.load(out / "preds_val_loop.npy").dtype == np.float32
    assert M.pearson(np.load(out / "preds_val_loop.npy"), world["data"][3]) == pytest.approx(
        m["val_r"], abs=1e-6
    )  # stored predictions are on the original scale


def test_same_seed_same_result_different_seed_differs(world):
    a, _ = _run(world, "seed_a", recipe={"max_epochs": 3, "patience": 3})
    b, _ = _run(world, "seed_b", recipe={"max_epochs": 3, "patience": 3})
    c, _ = _run(world, "seed_c", seed=1, recipe={"max_epochs": 3, "patience": 3})
    assert a["val_r"] == b["val_r"]
    assert a["val_r"] != c["val_r"]


def test_t2_label_scaling_invariance(tmp_path):
    """z-scoring with training statistics makes training independent of the label scale and offset. A power-of-two rescale is exact in float32, so the
    runs agree to rounding; an arbitrary shift and scale changes the labels' float32 rounding, which training amplifies, so only a loose agreement is required."""
    tr, ytr, va, yva = synth_data(seed=3)
    cd = write_candidate(tmp_path, "cnn", TINY_CNN)
    res = []
    for k, (scale, shift) in enumerate([(1.0, 0.0), (4.0, 0.0), (7.5, -40.0)]):
        d = tmp_path / f"s{k}"
        d.mkdir()
        tm = write_pool(d, tr, ytr * scale + shift, "train", "tr")
        vm = write_pool(d, va, yva * scale + shift, "val_loop", "va")
        res.append(run(make_spec(d, cd, tm, vm, run_id=f"t2_{k}", recipe={"max_epochs": 4})))
    assert res[0]["val_r"] == pytest.approx(res[1]["val_r"], abs=1e-5)
    assert res[1]["val_mse"] == pytest.approx(
        res[0]["val_mse"] * 16, rel=1e-3
    )  # reported on the original scale
    assert res[2]["val_r"] == pytest.approx(res[0]["val_r"], abs=0.03)


ASYM = """
import torch
import torch.nn as nn

HPARAMS = {}


class Net(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.w = nn.Parameter(torch.linspace(-1, 1, 200)[None, :])
        self.k = nn.Linear(200, out_dim)

    def forward(self, x):
        return self.k(x[:, 0, :] * self.w)   # reads only the A channel, position dependent: strand asymmetric


def build_model(out_dim, **hp):
    return Net(out_dim)
"""


def test_t3_reverse_complement_averaging(tmp_path):
    cd = write_candidate(tmp_path, "asym", ASYM)
    from ar2.evaluator.train import build_model

    torch.manual_seed(0)
    model = build_model(cd, 1, {})
    seqs, *_ = synth_data(30, 2)
    X = torch.from_numpy(encode(seqs))
    loss = build_loss({"name": "mse"})
    with torch.no_grad():
        fwd, rev = model(X)[:, 0].numpy(), model(revcomp_onehot(X))[:, 0].numpy()
    assert np.abs(fwd - rev).max() > 1e-3  # the toy model really is strand asymmetric
    assert np.allclose(predict_z(model, X, loss, rc_average=True), (fwd + rev) / 2, atol=1e-6)
    assert np.allclose(predict_z(model, X, loss, rc_average=False), fwd, atol=1e-6)


def _reference(curve_key, vals):
    return {curve_key: {"0.5": vals[0], "0.75": vals[1]}}


def test_t13_early_termination_same_schedule_only(world):
    rec = {"max_epochs": 8, "patience": 8, "schedule": "onecycle"}
    high = {"checkpoints": [0.5, 0.75], "margin": 0.03}
    # reference from cosine runs only: a one-cycle run is never compared with it, however poor it looks at 50%
    m1, _ = _run(
        world,
        "et_cos",
        recipe=rec,
        early_term={**high, "reference": _reference("cosine_warmup", [0.99, 0.99])},
    )
    assert m1["status"] == "ok" and m1["epochs_run"] == 8
    # same-schedule reference far above this run: terminated at the 50% checkpoint
    m2, _ = _run(
        world,
        "et_same",
        recipe=rec,
        early_term={**high, "reference": _reference("onecycle", [0.99, 0.99])},
    )
    assert (
        m2["status"] == "early_stopped_by_tuner"
        and m2["epochs_run"] == 4
        and m2["stop_reason"] == "tuner_early_term"
    )
    assert (
        m2["cost_equiv"] == pytest.approx(0.5) and m2["val_r"] is not None
    )  # best so far is still reported
    # never in the first look or in confirmation
    for purpose in ("first_look", "confirm"):
        m3, _ = _run(
            world,
            f"et_{purpose}",
            recipe=rec,
            purpose=purpose,
            early_term={**high, "reference": _reference("onecycle", [0.99, 0.99])},
        )
        assert m3["status"] == "ok" and m3["epochs_run"] == 8


LOWCOMP = """
import torch
import torch.nn as nn

HPARAMS = {"width": 16}


class Net(nn.Module):
    def __init__(self, out_dim, width):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(4, width, 9, padding=4), nn.BatchNorm1d(width), nn.ReLU())
        self.head = nn.Linear(width, out_dim)

    def forward(self, x):
        o = self.head(self.conv(x).mean(-1))
        same = (x[:, :, 1:] * x[:, :, :-1]).sum(1).mean(-1)      # fraction of adjacent identical bases
        return o + 1e6 * (same > 0.5).float()[:, None]           # absurd output on homopolymers


def build_model(out_dim, **hp):
    return Net(out_dim, **hp)
"""


def test_t14_stress_screen_marks_exploded(world):
    cd = write_candidate(world["tmp"], "lowcomp", LOWCOMP)
    m, _ = _run(world, "exploded", code_dir=cd, recipe={"max_epochs": 3})
    assert (
        m["status"] == "exploded" and m["stress"]["exploded"] is True and m["val_r"] is not None
    )  # stays in the ledger with its score


def test_stress_set_and_rule(tmp_path):
    seqs = synth_data(10, 120)[2]
    s1, s2 = make_stress_set(seqs), make_stress_set(seqs)
    assert len(s1) == 4000 and s1 == s2 and all(len(s) == 200 for s in s1)
    gc = [(s.count("G") + s.count("C")) / 200 for s in s1[1000:1500]]
    assert 0.1 < np.mean(gc) < 0.3
    save_stress_set(tmp_path / "s" / "stress.npy", s1)
    assert load_stress_set(tmp_path / "s" / "stress.npy") == s1
    assert not stress_check(np.array([0.0, 9.9]))["exploded"]
    assert (
        stress_check(np.array([0.0, 10.5]))["exploded"]
        and stress_check(np.array([np.nan]))["exploded"]
    )
    rng = np.random.default_rng(0)
    sh = dinuc_shuffle(seqs[0], rng)
    pairs = lambda s: sorted(s[i : i + 2] for i in range(len(s) - 1))  # noqa: E731
    assert (
        pairs(sh) == pairs(seqs[0])
        and sh[0] == seqs[0][0]
        and sh[-1] == seqs[0][-1]
        and sh != seqs[0]
    )


@pytest.mark.parametrize(
    "loss",
    [
        {"name": "mse"},
        {"name": "huber", "delta": 1.0},
        {"name": "hetero"},
        {"name": "corr"},
        {"name": "binned"},
    ],
)
def test_every_loss_trains(world, loss):
    m, _ = _run(world, f"loss_{loss['name']}", recipe={"loss": loss, "max_epochs": 3, "lr": 0.003})
    assert m["status"] == "ok" and m["val_r"] > 0.0, m


def test_loss_registry_shapes():
    assert {"mse", "huber", "hetero", "corr", "binned"} <= set(loss_names())
    for name, od in (("mse", 1), ("hetero", 2), ("binned", 32)):
        loss = build_loss({"name": name}, n_tasks=2)
        assert loss.out_dim == od * 2
        out = torch.randn(8, loss.out_dim, requires_grad=True)
        v = loss(out, torch.randn(8, 2))
        v.backward()
        assert v.ndim == 0 and torch.isfinite(v) and loss.predict(out).shape == (8, 2)


def test_failure_statuses(world):
    nan = """
import torch
import torch.nn as nn

HPARAMS = {}


class Net(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.p = nn.Parameter(torch.ones(out_dim))

    def forward(self, x):
        return self.p[None, :] * x.mean((1, 2))[:, None] * float("nan")


def build_model(out_dim, **hp):
    return Net(out_dim)
"""
    m, _ = _run(world, "diverged", code_dir=write_candidate(world["tmp"], "nan", nan))
    assert m["status"] == "diverged" and m["val_r"] is None
    oom = nan.replace(
        'return self.p[None, :] * x.mean((1, 2))[:, None] * float("nan")',
        'raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1 GiB")',
    )
    m, spec = _run(world, "oom", code_dir=write_candidate(world["tmp"], "oom", oom))
    assert m["status"] == "oom" and (Path(spec["out_dir"]) / "metrics.json").exists()


def test_time_cap_marks_censored(world):
    m, _ = _run(world, "capped", time_cap_s=1e-6, recipe={"max_epochs": 10, "patience": 10})
    assert (
        m["status"] == "time_cap"
        and m["censored"] is True
        and m["stop_reason"] == "time_cap"
        and m["val_r"] is not None
        and m["epochs_run"] == 1
    )
    m2, _ = _run(world, "capped_fast", time_cap_s=1e-6, speed_factor=1.0, recipe={"max_epochs": 2})
    assert m2["cap_s_used"] == pytest.approx(1e-6)


def test_overlap_is_an_error_status_not_a_crash(tmp_path):
    tr, ytr, va, yva = synth_data(60, 20)
    tm = write_pool(tmp_path, tr, ytr, "train", "tr")
    vm = write_pool(tmp_path, va[:-1] + [tr[0]], yva, "val_loop", "va")
    cd = write_candidate(tmp_path, "cnn", TINY_CNN)
    m = run(make_spec(tmp_path, cd, tm, vm, run_id="leak"))
    assert m["status"] == "error" and "occur" in m["error"]


def test_val_select_only_when_the_spec_carries_it(tmp_path):
    tr, ytr, va, yva = synth_data(100, 60)
    tm = write_pool(tmp_path, tr, ytr, "train", "tr")
    vm = write_pool(tmp_path, va[:30], yva[:30], "val_loop", "va")
    sm = write_pool(tmp_path, va[30:], yva[30:], "val_select", "sel")
    cd = write_candidate(tmp_path, "cnn", TINY_CNN)
    m = run(
        make_spec(
            tmp_path,
            cd,
            tm,
            vm,
            run_id="final",
            val_select_manifest=sm,
            purpose="final",
            recipe={"max_epochs": 3},
        )
    )
    assert m["val_select_r"] is not None and math.isfinite(m["val_select_r"])


def test_lr_schedules():
    T = 1000
    assert lr_factor("onecycle", 0, T, 0.05) == pytest.approx(1 / 25)
    assert lr_factor("onecycle", int(0.05 * T), T, 0.05) == pytest.approx(1.0, abs=1e-3)
    assert lr_factor("onecycle", T, T, 0.05) < 1e-3
    assert lr_factor("cosine_warmup", 0, T, 0.1) <= 1e-3 + 1e-9
    assert lr_factor("cosine_warmup", 100, T, 0.1) == pytest.approx(1.0, abs=1e-3)
    assert lr_factor("cosine_warmup", T, T, 0.1) <= 1e-3 + 1e-9


def test_metrics_helpers():
    y = np.arange(100.0)
    assert M.pearson(y, 2 * y + 1) == pytest.approx(1.0) and M.spearman(y, y**3) == pytest.approx(
        1.0
    )
    assert M.spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)
    assert M.pred_sd_ratio(0.5 * y, y) == pytest.approx(0.5)
    assert len(M.mse_by_activity_quintile(y, y + 1)) == 5 and M.mse_by_activity_quintile(y, y + 1)[
        0
    ] == pytest.approx(1.0)
    assert M.still_improving([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) and not M.still_improving(
        [0.6, 0.6, 0.6, 0.6, 0.6]
    )
