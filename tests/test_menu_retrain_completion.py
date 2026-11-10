"""A menu shard may only read as done when nothing is left to try, and may never loop.

Drives the real ``orchestration/menu_retrain.py`` ``main()`` through several simulated dispatcher jobs, with
the data loaders, the builder and the trainer replaced by stubs, so the control flow that decides
COMPLETE / DEGRADED / resubmit is the code that runs on the cluster.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mr = _load("menu_retrain_under_test", "orchestration/menu_retrain.py")


def test_failure_kind():
    assert mr.failure_kind("build: forward too slow: 9000ms > 8000ms") == "filtered"
    assert mr.failure_kind("build: too many parameters: 50,000,000 > 40,000,000") == "filtered"
    assert mr.failure_kind("build: banned name: getattr") == "final"
    assert mr.failure_kind("build: forward failed: OutOfMemoryError: CUDA out of memory") == "retry"
    assert mr.failure_kind("train: OutOfMemoryError: CUDA out of memory. Tried") == "retry"
    assert mr.failure_kind("train: non-finite loss ep3") == "retry"
    assert mr.failure_kind("exploded predictions on test_pred_x") == "final"
    assert mr.failure_kind("aux: wrong head") == "final"


def test_history_resets_once_per_tag(tmp_path):
    p = tmp_path / "failed.jsonl"
    rows = [
        {"id": "a", "why": "train: x"},
        {"id": "a", "why": "train: y"},
        {"id": "a", "reset": "T1"},
        {"id": "a", "why": "train: z"},
        {"id": "b", "why": "build: q"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    hist, reset = mr.failure_history(p, "T1")
    assert hist == {"a": ["train: z"], "b": ["build: q"]} and reset == {"a"}
    assert mr.failure_history(p, "T2")[1] == set()


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))


def _install_stubs(monkeypatch, tmp_path, outcomes: dict):
    """outcomes: menu entry id -> list of results per attempt ('ok' | 'oom' | 'build' | 'slow')."""
    rng = np.random.default_rng(0)

    def seqs(n):
        return ["".join(rng.choice(list("ACGT"), 200)) for _ in range(n)]

    shs = types.ModuleType("scaling_hp_search")
    shs.load_chr_train_pool = lambda *a, **k: (
        seqs(40),
        rng.normal(size=40),
        seqs(10),
        rng.normal(size=10),
    )
    shs.load_chr_test_genomic = lambda: (seqs(10), rng.normal(size=10), rng.normal(size=10))
    shs.load_all_test_sets = lambda: {}
    shs._atomic_savez = lambda p, **kw: np.savez(p, **kw)
    shs._atomic_write_text = lambda p, t: Path(p).write_text(t)
    monkeypatch.setitem(sys.modules, "scaling_hp_search", shs)

    calls = {k: 0 for k in outcomes}
    import albench.arch_search.arch_autoresearch as aa
    import albench.arch_search.validate as va

    def current(src):
        return outcomes[src][min(calls[src], len(outcomes[src]) - 1)]

    def build(src, **kw):
        r = current(src)
        v = types.SimpleNamespace(n_params=1, reason="")
        if r in ("build", "slow"):
            calls[src] += 1
            v.reason = "banned name: getattr" if r == "build" else "forward too slow: 9e3ms > 8e3ms"
            return None, v
        net = _Net()
        net.src = src
        return net, v

    def train(net, *a, **k):
        src = net.net.src if hasattr(net, "net") else net.src
        r = current(src)
        calls[src] += 1
        if r == "oom":
            return None, {"error": "OutOfMemoryError: CUDA out of memory"}, None
        return net, {"best_val_mse": 0.1, "epochs_trained": 1}, "adamw"

    monkeypatch.setattr(va, "build_and_check", build)
    monkeypatch.setattr(aa, "train_one", train)
    monkeypatch.setattr(aa, "predict_rc", lambda m, X: np.zeros(len(X), np.float32))
    monkeypatch.setattr(aa, "pear", lambda a, b: 0.5)
    menu = [
        {
            "id": e,
            "source": e,
            "train_config": {"epochs": 1, "lr": 1e-3, "weight_decay": 0.0, "batch_size": 8},
        }
        for e in outcomes
    ]
    (tmp_path / "menu.json").write_text(json.dumps(menu))
    return calls


def _job(monkeypatch, tmp_path, *extra):
    out = tmp_path / "out"
    argv = [
        "menu_retrain.py",
        "--menu",
        str(tmp_path / "menu.json"),
        "--reservoir_cache",
        "x",
        "--D",
        "40",
    ]
    monkeypatch.setattr(sys, "argv", argv + ["--out_dir", str(out), "--shard", "0/1", *extra])
    mr.main()
    return out


def _state(out):
    return (out / "COMPLETE_0of1").exists(), (out / "DEGRADED_0of1.json").exists()


def test_all_trained_writes_complete_only(monkeypatch, tmp_path):
    _install_stubs(monkeypatch, tmp_path, {"a": ["ok"], "b": ["ok"]})
    out = _job(monkeypatch, tmp_path)
    assert _state(out) == (True, False)
    assert (out / "COMPLETE_0of1").read_text().startswith("done=2 failed=0")


def test_transient_failure_is_retried_then_recovered(monkeypatch, tmp_path):
    calls = _install_stubs(monkeypatch, tmp_path, {"a": ["ok"], "b": ["oom", "ok"]})
    out = _job(monkeypatch, tmp_path)
    assert _state(out) == (False, False)  # no marker: the dispatcher resubmits
    out = _job(monkeypatch, tmp_path)
    assert (
        _state(out) == (True, False) and calls["a"] == 1
    )  # a was not retrained (its _meta.json exists)


def test_persistent_failure_is_bounded_and_recorded(monkeypatch, tmp_path):
    calls = _install_stubs(monkeypatch, tmp_path, {"a": ["ok"], "b": ["oom"]})
    for _ in range(2):
        out = _job(monkeypatch, tmp_path)
    assert _state(out) == (True, True)  # terminal after --max_attempts=2, loss recorded
    deg = json.loads((out / "DEGRADED_0of1.json").read_text())
    assert [x["entry"] for x in deg["lost"]] == ["b"] and deg["lost"][0]["attempts"] == 2
    _job(monkeypatch, tmp_path)  # a stray resubmit must not train b again
    assert calls["b"] == 2


def test_deterministic_failure_closes_at_once_and_filter_is_not_a_loss(monkeypatch, tmp_path):
    calls = _install_stubs(monkeypatch, tmp_path, {"a": ["build"], "b": ["slow"], "c": ["ok"]})
    out = _job(monkeypatch, tmp_path)
    assert _state(out) == (True, True)
    deg = json.loads((out / "DEGRADED_0of1.json").read_text())
    assert [x["entry"] for x in deg["lost"]] == ["a"] and [x["entry"] for x in deg["filtered"]] == [
        "b"
    ]
    assert "lost=1 filtered=1" in (out / "COMPLETE_0of1").read_text()
    assert calls["a"] == 1


def test_only_filtered_is_complete_not_degraded(monkeypatch, tmp_path):
    _install_stubs(monkeypatch, tmp_path, {"a": ["slow"], "b": ["ok"]})
    assert _state(_job(monkeypatch, tmp_path)) == (True, False)


def test_retry_failed_reruns_once_per_tag(monkeypatch, tmp_path):
    calls = _install_stubs(monkeypatch, tmp_path, {"a": ["build", "ok"], "b": ["ok"]})
    out = _job(monkeypatch, tmp_path)
    assert _state(out) == (True, True)
    out = _job(monkeypatch, tmp_path, "--retry_failed", "fix1")  # entry fixed: retried, recovered
    assert _state(out) == (True, False) and (out / "RETRY_fix1_0of1").exists()
    assert calls["a"] == 2 and calls["b"] == 1


def test_retry_tag_cannot_loop(monkeypatch, tmp_path):
    calls = _install_stubs(monkeypatch, tmp_path, {"a": ["oom"]})
    for _ in range(2):
        _job(monkeypatch, tmp_path)
    for _ in range(
        4
    ):  # the re-run cell gets resubmitted; the tag resets the count only the first time
        out = _job(monkeypatch, tmp_path, "--retry_failed", "t")
    assert calls["a"] == 4 and (out / "RETRY_t_0of1").exists()


def test_dispatcher_reports_degraded(tmp_path, monkeypatch):
    disp = _load("dispatcher_under_test", "orchestration/dispatcher.py")
    monkeypatch.setattr(disp, "R", tmp_path)
    (tmp_path / "o").mkdir()
    (tmp_path / "o" / "COMPLETE_1of4").write_text("done=3 failed=0\n")
    c = {"done": {"type": "file", "path": "o/COMPLETE_1of4"}}
    assert disp.is_done(c) and not disp.is_degraded(c)  # a pre-existing marker: done, not counted
    (tmp_path / "o" / "DEGRADED_1of4.json").write_text("{}")
    assert disp.is_done(c) and disp.is_degraded(c)
    assert not disp.is_degraded(
        {"done": {"type": "hab_keys", "file": "x", "cfg": 0, "seeds": [], "arms": []}}
    )


def test_resume_keeps_flank_context_and_legacy_dirs_stay_legacy(monkeypatch, tmp_path):
    import albench.arch_search.arch_autoresearch as aa

    _install_stubs(monkeypatch, tmp_path, {"a": ["ok"], "b": ["oom", "ok"]})
    seen = []
    inner = aa.train_one

    def spy(net, *a, **k):
        seen.append(hasattr(net, "net"))  # the WideCrop context wrapper holds the model as .net
        return inner(net, *a, **k)

    monkeypatch.setattr(aa, "train_one", spy)
    _job(monkeypatch, tmp_path)
    _job(monkeypatch, tmp_path)  # the resumed job: a model exists already
    assert seen == [True, True, True]

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "r00_menu_000_meta.json").write_text(
        "{}"
    )  # trained before context existed, no record
    seen.clear()
    argv = ["menu_retrain.py", "--menu", str(tmp_path / "menu.json"), "--reservoir_cache", "x"]
    monkeypatch.setattr(sys, "argv", argv + ["--D", "40", "--out_dir", str(legacy)])
    mr.main()
    assert seen and not any(seen)
